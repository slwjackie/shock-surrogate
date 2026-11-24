"""CPU cost of returning a checked star depth for independent wet Riemann faces.

Run from the repo root with python -m shallow_water.benchmark. On Linux this
pins the process to one of the allowed cores. Use --cpu to select a core.
"""
import argparse
import ctypes
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter_ns

import numpy as np

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent / "reports" / "shallow_water" / "benchmark.json"
METHODS = ("mean", "two_shock", "nn_initial", "nn_hard", "nn_uniform", "two_rarefaction", "rr_newton_filtered")
CLASSICAL = ("mean", "two_shock", "two_rarefaction", "rr_newton_filtered")
SEEDS = {"iid": 50001, "shift": 50002, "smooth": 50003}
TOLERANCES = (0.01, 0.001)
FLAGS = ["-std=c11", "-O3", "-march=native", "-fno-fast-math", "-ffp-contract=off", "-fPIC", "-shared"]
G = 9.81


def build_solver(directory, compiler="gcc"):
    """Build the same native kernel for benchmark and independent tests."""
    binary = Path(directory) / "shallow_water.so"
    subprocess.run([compiler, *FLAGS, str(HERE / "solver.c"), "-lm", "-o", str(binary)], check=True)
    lib = ctypes.CDLL(str(binary.resolve()))
    faces = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS")
    depths = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    counters = np.ctypeslib.ndpointer(dtype=np.int32, ndim=2, flags="C_CONTIGUOUS")
    lib.sw_solve_batch.argtypes = [ctypes.c_int, faces, ctypes.c_double, ctypes.c_double,
                                  ctypes.c_int, depths, counters]
    lib.sw_solve_batch.restype = ctypes.c_int
    return lib


def speed_comparison(samples):
    """Paired timing ratios. Each round includes all methods on the same faces."""
    times = {name: np.asarray(v) for name, v in samples.items()}
    best = min(CLASSICAL, key=lambda name: np.median(times[name]))
    return {"fastest_classical": best, "classical_over_method": {
        name: {"ratio_of_medians": float(np.median(times[best]) / np.median(t)),
               "paired_p25": float(np.quantile(times[best] / t, 0.25)),
               "paired_p75": float(np.quantile(times[best] / t, 0.75))}
        for name, t in times.items()}}


def time_batch(lib, faces, tol, rounds, rng):
    n = len(faces)
    result = np.empty(n)
    stats = np.empty((n, 4), dtype=np.int32)
    samples = {method: [] for method in METHODS}
    for run in range(-2, rounds):
        for mode in rng.permutation(len(METHODS)):
            mode = int(mode)
            start = perf_counter_ns()
            failures = lib.sw_solve_batch(n, faces, G, tol, mode, result, stats)
            elapsed = perf_counter_ns() - start
            if failures or not np.isfinite(result).all():
                raise RuntimeError(f"Native {METHODS[mode]} returned {failures} failures or NaN")
            if run >= 0:
                samples[METHODS[mode]].append(elapsed / n)
    return samples


def check_batch(lib, faces, tol, expected, audit_indices):
    from .data import decimal_audit

    n = len(faces)
    result = np.empty(n)
    stats = np.empty((n, 4), dtype=np.int32)
    summary = {}
    corrected = {}
    for mode, name in enumerate(METHODS):
        failures = lib.sw_solve_batch(n, faces, G, tol, mode, result, stats)
        if failures or not np.all(np.isfinite(result) & (result > 0)):
            raise RuntimeError(f"{name}: {failures} failures or inadmissible output")
        route, newton, bisect, checks = stats.T
        if not np.isin(route, (0, 1, 2)).all():
            raise RuntimeError(f"{name}: unexpected route codes")
        errors = np.abs(result - expected)
        # The tolerance is enforced independently using high precision on a random
        # subset, the most extreme result, and any apparent binary64 violation.
        borderline = np.flatnonzero(errors >= tol)
        indices = np.unique(np.r_[audit_indices, np.argmax(errors), borderline])
        for i in indices:
            if not decimal_audit(faces[i], result[i], tol, G):
                raise AssertionError(f"{name}: Decimal audit at index {i}, depth {result[i]}, tol {tol}")
        corrected[name] = (route == 2).copy()
        analytic = int(np.sum(route == 0))
        direct = int(np.sum(route == 1))
        fallback = int(np.sum(route == 2))
        tried = direct + fallback
        summary[name] = {
            "analytic_cases": analytic, "direct_cases": direct, "corrected_cases": fallback,
            "direct_rate_excluding_analytic": direct / tried if tried else None,
            "corrected_rate_all": fallback / n,
            "newton_mean": float(newton.mean()), "bisection_mean": float(bisect.mean()),
            "correction_updates_p95": float(np.quantile(newton + bisect, 0.95)),
            "verifier_calls_mean": float(checks.mean()),
            "depths_sha256": hashlib.sha256(result.tobytes()).hexdigest(),
            "error_mean_m": float(errors.mean()), "error_p95_m": float(np.quantile(errors, 0.95)),
            "error_max_m": float(errors.max()), "high_precision_audits": len(indices),
            "violations_in_binary64_reference_check": int((errors > tol).sum()), "failed_decimal_audits": 0}
    paired = {}
    hard = corrected["nn_hard"]
    for control in ("nn_initial", "nn_uniform"):
        base = corrected[control]
        paired[control] = {
            "correction_saved_by_hard": int((base & ~hard).sum()),
            "correction_added_by_hard": int((~base & hard).sum()),
            "net_corrections_saved_by_hard": int(base.sum() - hard.sum())}
    return summary, paired


def run(out, n, rounds, audits, cpu=None):
    from .data import sample, reference

    if n < 1 or rounds < 1 or not 0 <= audits <= n:
        raise ValueError("Require positive n and rounds, 0 <= audits <= n")
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        allowed = os.sched_getaffinity(0)
        chosen = max(allowed) if cpu is None else cpu
        os.sched_setaffinity(0, {chosen})
        affinity = sorted(os.sched_getaffinity(0))
    elif cpu is not None:
        raise ValueError("CPU affinity is unavailable on this system")
    compiler = "gcc"
    cpu_info = Path("/proc/cpuinfo")
    model = (next((line.split(":", 1)[1].strip() for line in cpu_info.read_text().splitlines()
                   if line.startswith("model name")), platform.machine())
             if cpu_info.exists() else platform.machine())
    report = {
        "cases_per_split": n, "gravity_m_s2": G, "abs_tolerance_m": TOLERANCES, "seeds": SEEDS,
        "python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
        "cpu_model": model, "cpu_affinity": affinity, "compiler": subprocess.check_output(
            [compiler, "--version"], text=True).splitlines()[0], "flags": FLAGS, "rounds": rounds,
        "warmup_rounds": 2, "timing_schedule_seed": 12934, "decimal_selection_seed": 83841,
        "timing_scope": ("Nanoseconds per face, warmed native scalar C loop on distinct faces. "
                         "Includes routing, analytic shortcut, inference for neural routes, certification, "
                         "all corrections and writing results/counters, plus one ctypes call per batch. "
                         "Excludes file reads, compilation, training, data generation and reference audits. "
                         "Intra-round method order is randomized. Aggregation is median per batch divided by case count."),
        "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in
                   [HERE / "solver.c", HERE / "certificate.h", HERE / "model.h", HERE / "data.py"]},
        "splits": {}, "equal_split_mix": {}}
    rng = np.random.default_rng(12934)
    audit_rng = np.random.default_rng(83841)
    with TemporaryDirectory() as tmp:
        lib = build_solver(tmp, compiler)
        for kind, seed in SEEDS.items():
            faces = np.ascontiguousarray(sample(n, seed, split=kind), dtype=np.float64)
            if faces.shape != (n, 4):
                raise RuntimeError(f"{kind}: invalid face array {faces.shape}")
            expected = reference(faces, g=G)
            if expected.shape != (n,) or not np.all(np.isfinite(expected) & (expected > 0)):
                raise RuntimeError(f"{kind}: invalid reference depths")
            indices = audit_rng.choice(n, audits, replace=False)
            split = {"input_sha256": hashlib.sha256(faces.tobytes()).hexdigest(), "tolerances": {}}
            report["splits"][kind] = split
            for tol in TOLERANCES:
                methods, paired = check_batch(lib, faces, tol, expected, indices)
                times = time_batch(lib, faces, tol, rounds, rng)
                for name, values in times.items():
                    methods[name]["raw_ns_per_face"] = values
                    methods[name]["median_ns_per_face"] = float(np.median(values))
                    methods[name]["timing_iqr_ns_per_face"] = np.quantile(values, [0.25, 0.75]).tolist()
                split["tolerances"][str(tol)] = {
                    "methods": methods, "paired_hard_vs_control": paired, **speed_comparison(times)}
                print(kind, tol, " / ".join(f"{name} {np.median(times[name]):.1f} ns" for name in METHODS),
                      flush=True)
    for tol in TOLERANCES:
        reports = [s["tolerances"][str(tol)] for s in report["splits"].values()]
        mean_times = {name: float(np.mean([r["methods"][name]["median_ns_per_face"]
                                           for r in reports])) for name in METHODS}
        best = min(CLASSICAL, key=mean_times.get)
        split_best_time = float(np.mean([min(r["methods"][name]["median_ns_per_face"] for name in CLASSICAL)
                                         for r in reports]))
        report["equal_split_mix"][str(tol)] = {
            "weight_each": 1 / len(reports), "mean_of_split_medians_ns": mean_times,
            "fastest_single_classical_method": best,
            "posthoc_best_classical_each_split_mean_ns": split_best_time,
            "speedup_vs_best_single_classical": {key: mean_times[best] / t for key, t in mean_times.items()},
            "corrected_rate_all": {name: float(np.mean([r["methods"][name]["corrected_rate_all"]
                                                        for r in reports])) for name in METHODS}}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def repeat_timing(report_path, rounds, cpu):
    """Fresh compilation and randomized timing pass; leave primary observations intact."""
    from .data import sample

    report_path = Path(report_path)
    report = json.loads(report_path.read_text())
    if rounds < 1:
        raise ValueError("Require a positive number of rounds")
    for name, old in report["sha256"].items():
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != old:
            raise ValueError(f"{name} changed since primary run")
    if hasattr(os, "sched_setaffinity"):
        chosen = report["cpu_affinity"][0] if cpu is None else cpu
        os.sched_setaffinity(0, {chosen})
    else:
        chosen = None
    second = {"rounds": rounds, "cpu": chosen, "schedule_seed": 912004,
              "description": "Independent repeat with fresh process/compiler invocation and reversed split/tolerance order.",
              "splits": {}}
    rng = np.random.default_rng(second["schedule_seed"])
    n = report["cases_per_split"]
    with TemporaryDirectory() as tmp:
        lib = build_solver(tmp)
        for kind, seed in reversed(tuple(SEEDS.items())):
            s = np.ascontiguousarray(sample(n, seed, split=kind), dtype=np.float64)
            if hashlib.sha256(s.tobytes()).hexdigest() != report["splits"][kind]["input_sha256"]:
                raise ValueError("Repeat faces differ from primary batch")
            second["splits"][kind] = {}
            for tol in reversed(TOLERANCES):
                times = time_batch(lib, s, tol, rounds, rng)
                second["splits"][kind][str(tol)] = {
                    "raw_ns_per_face": times,
                    "median_ns_per_face": {name: float(np.median(v)) for name, v in times.items()},
                    **speed_comparison(times)}
    report["repeat_timing"] = second
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPORT)
    parser.add_argument("--cases", type=int, default=8192)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--audits", type=int, default=32)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--repeat-timing", action="store_true",
                        help="Append independent CPU timings to an existing --out report")
    args = parser.parse_args()
    if args.repeat_timing:
        repeat_timing(args.out, args.rounds, args.cpu)
    else:
        run(args.out, args.cases, args.rounds, args.audits, args.cpu)
