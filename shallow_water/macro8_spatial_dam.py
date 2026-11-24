"""Compare frozen neural, fixed-step Godunov and spatially coupled dam rollouts."""
import argparse
import ctypes
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import subprocess
from tempfile import NamedTemporaryFile, TemporaryDirectory
from time import perf_counter_ns

import numpy as np

from .benchmark import FLAGS
from .macro8_benchmark import shock_error
from .macro8_data import FAMILIES, N, SPLIT_SEEDS, _dam, analytical_dam
from .macro8_model import OUT
from .macro8_replay_native import replay_source

HERE = Path(__file__).resolve().parent
METHODS = ("godunov", "neural", "spatial")
BLOCKS = 8


def build_kernel(folder):
    predictor = replay_source(folder)
    binary = Path(folder) / "spatial_dam.so"
    subprocess.run(["gcc", *FLAGS, str(predictor), str(HERE/"macro8_spatial_dam.c"),
                    str(HERE/"rollout.c"), str(HERE/"solver.c"), "-lm", "-o", str(binary)],
                   check=True)
    lib = ctypes.CDLL(str(binary))
    grid = np.ctypeslib.ndpointer(np.float64, ndim=2, shape=(N, 2), flags="C_CONTIGUOUS")
    integer = ctypes.c_int
    lib.sw_run_dam.argtypes = [grid, grid, integer, integer, integer, integer,
                               ctypes.POINTER(integer)]
    lib.sw_run_dam.restype = integer
    return lib.sw_run_dam


def case_input(index):
    if FAMILIES[index % len(FAMILIES)] != "dam":
        raise ValueError("Expected a dam ID from the interleaved test split")
    rng = np.random.default_rng(np.random.SeedSequence([SPLIT_SEEDS["test"], index]))
    return _dam(rng, shift=False)


def measure(kernel, index, repeats):
    q = case_input(index)
    gate = int(np.flatnonzero(np.diff(q[:, 0]))[0]+1)
    deep_left = bool(q[0, 0] > q[-1, 0])
    exact = analytical_dam(q, BLOCKS)
    peak = np.abs(exact[:, 1]).max()
    out = np.empty_like(q)
    failed = ctypes.c_int()
    measurements = {name: [] for name in METHODS}
    metrics = {}
    rng = np.random.default_rng([9137, index])
    for repeat in range(repeats+1):
        for method in rng.permutation(len(METHODS)):
            name = METHODS[method]
            tick = perf_counter_ns()
            status = kernel(q, out, int(method), gate, deep_left, BLOCKS, ctypes.byref(failed))
            elapsed = (perf_counter_ns()-tick)/1000
            if repeat:
                if (status, failed.value) != (metrics[name]["status"], metrics[name]["failed_block"]):
                    raise RuntimeError(f"Nonreproducible solver status for {index}/{name}")
                if status == 0:
                    measurements[name].append(elapsed)
                continue
            row = {"status": status, "failed_block": failed.value}
            if status == 0:
                h_mae, q_mae = np.abs(out-exact).mean(axis=0)
                row.update(h_mae_m=h_mae, q_mae_m2_s=q_mae,
                           shock_error_cells=shock_error(q, out, BLOCKS),
                           peak_abs_q_error_m2_s=abs(np.abs(out[:, 1]).max()-peak))
            metrics[name] = row
    for name, samples in measurements.items():
        if samples:
            metrics[name]["median_us"] = float(np.median(samples))
    return dict(id=f"test-{index:06d}", h_left_m=q[0, 0], h_right_m=q[-1, 0],
                velocity_m_s=q[0, 1]/q[0, 0], gate_face=gate, deep_left=deep_left,
                methods=metrics)


def summarize(cases):
    out = {}
    rows = [case["methods"] for case in cases]
    for method in METHODS:
        good = [r[method] for r in rows if r[method]["status"] == 0]
        paired = [r for r in rows if r["godunov"]["status"] == r[method]["status"] == 0]
        result = dict(completed=len(good), total=len(rows), paired=len(paired))
        keys = ("h_mae_m", "q_mae_m2_s", "shock_error_cells", "peak_abs_q_error_m2_s", "median_us")
        means = {}
        for key in keys:
            values = [r[key] for r in good if r[key] is not None]
            means[key] = float(np.mean(values)) if values else None
        result.update(mean_on_completed=means,
                      shocks_located=sum(r["shock_error_cells"] is not None for r in good))
        if paired:
            keys = ("h_mae_m", "q_mae_m2_s")
            baseline = np.array([[r["godunov"][k] for k in keys] for r in paired])
            candidate = np.array([[r[method][k] for k in keys] for r in paired])
            wins = candidate < baseline
            result.update(wins_h_q_both=[int(wins[:, 0].sum()), int(wins[:, 1].sum()),
                                         int(wins.all(axis=1).sum())],
                          mae_reduction_percent=(100*(1-candidate.mean(0)/baseline.mean(0))).tolist())
        out[method] = result
    return out


def print_panel(label, summary):
    print(f"\n{label} (64 reference steps, errors vs analytical cell averages)")
    print("method      completed     h MAE(m)  q MAE(m²/s)  shock(cells)  peak |q| error   time(µs)")
    for name, stats in summary.items():
        scores = stats["mean_on_completed"]
        values = ["--" if v is None else f"{v:.5f}" for v in scores.values()]
        print(f"{name:10s} {stats['completed']:4d}/{stats['total']:<4d} " +
              " ".join(f"{v:>12s}" for v in values))
    stats = summary["spatial"]
    if stats["paired"]:
        wins = "/".join(map(str, stats["wins_h_q_both"]))
        reductions = ", ".join(f"{v:+.2f}%" for v in stats["mae_reduction_percent"])
        print(f"Spatial vs Godunov on {stats['paired']} paired cases: "
              f"lower h / q / both counts = {wins}; mean MAE reduction (h, q) = {reductions}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=128, help="number of additional unseen dams")
    parser.add_argument("--repeats", type=int, default=7, help="timed repeats per case after warmup")
    parser.add_argument("--out", type=Path, default=OUT/"spatial_dam.json")
    args = parser.parse_args()
    if min(args.cases, args.repeats) < 1:
        parser.error("--cases and --repeats must be positive")
    cpu = None
    if hasattr(os, "sched_getaffinity"):
        cpu = min(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {cpu})
    report = dict(blocks=BLOCKS, cells=N, seed=SPLIT_SEEDS["test"], distribution="test/dam",
                  weights="on-policy uniform", weights_sha256=sha256((OUT/"replay.npz").read_bytes()).hexdigest(),
                  baseline="fixed-step reference Godunov, mode 6, depth tolerance 1e-8 m",
                  reference="analytical Riemann cell averages at t=0.0175 s",
                  timing="one native call per full rollout, including interface work and validity checks; "
                         "one warmup then shuffled repetitions; cohort mean of per-case medians on completed runs",
                  repeats=args.repeats, platform=platform.platform(), machine=platform.machine(),
                  processor=platform.processor(), pinned_cpu=cpu, numpy_version=np.__version__,
                  compiler=subprocess.check_output(["gcc", "--version"], text=True).splitlines()[0],
                  flags=FLAGS, cohorts={})
    with TemporaryDirectory() as folder:
        run = build_kernel(folder)
        for label, start, count in [("previous_16", 321, 16), ("new", 385, args.cases)]:
            cases = [measure(run, start+len(FAMILIES)*i, args.repeats) for i in range(count)]
            summary = summarize(cases)
            report["cohorts"][label] = dict(summary=summary, cases=cases)
            print_panel(label, summary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, indent=2, allow_nan=False)+"\n"
    with NamedTemporaryFile(mode="w", dir=args.out.parent, encoding="utf-8", delete=False) as tmp:
        tmp.write(serialized)
    Path(tmp.name).replace(args.out)
    print(f"\nPer-case inputs, failures, errors and times: {args.out}")


if __name__ == "__main__":
    main()
