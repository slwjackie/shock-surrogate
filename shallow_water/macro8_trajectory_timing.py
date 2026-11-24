"""Interleaved single-core timing of complete held-out 64-step trajectories."""
import argparse
import ctypes
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter_ns

import numpy as np

from .benchmark import FLAGS
from .macro8_benchmark import METHODS, apply, build, read_thresholds
from .macro8_data import N, ReferenceSolver, make_split


HERE = Path(__file__).resolve().parent
REPORT = HERE.parent / "reports" / "shallow_water_macro8"
TIMED = ("teacher8", "godunov8", "godunov_cfl", "engineering_cfl", "hlle_cfl",
         "rusanov_cfl", "routed_hard")


def build_runner(directory):
    lib = build(directory)
    binary = directory / "trajectory.so"
    source = HERE / "macro8_trajectory_workload.c"
    subprocess.run(["gcc", *FLAGS, str(source), str(directory / "macro8.so"),
                    "-o", str(binary)], check=True)
    run = ctypes.CDLL(str(binary)).sw_macro8_trajectory_batch
    grid = np.ctypeslib.ndpointer(dtype=np.float64, ndim=3, flags="C_CONTIGUOUS")
    counter = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS")
    run.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, grid, grid,
                    ctypes.c_int, ctypes.c_double, counter]
    return lib, run


def evaluate(out, trajectory, guard, rounds, cpu):
    if rounds < 1:
        raise ValueError("At least one timing round is required")
    if hasattr(os, "sched_getaffinity"):
        available = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {available[-1] if cpu is None else cpu})
    report = json.loads(trajectory.read_text())
    thresholds = read_thresholds(guard)
    if report["source_sha256"]["guard_calibration"] != sha256(guard.read_bytes()).hexdigest():
        raise ValueError("Use the guard from the accuracy experiment")
    blocks = report["blocks"]
    rng = np.random.default_rng(67349)
    results = {
        "scope": "One native call runs every block for an immutable batch of independent profiles through the same method, including all checks, routing, retries, storage, and output writes. Time is divided by the number of profiles. Input generation and accuracy diagnostics are untimed.",
        "rounds": rounds, "warmups": 2, "schedule_seed": 67349, "blocks": blocks,
        "flags": FLAGS, "compiler": subprocess.check_output(["gcc","--version"],text=True).splitlines()[0],
        "platform": platform.platform(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None,
        "accuracy_sha256": sha256(trajectory.read_bytes()).hexdigest(),
        "native_runner_sha256": sha256((HERE / "macro8_trajectory_workload.c").read_bytes()).hexdigest(),
        "splits": {}}
    with TemporaryDirectory() as tmp, ReferenceSolver() as solver:
        lib, run = build_runner(Path(tmp))
        for split, accuracy in report["splits"].items():
            n_cases = len(accuracy["ids"])
            data = make_split(split, n_cases, solver)
            if sha256(data.q.tobytes()).hexdigest() != accuracy["state_sha256"]:
                raise RuntimeError("Initial states differ from the accuracy experiment")
            out_state = np.empty_like(data.q)
            accepted = np.zeros(1, dtype=np.int64)
            summary = {}
            for name in TIMED:
                check = np.ones_like(out_state)
                expected_accepted = 0
                for i, q in enumerate(data.q):
                    for _ in range(blocks):
                        q, route = apply(lib, name, q, thresholds)
                        expected_accepted += int(route == 1)
                    check[i] = q
                status = run(n_cases, N, blocks, data.q, out_state, METHODS.index(name),
                             thresholds[2], accepted)
                if status or accepted[0] != expected_accepted:
                    raise RuntimeError(f"Native trajectory/step loop route disagreement for {name}")
                np.testing.assert_array_equal(out_state, check)
                if name == "routed_hard":
                    recorded = sum(b["accepted"] for b in accuracy["blockwise"][name])
                    if recorded != expected_accepted:
                        raise RuntimeError("Routing differs from the accuracy study")
                summary[name] = {"accepts": expected_accepted,
                                 "output_sha256": sha256(out_state.tobytes()).hexdigest(),
                                 "samples_us_per_trajectory": []}
            for iteration in range(-2, rounds):
                for j in rng.permutation(len(TIMED)):
                    name = TIMED[j]
                    start = perf_counter_ns()
                    status = run(n_cases, N, blocks, data.q, out_state, METHODS.index(name),
                                 thresholds[2], accepted)
                    stop = perf_counter_ns()
                    if status or accepted[0] != summary[name]["accepts"]:
                        raise RuntimeError(f"Inconsistent result timing {name}")
                    if iteration >= 0:
                        summary[name]["samples_us_per_trajectory"].append((stop-start)/(n_cases*1000))
            for entry in summary.values():
                times = entry["samples_us_per_trajectory"]
                entry["median_us"] = float(np.median(times))
                entry["iqr_us"] = np.quantile(times, [.25, .75]).tolist()
            results["splits"][split] = summary
    out.write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")
    return results


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPORT / "trajectory_timing.json")
    parser.add_argument("--trajectory", type=Path, default=REPORT / "trajectory.json")
    parser.add_argument("--guard", type=Path, default=REPORT / "guard.json")
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--cpu", type=int)
    args = parser.parse_args()
    data = evaluate(args.out, args.trajectory, args.guard, args.rounds, args.cpu)
    for split, summary in data["splits"].items():
        print(split, {name:round(rows["median_us"],2) for name,rows in summary.items()})
