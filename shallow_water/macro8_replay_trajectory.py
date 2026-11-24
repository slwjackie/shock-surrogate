"""Confirm frozen on-policy replay models on new, repeated eight-step trajectories."""
import argparse
import ctypes
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter_ns

import numpy as np

from .benchmark import FLAGS
from .macro8_benchmark import ENDPOINT_BUDGET, FINE_ERROR_FACTOR, METHODS, apply, read_thresholds
from .macro8_data import DURATION, N, STEPS, ReferenceSolver, analytical_dam
from .macro8_replay_evaluation import build_replay_workload, confirmatory_split
from .macro8_trajectory import aggregate, observation


HERE = Path(__file__).resolve().parent
REPORTS = HERE.parent / "reports" / "shallow_water_macro8"
# Native replay library modes 0, 1, 2 reuse the original hard network, the
# on-policy uniform network and the on-policy hard-case network, respectively.
ROUTES = {"original_hard": "routed_initial", "on_policy_uniform": "routed_uniform",
          "on_policy_hard": "routed_hard"}
CONTROLS = ("teacher8", "engineering_cfl", "hlle_cfl", "rusanov_cfl")
POLICIES = {**{name: name for name in CONTROLS}, **ROUTES}


def composition(directory, lib):
    path = directory / "trajectory.so"
    subprocess.run(["gcc", *FLAGS, str(HERE / "macro8_trajectory_workload.c"),
                    str(Path(lib._name).resolve()), "-o", str(path)], check=True)
    compose = ctypes.CDLL(str(path)).sw_macro8_trajectory_batch
    batch = np.ctypeslib.ndpointer(dtype=np.float64, ndim=3, flags="C_CONTIGUOUS")
    counter = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS")
    compose.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, batch, batch,
                        ctypes.c_int, ctypes.c_double, counter]
    return compose


def measure(split, solver, lib, compose, thresholds, count, offset, blocks, rounds, rng):
    dataset = confirmatory_split(split, count, solver, start=offset)
    cases, families, ids = dataset.q, dataset.families, dataset.ids
    rows = {name: [[] for _ in range(blocks)] for name in POLICIES}
    ends = {name: np.full_like(cases, np.nan) for name in POLICIES}
    per_case = {name: [] for name in POLICIES}
    failures = []
    for i, q0 in enumerate(cases):
        states = {name: q0.copy() for name in POLICIES if name != "teacher8"}
        coarse = q0.copy()
        fine = np.repeat(q0, 4, axis=0)
        for k in range(blocks):
            coarse = solver.advance(coarse)
            fine = solver.advance(fine, steps=4*STEPS)
            ref = fine.reshape(N, 4, 2).mean(axis=1)
            dam = analytical_dam(q0, blocks=k+1) if families[i] == "dam" else None
            rows["teacher8"][k].append(observation(coarse, coarse, ref, dam))
            for name in tuple(states):
                incoming = states[name]
                try:
                    pred, route = apply(lib, POLICIES[name], incoming, thresholds)
                    if not np.isfinite(pred).all() or pred[:, 0].min() <= 0:
                        raise ArithmeticError("Nonfinite state or nonpositive water depth")
                except (RuntimeError, ArithmeticError) as exc:
                    failures.append(dict(case=ids[i], method=name, block=k+1, cause=str(exc)))
                    del states[name]
                    continue
                local_error = None
                if route == 1:
                    local_error = np.abs(pred-solver.advance(incoming)).mean(axis=0)
                accepted = bool(route) if route >= 0 else None
                rows[name][k].append(observation(pred, coarse, ref, dam, accepted, local_error))
                states[name] = pred
        ends["teacher8"][i] = coarse
        for name, state in states.items():
            ends[name][i] = state
        physical_limit = FINE_ERROR_FACTOR * np.abs(coarse-ref).mean(axis=0)
        for name in ("teacher8", *states):
            terminal = rows[name][-1][-1]
            per_case[name].append(dict(case=ids[i], family=families[i], **terminal,
                                       meets_teacher_fine_target=bool(np.all(
                                           terminal["vs_fine"] <= physical_limit))))

    stats = {name: [aggregate(block, count) for block in records] for name, records in rows.items()}
    complete = [name for name in POLICIES if stats[name][-1]["completed"] == count]
    timings = {name: [] for name in complete}
    out = np.empty_like(cases)
    accepted = np.zeros(1, dtype=np.int64)
    for rep in range(-2, rounds):
        for index in rng.permutation(len(complete)):
            name = complete[index]
            mode = POLICIES[name]
            threshold = thresholds[tuple(ROUTES).index(name)] if name in ROUTES else 0.
            method = METHODS.index(mode)
            start = perf_counter_ns()
            status = compose(count, N, blocks, cases, out, method, threshold, accepted)
            elapsed = perf_counter_ns()-start
            if status:
                raise RuntimeError(f"Native composition failed after per-block success for {name}")
            if rep == -2:
                np.testing.assert_array_equal(out, ends[name])
                if name in ROUTES:
                    assert accepted[0] == sum(b["accepted"] for b in stats[name])
            if rep >= 0:
                timings[name].append(elapsed/(1000*count))
    time_report = {name:dict(median_us=float(np.median(ts)), iqr_us=np.quantile(ts, [.25,.75]).tolist(),
                             samples_us_per_trajectory=ts) for name, ts in timings.items()}
    return dict(ids=ids, families=families, input_sha256=sha256(cases.tobytes()).hexdigest(),
                blockwise=stats, terminal_by_case=per_case, failures=failures, timing=time_report)


def evaluate(out, guard, cpu, count=32, offset=320, blocks=8, rounds=9):
    if min(count, blocks, rounds) <= 0 or offset < 0:
        raise ValueError("Use positive counts, blocks and rounds and a nonnegative index offset")
    if hasattr(os, "sched_getaffinity"):
        cores = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {cores[-1] if cpu is None else cpu})
    thresholds = read_thresholds(guard)
    calibration = json.loads(guard.read_text())
    rng = np.random.default_rng(179841)
    report = dict(count_per_split=count, trajectory_offset=offset, blocks=blocks, cells=N,
                  terminal_time_s=blocks*DURATION, fine_refinement=4, guard_thresholds=thresholds,
                  physical_target="Both terminal component MAEs against the fourfold fine-grid reference must be at most 1.05 times those of the eight-step coarse Godunov trajectory from the same initial condition.",
                  local_mae_targets=ENDPOINT_BUDGET.tolist(), units=["h_m","hu_m2_s"],
                  gate_audit="Accepts audited against eight tight Godunov microsteps from the policy's own current state. Terminal errors compare to the coarse/fine trajectories from the original initial state.",
                  native_timing="All blocks, routing and any corrections in a single native call, interleaved by method and divided by the number of initial profiles. Every timed terminal array matched the per-block audit bitwise.",
                  timing_rounds=rounds, timing_warmups=2, timing_seed=179841,
                  cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None,
                  compiler=subprocess.check_output(["gcc","--version"],text=True).splitlines()[0],
                  flags=FLAGS, calibration_split=calibration.get("split"),
                  frozen_sha256={p.name: sha256(p.read_bytes()).hexdigest() for p in
                                  (guard, REPORTS/"replay.npz", REPORTS/"weights.npz",
                                   HERE/"macro8_trajectory_workload.c")}, splits={})
    with ReferenceSolver() as solver, TemporaryDirectory() as tmp:
        lib = build_replay_workload(Path(tmp))
        run = composition(Path(tmp), lib)
        for split in ("test", "shift"):
            report["splits"][split] = measure(split, solver, lib, run, thresholds, count,
                                              offset, blocks, rounds, rng)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPORTS/"replay_trajectory.json")
    parser.add_argument("--guard", type=Path, default=REPORTS/"replay_guard.json")
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--offset", type=int, default=320)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--cpu", type=int)
    args = parser.parse_args()
    result = evaluate(args.out, args.guard, args.cpu, args.count, args.offset, args.blocks, args.rounds)
    for split, study in result["splits"].items():
        print(split, {n:round(m["median_us"],2) for n,m in study["timing"].items()})
