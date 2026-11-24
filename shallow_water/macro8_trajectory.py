"""Test repeated eight-step surrogate calls on each method's own predicted state.

The local acceptance audit uses an eight-step Godunov rollout of the state
presented to the gate. To measure accumulated error, both Godunov trajectories
start at the original initial condition, once on the coarse mesh and once on
a refined mesh. All accuracy diagnostics run outside the timing benchmark.
"""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .macro8_benchmark import ENDPOINT_BUDGET, apply, build, read_thresholds
from .macro8_data import DURATION, G, N, STEPS, ReferenceSolver, analytical_dam, make_split


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "shallow_water_macro8"


def l1(q, ref):
    return np.mean(np.abs(q-ref), axis=0)


def observation(q, teacher, fine, dam_exact=None, accepted=None, local_error=None):
    record = dict(vs_teacher=l1(q, teacher).tolist(), vs_fine=l1(q, fine).tolist(),
                  min_h=float(q[:, 0].min()))
    if accepted is not None:
        record["accepted"] = accepted
    if local_error is not None:
        record["local_violation"] = bool(np.any(local_error > ENDPOINT_BUDGET))
    if dam_exact is not None:
        record["vs_dam_exact"] = l1(q, dam_exact).tolist()
    return record


def aggregate(records, requested):
    result = dict(requested=requested, completed=len(records))
    if not records:
        return result
    for metric in ("vs_teacher", "vs_fine", "vs_dam_exact"):
        errors = np.array([row[metric] for row in records if metric in row])
        if not len(errors):
            continue
        result[metric] = dict(count=len(errors), mean=errors.mean(axis=0).tolist(),
                              p95=np.quantile(errors, .95, axis=0).tolist(),
                              max=errors.max(axis=0).tolist())
    result["minimum_depth"] = min(row["min_h"] for row in records)
    if "accepted" in records[0]:
        accepted = [row for row in records if row["accepted"]]
        result["accepted"] = len(accepted)
        result["fallback"] = len(records)-len(accepted)
        result["accepted_audited"] = sum("local_violation" in row for row in accepted)
        result["accepted_over_local_limits"] = sum(row.get("local_violation", False)
                                                  for row in accepted)
    return result


def assess(split, count, blocks, refinement, lib, solver, thresholds):
    batch = make_split(split, count, solver)
    methods = ("teacher8", "godunov8", "godunov_cfl", "engineering_cfl",
               "hlle_cfl", "rusanov_cfl", "hard", "routed_hard")
    metrics = {name: [[] for _ in range(blocks)] for name in methods}
    terminal = defaultdict(list)
    failures = []
    for index, q0 in enumerate(batch.q):
        case_id = str(batch.ids[index])
        family = str(batch.families[index])
        teacher = q0.copy()
        fine = np.repeat(q0, refinement, axis=0)
        states = {name: q0.copy() for name in methods if name != "teacher8"}
        for k in range(blocks):
            teacher = solver.advance(teacher)
            fine = solver.advance(fine, steps=STEPS*refinement)
            fine_means = fine.reshape(N, refinement, 2).mean(axis=1)
            dam = analytical_dam(q0, blocks=k+1) if family == "dam" else None
            control = observation(teacher, teacher, fine_means, dam)
            metrics["teacher8"][k].append(control)
            if k == blocks-1:
                terminal["teacher8"].append(dict(case=case_id, family=family, **control))
            for name in tuple(states):
                incoming = states[name]
                try:
                    updated, route = apply(lib, name, incoming, thresholds)
                    if not np.isfinite(updated).all():
                        raise ArithmeticError("Nonfinite state")
                    if np.min(updated[:, 0]) <= 0:
                        raise ArithmeticError(f"Nonpositive depth (minimum {updated[:, 0].min():.9g} m)")
                except (ArithmeticError, RuntimeError) as exc:
                    failures.append(dict(case=case_id, family=family, method=name,
                                         first_failed_block=k+1, reason=str(exc)))
                    del states[name]
                    continue
                accepted = bool(route) if route in (0, 1) else None
                local = None
                if accepted:
                    local = l1(updated, solver.advance(incoming))
                row = observation(updated, teacher, fine_means, dam, accepted, local)
                metrics[name][k].append(row)
                if k == blocks-1:
                    terminal[name].append(dict(case=case_id, family=family, **row))
                states[name] = updated
    return dict(ids=list(map(str, batch.ids)), families=list(map(str, batch.families)),
                state_sha256=sha256(batch.q.tobytes()).hexdigest(),
                blockwise={name: [aggregate(rows, count) for rows in per_block]
                           for name, per_block in metrics.items()},
                terminal=terminal, failures=failures)


def evaluate(out, threshold_path, count, blocks, refinement, cpu):
    if count < 1 or blocks < 1 or refinement < 2:
        raise ValueError("Expected positive case and block counts and refinement >= 2")
    if hasattr(os, "sched_getaffinity"):
        available = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {available[-1] if cpu is None else cpu})
    thresholds = read_thresholds(threshold_path)
    dependencies = ["macro8_data.py", "macro8_predict.c", "macro8_guard.c",
                    "macro8_benchmark.py", "macro8_classical.c", "macro8_workload.c",
                    "rollout.c", "solver.c", "certificate.h",
                    "model.h", "macro8_weights.h"]
    hashes = {path: sha256((ROOT / "shallow_water" / path).read_bytes()).hexdigest()
              for path in dependencies}
    hashes["guard_calibration"] = sha256(threshold_path.read_bytes()).hexdigest()
    results = dict(cells=N, g=G, block_steps=STEPS, block_duration_s=DURATION,
                   blocks=blocks, terminal_time_s=DURATION*blocks,
                   fine_grid_refinement=refinement, local_mae_limits=ENDPOINT_BUDGET.tolist(),
                   components=["h_m", "hu_m2_s"], hard_model_threshold=thresholds[2],
                   reference="Tight eight-step Godunov on N cells with the same starting IC. A separate refined Godunov reference uses piecewise-constant prolongation and cell-average restriction at each observation. Single dam breaks also have an analytical entropy-solution reference sampled as cell averages.",
                   acceptance="Valid local acceptance means both component errors vs the tight eight-step teacher on that route's own incoming state meet their thresholds. Global trajectory errors use the initial-state references.",
                   timing="Only accuracy and route counts are collected during this study; separately interleaved native timing is reported by macro8_benchmark.",
                   source_sha256=hashes, splits={})
    with TemporaryDirectory() as tmp, ReferenceSolver() as solver:
        lib = build(Path(tmp))
        for split in ("test", "shift"):
            results["splits"][split] = assess(split, count, blocks, refinement, lib,
                                              solver, thresholds)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPORTS / "trajectory.json")
    parser.add_argument("--thresholds", type=Path, default=REPORTS / "guard.json")
    parser.add_argument("--count", type=int, default=48, help="Number of trajectories per split")
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--refinement", type=int, default=4)
    parser.add_argument("--cpu", type=int)
    args = parser.parse_args()
    study = evaluate(args.out, args.thresholds, args.count, args.blocks, args.refinement, args.cpu)
    for distribution, split in study["splits"].items():
        print(distribution, "routed final block", split["blockwise"]["routed_hard"][-1])
