"""Time the deployed flux network with free routing on the saved confirmation cases.

The preassigned masks are independent of the PDE states. These runs measure
only the cost that remains after omitting selection, certification, and budget
accounting. Enforce a single numerical thread before starting Python.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
from statistics import mean, median
from time import perf_counter_ns

import numpy as np

from certified_burgers.godunov import advance, godunov_flux
from .frozen_flux import FrozenFlux
from .experiments_a2 import discretize


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "reports/verifier_allocation/confirmation.json"
MODEL = ROOT / "reports/stage_a2/flux_w16_s0.json"
THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def algebraic_godunov(left, right):
    """Burgers G = max(left, -right, 0)^2/2, in the deployed FP multiplication order."""
    speed = np.maximum(np.maximum(left, -right), 0.0)
    return (0.5 * speed) * speed


def confirm_flux_equivalence(initials, lam, steps):
    """Check the algebraic baseline bitwise against the repository Godunov."""
    tiny = np.nextafter(0.0, 1.0)
    edge = np.array([-2., -1., -tiny, -0., 0., tiny, 1., 2.])
    left, right = np.meshgrid(edge, edge)
    rng = np.random.default_rng(4983)
    random = rng.uniform(-2., 2., (2, 8192))
    a = np.r_[left.ravel(), random[0], random[0], random[0]]
    b = np.r_[right.ravel(), random[1], -random[0], np.nextafter(-random[0], np.inf)]

    def check(u, v):
        classic = godunov_flux(u, v)
        np.testing.assert_array_equal(classic.view(np.uint64), algebraic_godunov(u, v).view(np.uint64))
        return classic

    check(a, b)
    comparisons = len(a)
    for initial in initials:
        state = initial.copy()
        for _ in range(steps):
            flux = check(state, np.roll(state, -1))
            state -= lam * (flux - np.roll(flux, 1))
            comparisons += len(state)
    return comparisons


def schedules(n, steps, accepted):
    cells = np.arange(n)
    result = []
    for j in range(steps):
        # A moving contiguous block favors locality. Gather/scatter remains timed.
        neural = (cells - 3*j) % n < accepted
        result.append((cells[neural], cells[~neural]))
    return result


def rollout(state, steps, lam, model, flux_fn, neural_count, schedule=None):
    """Common clipped, periodic FV update; omit the mask router and its certificate."""
    state = state.copy()
    for j in range(steps):
        right = np.roll(state, -1)
        count = len(schedule[j][0]) if neural_count is None else neural_count
        if count == 0:
            flux = flux_fn(state, right)
        elif count == len(state):
            flux = model.predict(np.column_stack((state, right)))
        else:
            use_f, use_g = schedule[j]
            flux = np.empty_like(state)
            flux[use_f] = model.predict(np.column_stack((state[use_f], right[use_f])))
            flux[use_g] = flux_fn(state[use_g], right[use_g])
        state = np.clip(state - lam * (flux - np.roll(flux, 1)), -2., 2.)
    return state


def sample_times(initial, candidates, warm, rng, repeats):
    samples = {name: [] for name in candidates}
    for _ in range(repeats):
        for name in rng.permutation(list(candidates)):
            start = perf_counter_ns()
            final = candidates[name](initial)
            seconds = (perf_counter_ns() - start) / 1e9
            samples[name].append(seconds)
            np.testing.assert_array_equal(final.view(np.uint64), warm[name].view(np.uint64))
    return samples


def run(repeats):
    threads = {name: os.environ.get(name) for name in THREAD_VARS}
    if any(value != "1" for value in threads.values()):
        raise ValueError("Set OPENBLAS_NUM_THREADS, OMP_NUM_THREADS, and MKL_NUM_THREADS to 1 before starting Python")
    reference = json.loads(CASES.read_text())
    cfg = reference["config"]
    n, steps, lam = cfg["N"], cfg["steps"], cfg["lam"]
    ics = {r["index"]: r["ic"] for r in reference["results"] if r["split"] == "confirmation"}
    assert list(sorted(ics)) == list(range(9))
    initials = [discretize(ics[i], n) for i in sorted(ics)]
    model = FrozenFlux.load(MODEL)
    assert model.digest == cfg["model_hash"]
    verified = confirm_flux_equivalence(initials, lam, steps)
    counts = (1, 13, 70, 115)
    masks = {count: schedules(n, steps, count) for count in counts}
    candidates = {
        "repo_advance": lambda u: advance(u, 1., steps, dt=lam)[0],
        "repo_flux_common_update": lambda u: rollout(u, steps, lam, model, godunov_flux, 0),
        "algebraic_flux_common_update": lambda u: rollout(u, steps, lam, model, algebraic_godunov, 0),
        "all_neural": lambda u: rollout(u, steps, lam, model, None, n),
    }
    for count in counts:
        candidates[f"preset_{count}_neural"] = (
            lambda u, k=count: rollout(u, steps, lam, model, godunov_flux, k, masks[k]))
    rng = np.random.default_rng(419)
    rows = []
    for i, initial in enumerate(initials):
        warm = {name: f(initial) for name, f in candidates.items()}
        for label in ("repo_flux_common_update", "algebraic_flux_common_update"):
            np.testing.assert_array_equal(warm[label].view(np.uint64), warm["repo_advance"].view(np.uint64))
        assert all(np.isfinite(final).all() for final in warm.values())
        samples = sample_times(initial, candidates, warm, rng, repeats)
        rows.append({"index": i, "kind": ics[i]["kind"], "wall_s": samples})
    summary = {name: mean(median(row["wall_s"][name]) for row in rows) for name in candidates}
    return {
        "scope": "Frozen CPU NumPy flux kernel and FV update; no certification, gating, or budget accounting. Precomputed contiguous masks use the repository Godunov on remaining faces. Timers exclude warmup, mask construction and output checks.",
        "n": n, "steps": steps, "lam": lam, "seed": 241124, "repeats": repeats,
        "preset_neural_counts_per_step": list(counts), "model_hash": model.digest,
        "arithmetic": model.arithmetic, "numpy": np.__version__, "python": platform.python_version(),
        "platform": platform.platform(), "threads_env": threads, "equivalent_flux_checks": verified,
        "wall_s_mean_of_case_medians": summary, "results": rows,
    }


def replay_actual_masks(repeats, gate_path):
    from .anchor_refinement import AnchorTable
    from .cheap_gate import DirectGate, gate_rollout
    from .stage_a2 import CertificateTables, CertifiedFlux

    threads = {name: os.environ.get(name) for name in THREAD_VARS}
    if any(value != "1" for value in threads.values()):
        raise ValueError("Set numerical thread counts to one before starting Python")
    ref = json.loads(CASES.read_text())
    cfg = ref["config"]
    n, steps, lam = cfg["N"], cfg["steps"], cfg["lam"]
    cases = [r for r in ref["results"] if r["method"] == "selective_anchor/all/128"]
    assert sorted(r["index"] for r in cases) == list(range(9))
    model = CertifiedFlux(FrozenFlux.load(MODEL))
    with np.load(gate_path, allow_pickle=False) as saved:
        if str(saved["model_hash"]) != model.digest or model.digest != cfg["model_hash"]:
            raise ValueError("Gate, frozen model and case hashes must agree")
        gate = DirectGate(saved["edges"], saved["signed_error"], saved["levels"],
                          int(saved["n_levels"]), int(saved["round"]))
    base = CertificateTables.build(model, bins=16)
    anchor = AnchorTable.build(model, base, bins=len(gate.edges)-1)
    rng = np.random.default_rng(10419)
    rows = []
    for case in cases:
        initial = discretize(case["ic"], n)
        certified = gate_rollout(model, base, anchor, gate, initial, steps=steps, lam=lam,
                                 total_budget=case["budget"], record_masks=True)
        schedule = [(np.flatnonzero(m), np.flatnonzero(~m)) for m in certified["masks"]]
        assert len(schedule) == steps and sum(len(f) for f, _ in schedule) == certified["accepted"]
        candidates = {
            "repo_advance": lambda u: advance(u, 1., steps, dt=lam)[0],
            "repo_flux_common_update": lambda u: rollout(u, steps, lam, model, godunov_flux, 0),
            "actual_masks_free": lambda u: rollout(u, steps, lam, model, godunov_flux, None, schedule)
        }
        warm = {name: fn(initial) for name, fn in candidates.items()}
        np.testing.assert_array_equal(warm["actual_masks_free"].view(np.uint64), certified["state"].view(np.uint64))
        np.testing.assert_array_equal(warm["repo_flux_common_update"].view(np.uint64), warm["repo_advance"].view(np.uint64))
        samples = sample_times(initial, candidates, warm, rng, repeats)
        rows.append({"index": case["index"], "kind": case["ic"]["kind"], "accepted": certified["accepted"],
                     "zero_neural_steps": sum(len(idx) == 0 for idx, _ in schedule), "wall_s": samples})
    summary = {name: mean(median(row["wall_s"][name]) for row in rows) for name in candidates}
    return {"scope": "Offline replay of actual certified round masks in the common FV update. Tracing masks, computing their indices, safety checks and budget accounting excluded from timed runs. Bitwise final-state parity tested for every case and repeat.",
            "round": gate.round, "model_hash": model.digest, "n": n, "steps": steps, "lam": lam,
            "repeats": repeats, "seed": 241124, "threads_env": threads, "numpy": np.__version__,
            "coverage": sum(row["accepted"] for row in rows)/(n*steps*len(rows)),
            "wall_s_mean_of_case_medians": summary, "results": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--actual-gate", type=Path, help="Replay the masks of this saved gate for free")
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    data = replay_actual_masks(args.repeats, args.actual_gate) if args.actual_gate else run(args.repeats)
    filename = "actual_mask_replay.json" if args.actual_gate else "flux_costs.json"
    out = args.out or Path("reports/cheap_routing") / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data["wall_s_mean_of_case_medians"], indent=2))
