"""Compare self-learning and fixed rules for selective Stage A2 verification.

Run from the repository root with a single CPU thread. Each policy gets the E3
30% budget for its initial condition. Reference/evaluation and online times
are deliberately measured separately.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from statistics import median

import numpy as np

from certified_burgers.godunov import advance, godunov_step
from .budget import GreedyPolicy, PacingPolicy
from .frozen_flux import FrozenFlux
from .experiments_a2 import LAM, discretize, ic_family
from .stage_a2 import CertifiedFlux, CertificateTables, FPConstants, cell_costs, certified_rollout_a2

N, STEPS, COUNT = 128, 60, 10
SNAPSHOT_STEPS = range(0, STEPS, 8)
MODEL_PATH = Path(__file__).resolve().parents[1] / "reports/stage_a2/flux_w16_s0.json"
SPLITS = {"e3": 300, "fresh": 9702}


def top_table(state, table_costs, count, *, allowance=None):
    return np.argsort(-table_costs[:, 1, 1], kind="stable")[:count]


def make_cases(model, tables, seed, number):
    cases = []
    for ic in ic_family(seed, number):
        v = discretize(ic, N)
        ref = certified_rollout_a2(model, tables, v, steps=STEPS, lam=LAM,
                                   total_budget=10., policy=GreedyPolicy(), mode="local", record=False)
        if any(row["trusted"] != N for row in ref["rows"]):
            raise ArithmeticError("Full-trust reference did not trust every face")
        cases.append((ic, v, .30 * float(ref["spent"])))
    return cases


def first_round_labels(model, tables, cases):
    consts = FPConstants.build(LAM, tables.envelope)
    samples = []
    for _, initial, budget in cases:
        v = initial.copy()
        for step in range(STEPS):
            if step in SNAPSHOT_STEPS:
                coarse = cell_costs(model, tables, consts, v, mode="table")
                tight = cell_costs(model, tables, consts, v, mode="local")
                samples.append((v.copy(), coarse, np.minimum(coarse, tight), budget * N / (STEPS * LAM)))
            v, _ = godunov_step(v, 1.0, dt=LAM)
    return samples


def second_round_labels(model, tables, cases, selector):
    consts = FPConstants.build(LAM, tables.envelope)
    samples = []
    for _, v, budget in cases:
        step = 0
        allowance = budget * N / (STEPS * LAM)
        def observe(state, coarse):
            nonlocal step
            if step in SNAPSHOT_STEPS:
                tight = cell_costs(model, tables, consts, state, mode="local")
                samples.append((state.copy(), coarse.copy(), np.minimum(coarse, tight), allowance))
            step += 1
            return selector.select(state, coarse, COUNT, allowance=allowance)
        certified_rollout_a2(model, tables, v, steps=STEPS, lam=LAM, total_budget=budget,
                             policy=PacingPolicy(), mode="selective", refiner=observe, record=False)
    return samples


def execute(model, tables, v, budget, name, selectors, *, audit=False):
    allowance = budget * N / (STEPS * LAM)
    mode = name if name in ("local", "table") else "selective"
    def choose(state, costs):
        indices = selectors[name](state, costs, COUNT, allowance=allowance)
        if len(indices) != COUNT:
            raise AssertionError("Wrong selection count")
        return indices
    start = time.perf_counter()
    result = certified_rollout_a2(
        model, tables, v, steps=STEPS, lam=LAM, total_budget=budget, policy=PacingPolicy(),
        mode=mode, refiner=choose if mode == "selective" else None, audit=audit, record=False)
    elapsed = time.perf_counter() - start
    if result["spent"] > result["total_budget"]:
        raise AssertionError("Certified budget exceeded")
    if audit and not all(r["step_certificate_holds"] and r["global_certificate_holds"] for r in result["rows"]):
        raise AssertionError("Audit failed")
    return {"accepted": sum(r["trusted"] for r in result["rows"]), "wall_s": elapsed,
            "certificate_s": result["timing"]["certificate"], "spent": float(result["spent"])}


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def time_godunov(out, repeats=9):
    """Time the plain same-grid solver over the identical trajectories' initial states."""
    timings = {}
    for split, seed in SPLITS.items():
        states = [discretize(ic, N) for ic in ic_family(seed, 9)]
        advance(states[0], 1.0, STEPS, dt=LAM)
        records = []
        for state in states:
            samples = []
            for _ in range(repeats):
                start = time.perf_counter()
                advance(state, 1.0, STEPS, dt=LAM)
                samples.append(time.perf_counter() - start)
            records.append(samples)
        timings[split] = {"wall_s": float(np.mean([median(s) for s in records])),
                          "wall_s_by_ic": records}
    write(out / "plain_godunov.json", {"n": N, "steps": STEPS, "repeats": repeats,
                                       "results": timings})


def run(out, repeats):
    import torch

    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    model = CertifiedFlux(FrozenFlux.load(MODEL_PATH))
    tables = CertificateTables.build(model, bins=16)
    start = time.perf_counter()
    train = make_cases(model, tables, 1100, 12)
    initial = first_round_labels(model, tables, train)
    from .refinement_learning import RefinementSelector, select_table_block
    learned = RefinementSelector()
    learned.fit_round(initial, refit=False)
    learned.save(out / "selector_round1.json")
    frozen_round1 = RefinementSelector.load(out / "selector_round1.json")
    feedback = second_round_labels(model, tables, train, frozen_round1)
    learned.fit_round(feedback, refit=True)
    learned.save(out / "selector_round2.json")
    train_time = time.perf_counter() - start
    print(f"Training complete: {len(initial)} + {len(feedback)} snapshots, {train_time:.1f} s.", flush=True)

    selectors = {"top_table": top_table, "untrained_block": select_table_block,
                 "round1": frozen_round1.select, "round2": learned.select}
    methods = ["local", "table", *selectors]
    rows, audits = [], []
    rng = np.random.default_rng(1742)
    cases = {name: make_cases(model, tables, seed, 9) for name, seed in SPLITS.items()}
    config = {"n": N, "steps": STEPS, "count": COUNT, "budget_fraction": .3,
              "training_seed": 1100, "n_training_ics": len(train), "evaluation_seeds": SPLITS,
              "snapshots_per_round": [len(initial), len(feedback)], "training_s": train_time,
              "repeats": repeats, "model_hash": model.digest, "numpy": np.__version__,
              "python": platform.python_version(), "platform": platform.platform()}
    for name in methods:
        _, v, budget = train[0]
        execute(model, tables, v, budget, name, selectors)
    for split, batch in cases.items():
        for j, (ic, v, budget) in enumerate(batch):
            runs = {name: [] for name in methods}
            for rep in range(repeats):
                for name in rng.permutation(methods):
                    runs[name].append(execute(model, tables, v, budget, name, selectors))
            for name, samples in runs.items():
                assert len({(r["accepted"], r["spent"]) for r in samples}) == 1
                rows.append({"split": split, "index": j, "ic": ic, "budget": budget, "policy": name,
                             "accepted": samples[0]["accepted"], "spent": samples[0]["spent"],
                             "wall_s": [r["wall_s"] for r in samples],
                             "certificate_s": [r["certificate_s"] for r in samples]})
            print(split, j, {name: round(samples[0]["accepted"]/(N*STEPS)*100,1)
                             for name,samples in runs.items()}, flush=True)
            write(out / "evaluation.json", {"config": config, "results": rows})
        # Audit one smooth, one shocked and one Riemann run on each split, outside timers.
        for j, (_, v, budget) in enumerate(batch[:3]):
            audited = execute(model, tables, v, budget, "round2", selectors, audit=True)
            audits.append({"split": split, "index": j, "steps": STEPS, "spent": audited["spent"],
                           "budget": budget, "one_step_and_global_passed": True})
    summary = {}
    for split in SPLITS:
        summary[split] = {}
        for name in methods:
            records = [r for r in rows if r["split"] == split and r["policy"] == name]
            summary[split][name] = {
                "neural_fraction": sum(r["accepted"] for r in records)/(N*STEPS*len(records)),
                "wall_s": float(np.mean([median(r["wall_s"]) for r in records])),
                "certificate_s": float(np.mean([median(r["certificate_s"]) for r in records]))}
    write(out / "evaluation.json", {"config": config, "results": rows, "audits": audits, "summary": summary})
    time_godunov(out)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/refinement"))
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    run(args.out, args.repeats)
