"""Train a direct router offline and time certified one-step routing on saved Burgers cases."""
import argparse
import json
import os
import platform
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import numpy as np

from certified_burgers.godunov import advance
from .anchor_refinement import AnchorTable
from .budget import PacingPolicy
from .cheap_gate import gate_rollout, train_gate
from .experiments_a2 import discretize, LAM
from .experiments_refinement import make_cases, MODEL_PATH, N, STEPS
from .frozen_flux import FrozenFlux
from .stage_a2 import CertificateTables, CertifiedFlux, certified_rollout_a2


def run(out, repeats):
    out.mkdir(parents=True, exist_ok=True)
    model = CertifiedFlux(FrozenFlux.load(MODEL_PATH))
    base = CertificateTables.build(model, bins=16)
    anchor = AnchorTable.build(model, base, bins=128)
    start = perf_counter()
    training = make_cases(model, base, 1100, 12)
    calibration_s = perf_counter() - start
    start = perf_counter()
    gates = train_gate(model, base, anchor, [(v, b) for _, v, b in training], steps=STEPS, lam=LAM)
    training_s = perf_counter() - start
    for gate in gates:
        np.savez_compressed(out / f"gate_round{gate.round}.npz", edges=gate.edges, levels=gate.levels,
                            signed_error=gate.signed_error, n_levels=gate.n_levels, round=gate.round,
                            model_hash=model.digest)
    ref = json.loads((MODEL_PATH.parents[1] / "verifier_allocation/confirmation.json").read_text())
    if ref["config"]["model_hash"] != model.digest:
        raise ValueError("Model hash differs from the budget reference")
    cases = [r for r in ref["results"] if r["method"] == "selective_anchor/all/128"]
    if sorted(r["index"] for r in cases) != list(range(9)):
        raise ValueError("Expected all nine distinct confirmation cases")
    print(f"Budget calibration {calibration_s:.2f} s, two training rounds {training_s:.2f} s.", flush=True)
    order = np.random.default_rng(4321)
    rows = []
    methods = ["plain_godunov", "gate_round1", "gate_round2", "anchor_dp"]
    all_indices = np.arange(N)
    for case in cases:
        v = discretize(case["ic"], N)
        budget = case["budget"]
        common = dict(steps=STEPS, lam=LAM, total_budget=budget)

        def execute(method, audit=False):
            if method == "plain_godunov":
                state, _ = advance(v, 1., STEPS, dt=LAM)
                return {"state": state, "accepted": 0}
            if method == "anchor_dp":
                r = certified_rollout_a2(model, base, v, **common, policy=PacingPolicy(),
                                         mode="selective_anchor", anchor=anchor,
                                         refiner=lambda *_: all_indices, record=False)
                return {"state": r["state"], "accepted": sum(row["trusted"] for row in r["rows"]),
                        "spent": float(r["spent"])}
            gate = gates[0] if method == "gate_round1" else gates[1]
            return gate_rollout(model, base, anchor, gate, v, **common, audit=audit)

        # Give each path a full warm-up. DP is timed once per repeat on the
        # same inputs as the candidate, without the extra reference audit.
        warm = {name: execute(name) for name in methods}
        if warm["anchor_dp"]["accepted"] != case["accepted"]:
            raise ValueError("A2 reference acceptance changed on this input")
        times = {name: [] for name in methods}
        for _ in range(repeats):
            for name in order.permutation(methods):
                start = perf_counter()
                r = execute(name)
                times[name].append(perf_counter() - start)
                if not np.array_equal(r["state"], warm[name]["state"]):
                    raise ArithmeticError("Timing repeats disagree")
        audit = execute("gate_round2", audit=True)
        if audit["audited_steps"] != STEPS or audit["spent"] > budget:
            raise ArithmeticError("Audit or budget failed")
        if not np.array_equal(audit["state"], warm["gate_round2"]["state"]):
            raise ArithmeticError("Audit trajectory differs")
        for name in methods:
            stats = {k: val for k, val in warm[name].items() if k != "state"}
            rows.append({"case": case["index"], "method": name, "kind": case["ic"]["kind"],
                         "budget": budget, **stats, "times_s": times[name]})
    summary = {}
    for name in methods:
        group = [row for row in rows if row["method"] == name]
        summary[name] = {"seconds": mean(median(row["times_s"]) for row in group),
                         "coverage": mean(row["accepted"]/(N*STEPS) for row in group)}
    report = {"n": N, "steps": STEPS, "training_seed": 1100, "training_cases": len(training),
              "evaluation_seed": 241124, "offline_training_s": training_s,
              "offline_budget_calibration_s": calibration_s, "table_s": base.build_seconds,
              "anchor_s": anchor.build_seconds, "model_hash": model.digest,
              "audited_round2_steps": len(cases)*STEPS, "repeats": repeats, "numpy": np.__version__,
              "python": platform.python_version(), "platform": platform.platform(),
              "threads": {k: os.getenv(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
              "summary": summary, "results": rows}
    (out / "gate_timing.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("reports/cheap_routing"))
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    run(args.out, args.repeats)
