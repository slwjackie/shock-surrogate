"""Offline diagnostic for the value of selectively tightening cell bounds."""

import operator
from fractions import Fraction

import numpy as np

from .stage_a2 import FPConstants, budget_parameters, cell_costs, per_step_reserve, trust_frontier
from .vinterval import U, up, up_float


def oracle_refinement_frontier(table_costs, local_costs, count):
    """Best raw cost for each neural-face count, using at most count local rows.

    Both inputs are full-grid (n,2,2) arrays for the same state. The oracle
    knows the local cost of every cell in advance. A local check tightens only
    the checked cell's four costs, with the table cost still available there.
    The returned vector uses the same floating point summation order as
    trust_frontier. It is a diagnostic for a single state, not a certificate.
    """
    table = np.asarray(table_costs, dtype=np.float64)
    local = np.asarray(local_costs, dtype=np.float64)
    if table.ndim != 3 or table.shape[1:] != (2, 2) or local.shape != table.shape:
        raise ValueError("Expected matching (n,2,2) cost arrays")
    if not np.all(np.isfinite(table)) or not np.all(np.isfinite(local)):
        raise ValueError("Expected finite certificate costs")
    n = len(table)
    count = operator.index(count)
    if n < 2 or not 0 <= count <= n:
        raise ValueError("Expected n >= 2 and 0 <= count <= n")
    if count == 0:
        return trust_frontier(table).cost
    tightened = np.minimum(table, local)
    if count == n:
        return trust_frontier(tightened).cost

    best = np.full(n + 1, np.inf)
    # Fix the periodic boundary face. The remaining two indices of the DP
    # count local checks performed and neural faces used, respectively.
    for boundary in (0, 1):
        previous = np.full((2, count + 1, n + 1), np.inf)
        previous[boundary, 0, 0] = 0.0
        for i in range(n):
            following = np.full_like(previous, np.inf)
            for right in (0, 1):
                # Face i adds right to the neural face count. Checking cell i
                # adds one local check and changes its cost for this face pair.
                dest = following[right, :, right : right + i + 1]
                prior = previous[:, :, :i + 1]
                np.minimum(prior[0] + table[i, 0, right],
                           prior[1] + table[i, 1, right], out=dest)
                checked = np.minimum(prior[0, :-1] + tightened[i, 0, right],
                                     prior[1, :-1] + tightened[i, 1, right])
                np.minimum(dest[1:], checked, out=dest[1:])
            previous = following
        np.minimum(best, previous[boundary].min(axis=0), out=best)
    return best


def diagnose_first_step(model, tables, state, total_budget, *, steps, lam, count, anchor=None):
    """Compare first-step face counts at the same pacing allowance as E3."""
    n = len(state)
    constants = FPConstants.build(lam, tables.envelope)
    budget = budget_parameters(tables, lam, n, steps, total_budget)
    c0, _ = per_step_reserve(tables, constants, n)
    allowance = budget["excess"] / steps
    values = np.arange(n + 1, dtype=np.float64)
    scale = up_float(Fraction(float(lam)) / n / (1 - (n-1)*U))
    table = cell_costs(model, tables, constants, state, mode="table")
    if anchor is None:
        local = cell_costs(model, tables, constants, state, mode="local")
        dense_label = "full_local"
    else:
        local = cell_costs(model, tables, constants, state, mode="selective_anchor",
                           refiner=lambda *_: np.arange(n), anchor=anchor)
        dense_label = "full_anchor"
    chosen = np.argsort(-table[:, 1, 1], kind="stable")[:count]
    fixed = table.copy()
    fixed[chosen] = np.minimum(table[chosen], local[chosen])
    choices = {
        "table": trust_frontier(table).cost,
        "top_table": trust_frontier(fixed).cost,
        "oracle": oracle_refinement_frontier(table, local, count),
        dense_label: trust_frontier(local).cost,
    }
    if anchor is not None:
        from .refinement_learning import select_table_block

        # The online block ranker uses the raw average budget as its hint.
        # Actual acceptance always uses the rounded pacing menu below.
        selected = select_table_block(state, table, count, allowance=total_budget*n/(steps*lam))
        block = table.copy()
        block[selected] = local[selected]
        choices["block"] = trust_frontier(block).cost
    counts = {}
    for name, raw in choices.items():
        # Match A2's floating-point menu construction before applying pacing.
        cbar = up(raw * scale)
        charges = np.where(cbar > c0, up(cbar - c0), 0.0)
        weights = up(charges + up(values * budget["c_min"])) if budget["c_min"] > 0 else charges
        weights[0] = 0.0
        counts[name] = int(np.flatnonzero(weights <= allowance)[-1])
    return {"total_budget": float(total_budget), "step_excess_allowance": float(allowance),
            "trusted_faces": counts}


def main():
    """Reproduce the nine E3 first-step diagnostics at the reported budgets."""
    import argparse
    import json
    from pathlib import Path

    from .burgers import FrozenFlux
    from .experiments_a2 import discretize, ic_family
    from .stage_a2 import CertificateTables, CertifiedFlux

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--anchor-fresh", action="store_true",
                        help="Compare table, top, block, joint hindsight optimum, and dense anchor on fresh states")
    parser.add_argument("--anchor-bins", type=int, default=128)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    if not 0 <= args.count <= 128:
        parser.error("--count must be between 0 and 128")
    artifacts = Path(__file__).resolve().parents[1] / "reports" / "stage_a2"
    model = CertifiedFlux(FrozenFlux.load(artifacts / "flux_w16_s0.json"))
    tables = CertificateTables.build(model, bins=16)
    anchor = None
    if args.anchor_fresh:
        from .anchor_refinement import AnchorTable

        anchor = AnchorTable.build(model, tables, bins=args.anchor_bins)
        reference = json.loads((artifacts.parent / "selective_refinement/evaluation.json").read_text())
        if reference["config"]["model_hash"] != model.digest:
            raise ValueError("Reference budgets belong to a different model")
        seed = reference["config"]["evaluation_seeds"]["fresh"]
        cases = [(r["index"], r["ic"], r["budget"]) for r in reference["results"]
                 if r["split"] == "fresh" and r["policy"] == "table"]
        if sorted(index for index, _, _ in cases) != list(range(9)):
            raise ValueError("Expected nine distinct fresh reference cases")
        default_out = Path("reports/verifier_allocation/oracle_anchor_first_steps.json")
    else:
        seed = 300
        records = json.loads((artifacts / "e3_budget.json").read_text())
        budgets = {r["ic"]:r["total_budget"] for r in records
                   if r["fraction"] == .3 and r["policy"] == "pacing"}
        cases = [(j, ic, budgets[j]) for j, ic in enumerate(ic_family(seed, 9))]
        default_out = Path("reports/selective_refinement/oracle_first_steps.json")
    results = []
    for j, ic, budget in cases:
        diagnostic = diagnose_first_step(model, tables, discretize(ic, 128), budget,
                                         steps=60, lam=.2, count=args.count, anchor=anchor)
        results.append({"ic": j, "kind": ic["kind"], **diagnostic})
    payload = {"seed": seed, "n": 128, "refined_cells": args.count, "step_index": 0,
               "description": "Snapshot optimum with hindsight access to every local bound.",
               "results": results}
    if anchor is not None:
        payload.update(anchor_bins=anchor.bins, model_hash=model.digest)
        counts = [r["trusted_faces"] for r in results]
        if not all(c["table"] <= min(c["top_table"], c["block"])
                   and max(c["top_table"], c["block"]) <= c["oracle"] <= c["full_anchor"]
                   for c in counts):
            raise AssertionError("Hindsight feasibility ordering failed")
        payload["neural_fraction"] = {m: sum(c[m] for c in counts) / (128*len(counts))
                                      for m in counts[0]}
    out = args.out or default_out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
