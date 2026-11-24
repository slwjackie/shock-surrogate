"""Measure certified coverage, verification time and rollout time at fixed per-case error budgets."""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median

import numpy as np
import torch

from .budget import PacingPolicy
from .burgers import FrozenFlux
from .experiments_a2 import LAM, discretize
from .experiments_refinement import MODEL_PATH, N, STEPS, make_cases, top_table
from .refinement_learning import select_table_block
from .stage_a2 import CertifiedFlux, CertificateTables, certified_rollout_a2

REFERENCE = MODEL_PATH.parents[1] / "selective_refinement" / "evaluation.json"
DEFAULT_MODES = ("table", "local", "selective", "selective_point", "selective_replay", "selective_anchor")
MODES = (*DEFAULT_MODES, "anchor_then_point")
ANCHOR_MODES = ("selective_anchor", "anchor_then_point")
CONFIRMATION_SEED = 241124
RANKERS = {"top": top_table, "block": select_table_block}


@dataclass(frozen=True)
class Method:
    mode: str
    rule: str
    count: int

    @property
    def label(self):
        return f"{self.mode}/{self.rule}/{self.count}"


def parse_method(label):
    try:
        mode, rule, count_text = label.split("/")
        count = int(count_text)
    except ValueError:
        raise argparse.ArgumentTypeError("Use MODE/RULE/COUNT") from None
    if mode not in MODES:
        raise argparse.ArgumentTypeError(f"Unknown mode {mode}")
    if mode == "table":
        valid = rule == "none" and count == 0
    elif mode == "local":
        valid = rule == "all" and count == N
    else:
        valid = (rule in RANKERS and 0 < count < N) or (rule == "all" and count == N)
    if mode == "anchor_then_point" and rule == "uncertainty":
        valid = 0 < count < N
    if not valid:
        raise argparse.ArgumentTypeError(f"Unsupported rule/count for {mode}")
    return Method(mode, rule, count)


def methods(args):
    if args.only_methods:
        return list(dict.fromkeys(args.only_methods))
    result = []
    for mode in dict.fromkeys(args.modes):
        if mode in ("local", "table"):
            result.append(Method(mode, "all" if mode == "local" else "none", N if mode == "local" else 0))
            continue
        counts = args.counts if mode in args.sweep_modes else [10]
        for count in sorted(set(counts)):
            for rule in args.rules:
                result.append(Method(mode, rule, count))
        if args.full_point and mode in ("selective_point", "selective_replay", "selective_anchor"):
            result.append(Method(mode, "all", N))
    return result


def run_case(model, tables, state, budget, method, *, anchor=None, audit=False, quick=False, trace_overlap=False):
    decisions = []
    allowance = budget * N / (STEPS * LAM)
    all_cells = np.arange(N)
    if method.rule == "uncertainty":
        from .refinement_learning import select_anchor_uncertainty

    def choose(v, table):
        if method.rule == "all":
            selected = all_cells
        elif method.rule == "uncertainty":
            selected = select_anchor_uncertainty(v, table, method.count, allowance=allowance, anchor=anchor)
        else:
            selected = RANKERS[method.rule](v, table, method.count, allowance=allowance)
        decisions.append(selected)
        return selected

    # A short warmup exercises the same code path, outside reported timings.
    horizon = 3 if quick else STEPS
    timed_budget = budget*horizon/STEPS if quick else budget
    selective = method.mode not in ("local", "table")
    start = time.perf_counter()
    rollout = certified_rollout_a2(model, tables, state, steps=horizon, lam=LAM,
                                   total_budget=timed_budget, policy=PacingPolicy(),
                                   mode=method.mode, refiner=choose if selective else None,
                                   record=False, audit=audit, record_masks=trace_overlap,
                                   **({"anchor": anchor} if method.mode in ANCHOR_MODES else {}))
    wall = time.perf_counter() - start
    if rollout["spent"] > rollout["total_budget"]:
        raise AssertionError("Certified cumulative budget exceeded")
    rows = rollout["rows"]
    if len(rows) != horizon:
        raise AssertionError("Incomplete rollout")
    if audit and not all(r["step_certificate_holds"] and r["global_certificate_holds"] for r in rows):
        raise AssertionError("One-step or cumulative audit failed")
    if selective:
        if len(decisions) != horizon or any(len(x) != method.count for x in decisions):
            raise AssertionError("Inspection count changed")
        unique_faces = sum(len(np.union1d(x, (x - 1) % N)) for x in decisions)
    else:
        unique_faces = method.count * horizon
    # Count verifier-stage face queries separately, excluding later routed solver evaluations.
    anchor_faces = N*horizon if method.mode == "anchor_then_point" else 0
    if method.mode == "selective_anchor":
        anchor_faces = unique_faces
    point_faces = 0 if method.mode in ("selective_anchor", "table") else unique_faces
    result = {"anchor_faces": anchor_faces, "point_faces": point_faces, "accepted": sum(r["trusted"] for r in rows), "spent": float(rollout["spent"]),
            "unique_faces": unique_faces, "wall_s": wall,
            "certificate_s": rollout["timing"]["certificate"]}
    if trace_overlap:
        overlap = 0
        for step, row in enumerate(rows):
            trusted = row["trust_mask"]
            if np.count_nonzero(trusted) != row["trusted"]:
                raise AssertionError("Recorded mask does not match routing count")
            if method.mode == "local":
                overlap += row["trusted"]
            elif point_faces:
                checked = decisions[step]
                queried = np.union1d(checked, (checked - 1) % N)
                overlap += int(np.count_nonzero(trusted[queried]))
        result["trusted_with_point_query"] = overlap
        result["trusted_without_point_query"] = result["accepted"] - overlap
    return result


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["split"],row["method"]].append(row)
    splits = defaultdict(dict)
    for (split, name), samples in groups.items():
        splits[split][name] = {
            "n_cases": len(samples),
            "neural_fraction": mean(r["accepted"]/(STEPS*N) for r in samples),
            "unique_faces_per_step": mean(r["unique_faces"]/STEPS for r in samples),
            "anchor_faces_per_step": mean(r["anchor_faces"]/STEPS for r in samples),
            "point_faces_per_step": mean(r["point_faces"]/STEPS for r in samples),
            "wall_s": mean(median(r["wall_s"]) for r in samples),
            "certificate_s": mean(median(r["certificate_s"]) for r in samples)}
    for stats in splits.values():
        for entry in stats.values():
            for cost in ("wall_s", "certificate_s"):
                entry["pareto_" + cost] = not any(
                    other[cost] <= entry[cost] and other["neural_fraction"] >= entry["neural_fraction"]
                    and (other[cost] < entry[cost] or other["neural_fraction"] > entry["neural_fraction"])
                    for other in stats.values())
    return dict(splits)


def run(args):
    torch.set_num_threads(1)
    previous = json.loads(REFERENCE.read_text())
    model = CertifiedFlux(FrozenFlux.load(MODEL_PATH))
    if previous["config"]["model_hash"] != model.digest:
        raise ValueError("Reference budgets were generated with different weights")
    start = time.perf_counter()
    tables = CertificateTables.build(model, bins=16)
    setup = {"table_s": time.perf_counter() - start}
    anchor = None
    policies = methods(args)
    if any(m.mode in ANCHOR_MODES for m in policies):
        from .anchor_refinement import AnchorTable
        anchor = AnchorTable.build(model, tables, bins=args.anchor_bins)
        setup.update(anchor_s=anchor.build_seconds, anchor_bytes=anchor.nbytes,
                     anchor_bins=anchor.bins)
    cases = [r for r in previous["results"] if r["policy"] == "table" and r["split"] in args.splits
             and r["index"] < args.cases]
    budget_sources = {name: "reports/selective_refinement/evaluation.json" for name in args.splits}
    if "confirmation" in args.splits:
        start = time.perf_counter()
        calibration = make_cases(model, tables, CONFIRMATION_SEED, args.cases)
        setup["confirmation_calibration_s"] = time.perf_counter() - start
        cases.extend({"index": j, "split": "confirmation", "ic": ic, "budget": budget}
                     for j, (ic, _, budget) in enumerate(calibration))
        budget_sources["confirmation"] = f"30% of all-trust, full-local rollout; seed {CONFIRMATION_SEED}"
    if len(cases) != len(set(args.splits)) * args.cases:
        raise ValueError("Reference is missing selected initial conditions")
    config = {"setup": setup, "N": N, "steps": STEPS, "lam": LAM, "bins": 16, "model_hash": model.digest,
              "budget_sources": budget_sources,
              "budget_fraction": previous["config"]["budget_fraction"], "policy": "pacing",
              "methods": {m.label: asdict(m) for m in policies}, "repeats": args.repeats,
              "case_count_per_split": args.cases, "splits": args.splits, "schedule_seed": args.seed,
              "python": platform.python_version(), "numpy": np.__version__, "platform":platform.platform(),
              "threads_env": {v: os.getenv(v) for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")}}
    rows, audits = [], []
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        data = {"config": config, "results": rows, "audits": audits, "summary": summarize(rows)}
        temp = args.out.with_suffix(".tmp")
        temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        temp.replace(args.out)

    first = cases[0]
    warm_state = discretize(first["ic"], N)
    for m in policies:
        run_case(model, tables, warm_state, first["budget"], m, anchor=anchor, quick=True)
    generator = np.random.default_rng(args.seed)
    for case in cases:
        state = discretize(case["ic"], N)
        samples = defaultdict(list)
        for rep in range(args.repeats):
            for j in generator.permutation(len(policies)):
                m = policies[j]
                samples[m].append(run_case(model, tables, state, case["budget"], m, anchor=anchor))
        for m in policies:
            trial = samples[m]
            if len({(r["accepted"], r["spent"], r["unique_faces"]) for r in trial}) != 1:
                raise AssertionError("Nondeterministic decisions across timing repeats")
            rows.append({"split": case["split"], "index": case["index"], "ic": case["ic"],
                         "budget": case["budget"], "method": m.label,
                         "accepted": trial[0]["accepted"], "spent": trial[0]["spent"],
                         "unique_faces": trial[0]["unique_faces"],
                         "anchor_faces": trial[0]["anchor_faces"], "point_faces": trial[0]["point_faces"],
                         **{k: [r[k] for r in trial] for k in ("wall_s", "certificate_s")}})
        save()
        print(f"Finished {case['split']} case {case['index']} ({len(policies)} methods)", flush=True)
    for case in cases:
        if case["index"] >= args.audit_cases:
            continue
        for m in policies:
            if m.mode not in ("selective_point","selective_replay","selective_anchor","anchor_then_point") or m.rule not in ("block","uncertainty") or m.count != 10:
                continue
            state = discretize(case["ic"], N)
            audited = run_case(model, tables, state, case["budget"], m, anchor=anchor, audit=True)
            recorded = next(r for r in rows if r["split"] == case["split"] and r["index"] == case["index"]
                            and r["method"] == m.label)
            if any(recorded[k] != audited[k] for k in ("accepted", "spent", "unique_faces")):
                raise AssertionError("Audited run differs from measured run")
            audits.append({"split": case["split"], "index": case["index"], "method": m.label,
                           "steps": STEPS, "one_step_and_cumulative_checks_passed": True})
    save()


def trace_query_overlap(source, destination):
    """Replay saved budgets and distinguish trusted faces from queried-and-trusted faces."""
    if source.resolve() == destination.resolve():
        raise ValueError("Trace output must differ from the timed dataset")
    data = json.loads(source.read_text())
    cfg = data["config"]
    torch.set_num_threads(1)
    model = CertifiedFlux(FrozenFlux.load(MODEL_PATH))
    if (cfg["model_hash"], cfg["N"], cfg["steps"], cfg["lam"]) != (model.digest, N, STEPS, LAM):
        raise ValueError("Trace requires the same model, grid and horizon as the timed run")
    methods_by_name = {name: Method(**spec) for name, spec in cfg["methods"].items()}
    tables = CertificateTables.build(model, bins=cfg["bins"])
    anchor = None
    if any(m.mode in ANCHOR_MODES for m in methods_by_name.values()):
        from .anchor_refinement import AnchorTable
        anchor = AnchorTable.build(model, tables, bins=cfg["setup"]["anchor_bins"])
    rows = []
    counts = ("accepted", "point_faces", "trusted_with_point_query", "trusted_without_point_query")
    for record in data["results"]:
        m = methods_by_name[record["method"]]
        state = discretize(record["ic"], N)
        result = run_case(model, tables, state, record["budget"], m, anchor=anchor, trace_overlap=True)
        for k in ("accepted", "spent", "unique_faces", "point_faces"):
            if k in record and result[k] != record[k]:
                raise AssertionError("Replay differs from the timed run")
        rows.append({"split": record["split"], "index": record["index"], "method": m.label,
                     **{k: result[k] for k in counts}})
    groups = defaultdict(list)
    for row in rows:
        groups[row["split"], row["method"]].append(row)
    summary = defaultdict(dict)
    for (split, name), cases in groups.items():
        totals = {k: sum(row[k] for row in cases) for k in counts}
        total_faces = N*STEPS*len(cases)
        summary[split][name] = {"face_decisions": total_faces, **totals,
                                "fraction_trusted_without_point_query": totals["trusted_without_point_query"]/total_faces}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"source_file": source.name, "results": rows, "summary": summary},
                                      indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-query-overlap", type=Path, metavar="TIMED_JSON")
    parser.add_argument("--out", type=Path, default=Path("outputs/verifier_allocation/evaluation.json"))
    parser.add_argument("--splits", nargs="+", choices=("e3","fresh","confirmation"), default=["fresh"])
    parser.add_argument("--cases", type=int, choices=range(1,10), default=9)
    parser.add_argument("--counts", nargs="+", type=int, default=[4,10,20])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=DEFAULT_MODES)
    parser.add_argument("--sweep-modes", nargs="+", choices=MODES[2:], default=["selective_point", "selective_anchor"])
    parser.add_argument("--only-methods", nargs="+", type=parse_method, metavar="MODE/RULE/COUNT")
    parser.add_argument("--rules", nargs="+", choices=RANKERS, default=list(RANKERS))
    parser.add_argument("--full-point", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--audit-cases", type=int, choices=range(10), default=3)
    parser.add_argument("--anchor-bins", type=int, default=128)
    parser.add_argument("--seed", type=int, default=4401)
    args = parser.parse_args()
    if args.repeats <= 0 or any(k <= 0 or k >= N for k in args.counts):
        parser.error("Use positive repeats and counts strictly between 0 and N; the table and full policies cover the endpoints")
    if args.trace_query_overlap:
        trace_query_overlap(args.trace_query_overlap, args.out)
    else:
        run(args)
