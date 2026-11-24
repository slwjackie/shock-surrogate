"""Secondary endpoint-error operating point for the frozen macro8 models."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter_ns

import numpy as np

from .macro8_benchmark import HERE, METHODS, MODEL_NAMES, OUT, build
from .macro8_data import FAMILIES, N, SPLIT_SEEDS, ReferenceSolver, make_split
from .macro8_guard import fit_cutoff

# Fixed after the primary training/calibration yielded no calibration failures.
# The stricter teacher deviations were chosen without consulting held-out errors.
STRICT_MAE = np.array([0.001, 0.004])
COMPARED = ("teacher8", "godunov8", "engineering_cfl", "hlle_cfl", "rusanov_cfl",
            "routed_initial", "routed_uniform", "routed_hard")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calibrate(path):
    raw = json.loads(path.read_text())
    if raw["split"] != "calibrate" or raw["seed"] != SPLIT_SEEDS["calibrate"] or raw["n"] != 256:
        raise ValueError("Expected the 256-profile independent calibration split")
    for filename, version in raw["sha256"].items():
        if digest(HERE/filename) != version:
            raise ValueError(f"Calibration does not match current source: {filename}")
    cuts, counts = {}, {}
    for name in MODEL_NAMES:
        rows = raw["models"][name]["rows"]
        risk = np.array([np.inf if row["risk"] is None else row["risk"] for row in rows])
        errors = np.array([row["mae"] if row["mae"] is not None else [np.inf, np.inf]
                           for row in rows])
        if len(rows) != raw["n"]:
            raise ValueError(f"Incomplete calibration cases for {name}")
        accurate = np.all(errors <= STRICT_MAE, axis=1)
        cut = fit_cutoff(risk, accurate)
        accepted = np.isfinite(risk) & (risk <= cut)
        if np.any(accepted & ~accurate):
            raise RuntimeError(f"{name} failed the zero calibration false-acceptance rule")
        cuts[name] = cut
        counts[name] = dict(cutoff=cut, within_budget=int(accurate.sum()),
                            proposals_over_budget=int((~accurate).sum()),
                            accepted=int(accepted.sum()), accepted_over_budget=0)
    return cuts, counts


def measure(lib, data, cuts, rounds, rng):
    q, target = data.q, data.target
    count = len(q)
    out = np.empty_like(q)
    record = np.empty((count, 3), np.int32)
    bare, info = {}, {}
    for name in MODEL_NAMES:
        status = lib.sw_macro8_batch(count, N, q, out, METHODS.index(name), 0., record)
        if status:
            raise RuntimeError(f"{name} prediction returned {status}")
        bare[name] = out.copy()
    times = {key: [] for key in COMPARED}
    for rep in range(-2, rounds):
        for idx in rng.permutation(len(COMPARED)):
            key = COMPARED[idx]
            name = key.removeprefix("routed_")
            cut = cuts.get(name, 0.)
            t0 = perf_counter_ns()
            status = lib.sw_macro8_batch(count, N, q, out, METHODS.index(key), cut, record)
            elapsed = (perf_counter_ns()-t0)/(1000*count)
            if status:
                raise RuntimeError(f"{key} returned {status}")
            if rep >= 0:
                times[key].append(elapsed)
            if rep != -2:
                continue
            if not np.isfinite(out).all():
                raise ArithmeticError(f"{key} produced nonfinite cells")
            errors = np.abs(out-target).mean(axis=1)
            within = np.all(errors <= STRICT_MAE, axis=1)
            row = dict(within_budget=int(within.sum()), output_mae=errors.mean(axis=0).tolist())
            if key.startswith("routed_"):
                acc = record[:,0] == 1
                if not np.isin(record[:,0], [0, 1]).all():
                    raise ArithmeticError("Unknown native route")
                np.testing.assert_array_equal(out[acc], bare[name][acc])
                np.testing.assert_array_equal(out[~acc], target[~acc])
                proposal_error = np.abs(bare[name]-target).mean(axis=1)
                unsafe = np.any(proposal_error > STRICT_MAE, axis=1)
                row.update(accepted=int(acc.sum()), accept_rate=float(acc.mean()),
                           unsafe_proposed=int(unsafe.sum()), unsafe_accepted=int((unsafe & acc).sum()),
                           unsafe_rejected=int((unsafe & ~acc).sum()), safe_rejected=int((~unsafe & ~acc).sum()),
                           safe_accepted=int((~unsafe & acc).sum()),
                           accepted_by_family={
                               family:int(np.sum(acc & (np.array(data.families) == family)))
                               for family in FAMILIES},
                           inaccurate_accepted_cases=[
                               {"id":data.ids[i], "family":data.families[i],
                                "proposal_mae":proposal_error[i].tolist()}
                               for i in np.flatnonzero(acc & unsafe)])
            info[key] = row
    teacher_time = float(np.median(times["teacher8"]))
    for key, samples in times.items():
        t = float(np.median(samples))
        info[key].update(median_us=t, time_iqr_us=np.quantile(samples, [.25,.75]).tolist(),
                         teacher_time_over_method=teacher_time/t, samples_us=samples)
    return info


def study(output, rounds=15, cpu=None):
    if rounds < 1:
        raise ValueError("Positive number of repetitions required")
    cpu_set = None
    if hasattr(os, "sched_getaffinity"):
        default = min(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {default if cpu is None else cpu})
        cpu_set = sorted(os.sched_getaffinity(0))
    rng = np.random.default_rng(319947)
    cal_path = OUT/"guard.json"
    cuts, calibration = calibrate(cal_path)
    panels = {}
    with TemporaryDirectory() as d, ReferenceSolver() as reference:
        lib = build(Path(d))
        for split in ("test", "shift"):
            data = make_split(split, 128, reference)
            panels[split] = measure(lib, data, cuts, rounds, rng)
    result = dict(
        scope="Secondary single-block operating point. The 10x tighter component thresholds were fixed after observing no primary inaccuracy on the independent calibration cases, before selecting any cutoff using held-out endpoint errors.",
        target="Both per-profile spatial mean absolute deviations from the 256-cell, eight-step tight Godunov reference must meet their respective bounds.",
        mae_budget={"depth_m": float(STRICT_MAE[0]), "discharge_m2_s": float(STRICT_MAE[1])},
        calibration_rule="Same finite-score prefix as the primary guard, zero inaccurate calibration accepts; fallback everywhere if no valid prefix.",
        training="Frozen existing model weights and raw guard score including its original normalizations.",
        count_per_panel=128, calibration_count=256, affinity=cpu_set, rounds=rounds,
        timing="Two warmups, shuffled native methods within each repeat, one Python call per 128 independent 256-cell profiles, microseconds per profile.",
        seeds={k: SPLIT_SEEDS[k] for k in ("calibrate","test","shift")},
        sha256={"guard.json":digest(cal_path), **{p: digest(HERE/p) for p in (
            "macro8_strict.py","macro8_predict.c","macro8_guard.c","macro8_weights.h",
            "macro8_workload.c","macro8_classical.c","rollout.c","solver.c")}},
        calibration=calibration, panels=panels)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT/"strict.json")
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--rounds", type=int, default=15)
    args = parser.parse_args()
    result = study(args.out, args.rounds, args.cpu)
    print("calibration",result["calibration"])
    for split, rows in result["panels"].items():
        for name in MODEL_NAMES:
            print(split, name, {k:v for k,v in rows["routed_"+name].items()
                                if k not in ("samples_us", "time_iqr_us")})
