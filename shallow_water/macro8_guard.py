"""Calibrate the whole-field guard on independent eight-step trajectories."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np

from .benchmark import FLAGS
from .macro8_data import ReferenceSolver, make_split, SPLIT_SEEDS

PACKAGE = Path(__file__).resolve().parent
REPORT = PACKAGE.parent / "reports" / "shallow_water_macro8" / "guard.json"
ERROR_TARGETS = np.array([0.01, 0.04])
MODELS = ("initial", "uniform", "hard")


def build_guard(folder, predictor=None):
    """Build the same prediction, postcheck and fallback used for timed routing."""
    source = PACKAGE / "macro8_predict.c" if predictor is None else Path(predictor)
    binary = Path(folder) / "macro8_guard.so"
    subprocess.run(["gcc", *FLAGS, str(source), str(PACKAGE/"macro8_guard.c"),
                    str(PACKAGE/"rollout.c"), str(PACKAGE/"solver.c"),
                    "-I", str(PACKAGE), "-lm", "-o", str(binary)], check=True)
    lib = ctypes.CDLL(str(binary))
    grid = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS")
    score = ctypes.POINTER(ctypes.c_double)
    lib.sw_macro8_predict.argtypes = [ctypes.c_int, grid, grid, ctypes.c_int]
    lib.sw_macro8_check.argtypes = [ctypes.c_int, grid, grid, ctypes.c_double, score]
    lib.sw_macro8_routed.argtypes = [ctypes.c_int, grid, grid, ctypes.c_int,
                                    ctypes.c_double, score]
    for kernel in (lib.sw_macro8_check, lib.sw_macro8_predict, lib.sw_macro8_routed):
        kernel.restype = ctypes.c_int
    return lib


def fit_cutoff(risks, pass_accuracy):
    """Largest observed score below every inaccurate finite-risk trajectory.

    Ties are rejected as a group. The cutoff never exceeds the largest
    finite, accurate score in the calibration data.
    """
    risks = np.asarray(risks)
    safe = np.asarray(pass_accuracy, dtype=bool)
    eligible = np.isfinite(risks) & safe
    observed_bad = risks[np.isfinite(risks) & ~safe]
    if observed_bad.size:
        eligible &= risks < observed_bad.min()
    return float(np.max(risks[eligible])) if np.any(eligible) else -1.0


def calibrate(count, output):
    with TemporaryDirectory() as folder, ReferenceSolver() as reference:
        lib = build_guard(folder)
        data = make_split("calibrate", count, reference)
        models, thresholds = {}, {}
        for model, name in enumerate(MODELS):
            risks = np.full(count, np.inf)
            errors = np.full((count,2), np.inf)
            for i, (q,target) in enumerate(zip(data.q, data.target)):
                pred = np.empty_like(q)
                if lib.sw_macro8_predict(len(q), q, pred, model):
                    continue
                risk = ctypes.c_double()
                lib.sw_macro8_check(len(q), q, pred, np.inf, ctypes.byref(risk))
                risks[i] = risk.value
                errors[i] = np.abs(pred-target).mean(axis=0)
            valid = (errors<=ERROR_TARGETS).all(axis=1)
            cut = fit_cutoff(risks,valid)
            accepted = np.isfinite(risks) & (risks<=cut)
            rows = [{"id":tid, "family":family,
                     "risk":float(score) if np.isfinite(score) else None,
                     "mae":e.tolist() if np.isfinite(e).all() else None,
                     "within_target":bool(ok), "accepted":bool(passed)}
                    for tid,family,score,e,ok,passed in zip(
                        data.ids, data.families, risks, errors, valid, accepted)]
            models[name] = dict(max_risk=cut, hard_check_rejections=int((~np.isfinite(risks)).sum()),
                                eligible=int(valid.sum()), accepted=int(accepted.sum()),
                                false_accepts=int((accepted & ~valid).sum()), rows=rows)
            thresholds[str(model)] = cut
    names = ["macro8_guard.c", "macro8_guard.py", "macro8_data.py", "macro8_predict.c",
             "macro8_weights.h", "rollout.c", "solver.c"]
    report = dict(
        split="calibrate", seed=SPLIT_SEEDS["calibrate"], n=count,
        block_mae_target={"depth_m":0.01, "discharge_m2_s":0.04},
        acceptance="Positive finite fields; eight equal starting cells on each boundary; mass and momentum balance with boundary flux; empirical patch score at or below the model-specific frozen cutoff.",
        score="Every two cell centres, compare change of cell-integrated state with a trapezoid-in-time estimate of net physical flux. Sum absolute weak residuals, divide by cell count and component target, take maximum over the two components.",
        calibration_rule="Largest observed finite risk score strictly below all inaccurate calibration cases and attained by an accurate case; if none qualify use -1 (reject all).",
        limitations="A small weak residual can coincide with a wrong solution. A threshold fitted on the independent calibration set is not a deterministic nor probabilistic guarantee of block or trajectory error.",
        sha256={n:hashlib.sha256((PACKAGE/n).read_bytes()).hexdigest() for n in names},
        max_risk=thresholds["2"], thresholds=thresholds, models=models)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--out", type=Path, default=REPORT)
    args = parser.parse_args()
    result = calibrate(args.count, args.out)
    for name, stats in result["models"].items():
        print(name, {k:v for k,v in stats.items() if k!="rows"})
