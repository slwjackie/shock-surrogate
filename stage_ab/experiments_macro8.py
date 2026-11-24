"""Evaluate frozen H=8 predictors at matched physical times on periodic Burgers."""
import argparse
import json
import os
import platform
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from certified_burgers.godunov import advance
from .burgers_entropy_reference import entropy_averages
from .experiments_a2 import discretize, ic_family
from .macro_controls import fixed_macro_step
from .macro_flux import MacroFlux
from .macro_network import load_network

ROOT = Path(__file__).resolve().parents[1]
N = 128
HORIZONS = (8, 60, 64)
LAM = 0.2
ERROR_FACTOR = 1.05


def cases():
    path = ROOT / "reports/verifier_allocation/confirmation.json"
    old = json.loads(path.read_text())["results"]
    records = [("confirmation", r["index"], r["ic"], r["budget"])
               for r in old if r["method"] == "selective_anchor/all/128"]
    records.extend(("fresh", i, ic, None) for i, ic in enumerate(ic_family(79023, 30)))
    return [(split, i, ic, budget, discretize(ic, N)) for split, i, ic, budget in records]


def fast_godunov(u, steps, lam=LAM):
    p = np.empty(u.size + 1)
    p[:-1] = u
    p[-1] = p[0]
    for _ in range(steps):
        speed = np.maximum(np.maximum(p[:-1], -p[1:]), 0.0)
        f = 0.5 * speed * speed
        p[1:-1] -= lam * (f[1:] - f[:-1])
        p[0] -= lam * (f[0] - f[-1])
        p[-1] = p[0]
    return p[:-1]


def rollout(step, u, h):
    for _ in range(h // 8):
        u = step(u)
    return fast_godunov(u, h % 8) if h % 8 else u


def mae(x, y):
    return float(np.mean(np.abs(x - y)))


def evaluate(records, predictors):
    rows = []
    for split, i, ic, budget, u in records:
        x0 = u.copy()
        teacher = {h: fast_godunov(u, h) for h in HORIZONS}
        ref = {h: entropy_averages(u, h * LAM / N) for h in HORIZONS}
        states = {name: {h: rollout(step, u, h) for h in HORIZONS}
                  for name, step in predictors.items()}
        states["godunov_dt4"] = {h: fast_godunov(u, h // 4, 4 * LAM) for h in HORIZONS}
        states["godunov"] = teacher
        for h in HORIZONS:
            original = advance(u, 1 / N, h, dt=LAM / N)[0]
            np.testing.assert_array_equal(original, teacher[h])
            assert 4 * LAM * np.max(np.abs(u)) <= 1
            for name in states:
                v = states[name][h]
                if not np.isfinite(v).all():
                    raise ValueError(f"Nonfinite rollout: {name}, {split}/{i}, {h}")
                gap = mae(v, teacher[h])
                e = mae(v, ref[h])
                eg = mae(teacher[h], ref[h])
                rows.append(dict(split=split, index=i, kind=ic["kind"], steps=h, model=name,
                                 same_grid_mae=gap, entropy_mae=e, godunov_entropy_mae=eg,
                                 matched=e <= ERROR_FACTOR * eg + 1e-12,
                                 no_worse=e <= eg + 1e-12, mass_drift=float(abs(v.mean()-u.mean())),
                                 budget=budget, terminal_budget_pass=gap <= budget if h==60 and budget is not None else None))
        np.testing.assert_array_equal(u, x0)
    return rows


def audit_existing_budgets(records, predictors):
    """Observed grid discrepancies at available times through step 60."""
    audit = []
    for split, index, ic, budget, initial in records:
        if budget is None:
            continue
        for name, step in predictors.items():
            ref = initial
            state = initial
            errors = {}
            for t in (*range(8, 57, 8), 57, 58, 59, 60):
                if t <= 56:
                    state = step(state)
                    ref = fast_godunov(ref, 8)
                else:
                    state = fast_godunov(state, 1)
                    ref = fast_godunov(ref, 1)
                errors[t] = mae(state, ref)
            audit.append(dict(index=index, kind=ic["kind"], model=name, budget=budget,
                              errors=errors, passed=max(errors.values()) <= budget))
    return audit


def summary(rows):
    result = []
    for split in ("confirmation", "fresh"):
        for h in HORIZONS:
            for name in sorted({r["model"] for r in rows}):
                subset = [r for r in rows if (r["split"], r["steps"], r["model"]) == (split, h, name)]
                e = np.mean([r["entropy_mae"] for r in subset])
                eg = np.mean([r["godunov_entropy_mae"] for r in subset])
                result.append(dict(split=split, steps=h, model=name, mean_entropy_mae=float(e),
                                   error_ratio=float(e/eg), matched=sum(r["matched"] for r in subset),
                                   no_worse=sum(r["no_worse"] for r in subset), count=len(subset),
                                   mean_same_grid_mae=float(np.mean([r["same_grid_mae"] for r in subset]))))
    return result


def time_methods(records, predictors, rounds=9):
    # Each sample covers the same nine single-trajectory jobs. No batching across ICs.
    states = [u for split, i, ic, budget, u in records if split == "confirmation"]
    rng = np.random.default_rng(2781)
    results = {}
    for h in HORIZONS:
        calls = {"godunov_public": lambda u: advance(u, 1/N, h, dt=LAM/N)[0],
                 "godunov": lambda u: fast_godunov(u, h),
                 "godunov_dt4": lambda u: fast_godunov(u, h//4, 4*LAM)}
        for name, step in predictors.items():
            calls[name] = lambda u, step=step: rollout(step, u, h)
        samples = {name: [] for name in calls}
        for f in calls.values():
            for u in states:
                f(u)
        for _ in range(rounds):
            for name in rng.permutation(list(calls)):
                f = calls[name]
                start = perf_counter_ns()
                for u in states:
                    f(u)
                samples[name].append((perf_counter_ns()-start)/len(states)/1000)
        results[h] = {name: dict(median_us=float(np.median(times)), samples_us=times) for name,times in samples.items()}
    return results


def run(path):
    records = cases()
    predictors = {"mlp": load_network(), "learned_stages": MacroFlux.load().step, "fixed_stages": fixed_macro_step}
    rows = evaluate(records, predictors)
    payload = dict(n=N, dt=LAM/N, horizons=HORIZONS, fresh_seed=79023, relative_error_factor=ERROR_FACTOR,
                   absolute_roundoff_allowance=1e-12, python=platform.python_version(),
                   numpy=np.__version__, threads={k: os.getenv(k) for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")},
                   initial_conditions=[dict(split=s, index=i, ic=ic, budget=b) for s,i,ic,b,u in records],
                   summary=summary(rows), budget_audit=audit_existing_budgets(records, predictors),
                   numpy_timing=time_methods(records, predictors), evaluations=rows)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "reports/macro8/evaluation.json")
    run(parser.parse_args().out)
