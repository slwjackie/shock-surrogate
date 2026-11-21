"""Stage A2 experiments (docs/stage_a2_theory.md, reports/stage_a2_validation.md).

python -m stage_ab.experiments_a2 --out outputs/stage_a2 [--quick]

E1 tightness vs resolution   one-step bound/actual for the separable (original),
                             table-difference and local-difference certificates
E2 trust frontier / x-t map  how many faces can be trusted per unit of certified error
E3 budget policies           online allocation of a global certified budget on
                             Burgers rollouts, with hindsight LP bounds
E4 adversarial menus         the lower-bound instances of Propositions K4/K5
E5 cost                      wall-clock breakdown (no speed claim is made)
All oracles (exact rational Godunov) are used for measurement only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import torch

from certified_burgers.godunov import godunov_step
from certified_burgers.initial_conditions import smooth_state, riemann_state
from .burgers import FrozenFlux
from .stage_a2 import (CertifiedFlux, CertificateTables, FPConstants, cell_costs, trust_frontier,
                       frontier_upper, rigorous_step_cost, exact_actual_face_error, mixed_step,
                       certified_rollout_a2, train_flux_v2, fit_report, budget_parameters,
                       local_regularity, pairs)
from .burgers import exact_godunov
from fractions import Fraction as Q
from .budget import (GreedyPolicy, AllOrNothingPolicy, PacingPolicy, ThresholdPolicy,
                     RobustifiedPolicy, DensityThresholdPolicy, AllOrNothingWrapper,
                     run_policy_on_menus, lp_dual_bound)

LAM = 0.2
ENVELOPE = 2.0


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(float(x)) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(payload), indent=1), encoding="utf-8")


# ------------------------------------------------------------ initial data
def ic_family(seed, count):
    """Resolution-independent initial conditions: each is a function of N."""
    rng = np.random.default_rng(seed)
    out = []
    for j in range(count):
        kind = ["smooth", "shocked", "riemann"][j % 3]
        if kind == "riemann":
            a, b = rng.uniform(-1, 1, 2)
            if abs(a-b) < .4:
                b = float(np.clip(a - np.sign(a-b+1e-12)*.6, -1, 1))
            out.append({"kind": kind, "left": float(a), "right": float(b), "loc": float(rng.uniform(.25, .75))})
        else:
            out.append({"kind": kind, "amp": float(rng.uniform(.4, 1.)), "offset": float(rng.uniform(-.15, .15)),
                        "phase": float(rng.uniform(0, 2*np.pi)), "mode": int(rng.integers(1, 3)),
                        "t": 0.0 if kind == "smooth" else float(rng.uniform(.3, .5))})
    return out


def discretize(ic, n):
    if ic["kind"] == "riemann":
        return riemann_state(n, ic["left"], ic["right"], location=ic["loc"])
    v = np.clip(smooth_state(n, amplitude=ic["amp"], offset=ic["offset"], phase=ic["phase"], mode=ic["mode"]), -1, 1)
    steps = int(round(ic["t"]/(LAM/n)))
    for _ in range(steps):
        v, _ = godunov_step(v, 1.0, dt=LAM)
    return v


# ------------------------------------------------------------------- model
def get_model(out: Path, seed=0, width=16, steps=6000):
    path = out/f"flux_w{width}_s{seed}.json"
    if path.exists():
        return CertifiedFlux(FrozenFlux.load(path)), json.loads((out/f"flux_w{width}_s{seed}_fit.json").read_text())
    t = time.perf_counter()
    m, losses = train_flux_v2(width=width, steps=steps, seed=seed, envelope=ENVELOPE)
    frozen = m.freeze()
    frozen.save(path)
    model = CertifiedFlux(frozen)
    fit = {**fit_report(model), "train_sec": time.perf_counter()-t, "final_loss": losses[-1],
           "width": width, "steps": steps, "seed": seed}
    write(out/f"flux_w{width}_s{seed}_fit.json", fit)
    return model, fit


# ---------------------------------------------------------------------- E1
def e1_tightness(model, tables_by_bins, ics, ns, seed=0):
    consts = FPConstants.build(LAM, ENVELOPE)
    t16 = tables_by_bins[min(tables_by_bins)]
    rows = []
    for j, ic in enumerate(ics):
        for n in ns:
            v = discretize(ic, n)
            actual = exact_actual_face_error(model, v, LAM)
            row = {"seed": seed, "ic": j, "kind": ic["kind"], "n": n, "actual": float(actual)}
            for bins, tab in tables_by_bins.items():
                for mode in ("separable", "table", "local"):
                    if mode == "local" and bins != min(tables_by_bins):
                        continue
                    costs = cell_costs(model, tab, consts, v, mode=mode)
                    b = rigorous_step_cost(costs, np.ones(n, bool), LAM, n)
                    if b < actual:
                        raise AssertionError("certificate violated")
                    key = f"{mode}" if mode == "local" else f"{mode}_b{bins}"
                    row[key] = float(b)
            row.update(r3_decomposition(model, t16, consts, v))
            rows.append(row)
    return rows


def r3_decomposition(model, tables, consts, v):
    """Per-cell check of Theorem R3: on regular cells the local certificate exceeds
    the exact face-error difference by at most d_branch^2 + 2(eps_(i-1)+eps_i)
    (+ outward-rounding widening, reported as the residual)."""
    n = len(v)
    kappa = cell_costs(model, tables, consts, v, mode="local")[:, 1, 1]
    regular, branch = local_regularity(model, v)
    pred = model.predict(pairs(v))
    vq = [Q(float(x)) for x in v]
    e = [Q(float(pred[i])) - exact_godunov(vq[i], vq[(i+1) % n]) for i in range(n)]
    idx = tables.bin_index(v)
    eps_face = tables.eps_fp[idx, np.roll(idx, -1)]
    lam_h = Q(LAM)/n
    gap_reg, bound_reg, gap_irr, worst_resid = Q(0), Q(0), Q(0), 0.0
    for i in range(n):
        act = abs(e[i]-e[i-1])
        gap = Q(float(kappa[i])) - act
        if regular[i]:
            d = (vq[i]-vq[i-1]) if branch[i] == 0 else (vq[(i+1) % n]-vq[i])
            allow = d*d + 2*(Q(float(eps_face[i-1])) + Q(float(eps_face[i])))
            gap_reg += gap
            bound_reg += allow
            worst_resid = max(worst_resid, float((gap-allow)/max(Q(float(kappa[i])), Q(1, 10**300))))
        else:
            gap_irr += gap
    return {"regular_cells": int(regular.sum()), "irregular_cells": int(n-regular.sum()),
            "gap_regular": float(lam_h*gap_reg), "r3_allowance_regular": float(lam_h*bound_reg),
            "gap_irregular": float(lam_h*gap_irr), "r3_worst_relative_residual": worst_resid}


# ---------------------------------------------------------------------- E2
def e2_frontier(model, tables, ics, n):
    consts = FPConstants.build(LAM, ENVELOPE)
    out = []
    for j, ic in enumerate(ics):
        v = discretize(ic, n)
        rec = {"ic": j, "kind": ic["kind"], "n": n}
        for mode in ("separable", "table", "local"):
            fr = trust_frontier(cell_costs(model, tables, consts, v, mode=mode))
            c = frontier_upper(fr, LAM, n)
            rec[mode] = c.tolist()
        out.append(rec)
    return out


def e2_trust_map(model, tables, ic, n, steps, budget_fraction):
    """x-t map of trusted faces under greedy allocation of a per-step allowance."""
    consts = FPConstants.build(LAM, ENVELOPE)
    v = discretize({**ic, "t": 0.0} if ic["kind"] != "riemann" else ic, n)
    trust_rows, states = [], []
    for _ in range(steps):
        costs = cell_costs(model, tables, consts, v, mode="local")
        fr = trust_frontier(costs)
        c = frontier_upper(fr, LAM, n)
        allowance = budget_fraction*c[-1]
        k = int(np.max(np.flatnonzero(c <= max(allowance, c[0]))))
        s = fr.select(k)
        trust_rows.append(s.astype(int).tolist())
        states.append(v.tolist())
        v, _, _, _ = mixed_step(model, consts, v, s)
    return {"trust": trust_rows, "states": states, "budget_fraction": budget_fraction, "n": n}


# ---------------------------------------------------------------------- E3
def _policies(params, total, steps, price):
    L, U = params["L"], params["U"]
    pol = {
        "all_or_nothing": lambda: AllOrNothingPolicy(),
        "greedy": lambda: GreedyPolicy(),
        "greedy_capped": lambda: GreedyPolicy(cap=2*params["excess"]/steps),
        "pacing": lambda: PacingPolicy(),
        "pacing_all_or_nothing": lambda: AllOrNothingWrapper(PacingPolicy()),
        "threshold": lambda: ThresholdPolicy(L, U),
        "learned_price": lambda: DensityThresholdPolicy(price),
        "robust_learned(0.25)": lambda: RobustifiedPolicy(ThresholdPolicy(L, U), DensityThresholdPolicy(price), .25),
    }
    for g in GAMMAS:
        pol[f"robust_pacing({g})"] = (lambda g: lambda: RobustifiedPolicy(ThresholdPolicy(L, U), PacingPolicy(), g))(g)
    return pol


GAMMAS = (0.1, 0.25, 0.5)


def e3_budget(model, tables, calib_ics, test_ics, n, steps, fractions, audit_every=1):
    rows = []
    for frac in fractions:
        # Reference: full-trust total certified cost, per initial condition.
        def full_cost(ic):
            r = certified_rollout_a2(model, tables, discretize(ic, n), steps=steps, lam=LAM, total_budget=10.,
                                     policy=GreedyPolicy(), mode="local", audit=False)
            return float(r["spent"]), r
        # Learned price from calibration ICs (disjoint from test ICs).
        prices = []
        for ic in calib_ics:
            spent_full, r = full_cost(ic)
            total = frac*spent_full
            params = budget_parameters(tables, LAM, n, steps, total)
            vals = np.arange(n+1, dtype=np.float64)
            weights = [(vals, np.where(vals > 0, d + vals*params["c_min"], 0.0)) for d in r["raw_excess"]]
            _, beta = lp_dual_bound(weights, params["excess"], return_price=True)
            prices.append(beta)
        price = float(np.median(prices))
        for j, ic in enumerate(test_ics):
            spent_full, _ = full_cost(ic)
            total = frac*spent_full
            params = budget_parameters(tables, LAM, n, steps, total)
            for name, make in _policies(params, total, steps, price).items():
                r = certified_rollout_a2(model, tables, discretize(ic, n), steps=steps, lam=LAM,
                                         total_budget=total, policy=make(), mode="local",
                                         audit=(j % audit_every == 0))
                value = sum(x["trusted"] for x in r["rows"])
                lp = lp_dual_bound(r["menus"], r["excess_budget"])
                rec = {"fraction": frac, "ic": j, "kind": ic["kind"], "policy": name, "value": value,
                       "trusted_fraction": value/(n*steps), "lp_bound": lp,
                       "empirical_ratio": lp/max(value, 1e-12), "spent": float(r["spent"]),
                       "total_budget": total, "full_trust_cost": spent_full,
                       "excess": params["excess"], "L": params["L"], "U": params["U"],
                       "alpha": params["alpha"], "price": price,
                       "w_hat": max(float(np.max(w)) for _, w in r["menus"])/max(params["excess"], 1e-300),
                       "per_step_trusted": [x["trusted"] for x in r["rows"]]}
                if r["rows"] and "step_certificate_holds" in r["rows"][0]:
                    rec["all_step_certificates_hold"] = all(x["step_certificate_holds"] for x in r["rows"])
                    rec["all_global_certificates_hold"] = all(x["global_certificate_holds"] for x in r["rows"])
                    rec["max_step_ratio"] = max(x["eta"]/max(x["actual_step_error"], 1e-300) for x in r["rows"])
                rows.append(rec)
    return rows


# ---------------------------------------------------------------------- E4
def e4_adversarial(thetas=(10, 100, 1000, 10000), m=200, horizon=50):
    rows = []
    for theta in thetas:
        L, U, B = 1.0, float(theta), 1.0
        # Prop K4: low-density items first, then high-density ones.
        inst = ([(np.array([0., L*B/m]), np.array([0., B/m]))]*m +
                [(np.array([0., U*B/m]), np.array([0., B/m]))]*m)
        # Reversed order (greedy's favorable case) for contrast.
        rev = inst[m:] + inst[:m]
        # Prop K5: everything offered in one burst at the first step.
        grid = np.linspace(B/m, B, m)
        burst = [(np.r_[0., U*grid], np.r_[0., grid])] + [(np.array([0.]), np.array([0.]))]*(horizon-1)
        for name, menus in (("low_then_high", inst), ("high_then_low", rev), ("burst", burst)):
            opt = lp_dual_bound(menus, B)
            pols = [("greedy", GreedyPolicy()), ("pacing", PacingPolicy()), ("threshold", ThresholdPolicy(L, U))]
            pols += [(f"robust_pacing({g})", RobustifiedPolicy(ThresholdPolicy(L, U), PacingPolicy(), g)) for g in GAMMAS]
            for pname, pol in pols:
                v = run_policy_on_menus(pol, menus, B)["value"]
                rows.append({"theta": theta, "instance": name, "policy": pname, "value": v, "opt_lp": opt,
                             "ratio": opt/max(v, 1e-300), "alpha": 1+math.log(theta)})
    return rows


# ---------------------------------------------------------------------- E5
def _median_time(fn, reps):
    fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter()-t)
    return float(np.median(ts))


def e5_timing(model, tables, ns, reps=15):
    from .burgers import neural_step
    consts = FPConstants.build(LAM, ENVELOPE)
    rows = []
    for n in ns:
        v = discretize({"kind": "shocked", "amp": .8, "offset": .1, "phase": .3, "mode": 1, "t": .35}, n)
        costs = cell_costs(model, tables, consts, v, mode="local")
        fr = trust_frontier(costs)
        s = fr.select(n)
        rec = {"n": n,
               "godunov_step": _median_time(lambda: godunov_step(v, 1.0, dt=LAM), reps),
               "neural_step": _median_time(lambda: neural_step(model._frozen, v, LAM), reps),
               "cert_separable": _median_time(lambda: cell_costs(model, tables, consts, v, mode="separable"), reps),
               "cert_table": _median_time(lambda: cell_costs(model, tables, consts, v, mode="table"), reps),
               "cert_local": _median_time(lambda: cell_costs(model, tables, consts, v, mode="local"), reps),
               "frontier_dp": _median_time(lambda: trust_frontier(costs), max(3, reps//3)),
               "mixed_step_with_fp_bound": _median_time(lambda: mixed_step(model, consts, v, s), reps)}
        rows.append(rec)
    return rows


# -------------------------------------------------------------------- main
def run(out, quick=False, seed=0):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    t0 = time.perf_counter()
    env = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
           "platform": platform.platform()}
    model, fit = get_model(out, seed=seed, width=16, steps=1500 if quick else 6000)
    tables = {b: CertificateTables.build(model, envelope=ENVELOPE, bins=b) for b in ((16,) if quick else (16, 32))}
    build = {b: t.build_seconds for b, t in tables.items()}
    ics = ic_family(seed+100, 6 if quick else 12)
    ns = (64, 128, 256) if quick else (64, 128, 256, 512, 1024)
    res = {"environment": env, "fit": fit, "table_build_sec": build, "config": {"lam": LAM, "envelope": ENVELOPE}}
    res["e1"] = e1_tightness(model, tables, ics, ns, seed=seed)
    res["fits"] = [fit]
    for extra in ((seed+1, seed+2) if not quick else ()):
        m2, f2 = get_model(out, seed=extra, width=16, steps=6000)
        t2 = {b: CertificateTables.build(m2, envelope=ENVELOPE, bins=b) for b in tables}
        res["e1"] += e1_tightness(m2, t2, ics, ns, seed=extra)
        res["fits"].append(f2)
    write(out/"e1_tightness.json", res["e1"])
    print("E1 done", round(time.perf_counter()-t0, 1), flush=True)
    t16 = tables[16]
    res["e2_frontier"] = e2_frontier(model, t16, ics, 256)
    shock_ic = {"kind": "smooth", "amp": .8, "offset": .1, "phase": .3, "mode": 1, "t": 0.0}
    res["e2_map"] = e2_trust_map(model, t16, shock_ic, 128, 160 if not quick else 60, .5)
    write(out/"e2.json", {"frontier": res["e2_frontier"], "map": res["e2_map"]})
    print("E2 done", round(time.perf_counter()-t0, 1), flush=True)
    calib = ic_family(seed+200, 3 if quick else 6)
    test = ic_family(seed+300, 3 if quick else 9)
    res["e3"] = e3_budget(model, t16, calib, test, 64 if quick else 128, 20 if quick else 60,
                          (0.3, 0.6), audit_every=1 if quick else 3)
    write(out/"e3_budget.json", res["e3"])
    print("E3 done", round(time.perf_counter()-t0, 1), flush=True)
    res["e4"] = e4_adversarial()
    write(out/"e4_adversarial.json", res["e4"])
    res["e5"] = e5_timing(model, t16, (128, 512) if not quick else (128,))
    write(out/"e5_timing.json", res["e5"])
    res["total_sec"] = time.perf_counter()-t0
    write(out/"results.json", res)
    print("done", round(res["total_sec"], 1), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a.out, quick=a.quick, seed=a.seed)


if __name__ == "__main__":
    main()
