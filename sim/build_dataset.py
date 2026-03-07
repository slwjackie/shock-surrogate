#!/usr/bin/env python3
"""
Generalized dataset builder for Hybrid Transformer Burgers+reaction surrogate.

Key upgrade vs original:
  - Regime labeling is *condition-aware* (depends on PDE coefficients).
  - Calibration builds per-(nu,k,E) bucket reference statistics so that
    det/def/no thresholds remain meaningful under solver-mismatch OOD.

Why:
  Using a single global normalization (g_mean/std, fs_mean/std) can cause
  mismatch cases to become extreme outliers and collapse all labels to one class.

Outputs (same as original + extra):
  - data/meta.csv
  - data/u_{split}.npz
  - data/grid.npz
  - data/thresholds_by_coeff.json   (NEW)
"""

import os, sys, math, time, json, argparse
import numpy as np
import pandas as pd

# Robust import: allow running from sim/ or repo root
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from config import SimCfg, DataCfg
from sim.solver_burgers_weno import simulate_case

LABELS = ["no_detonation", "deflagration_like", "detonation_like"]
LABEL2ID = {k: i for i, k in enumerate(LABELS)}

# -----------------------------
# Utilities
# -----------------------------
def _get(obj, name, default):
    return getattr(obj, name) if hasattr(obj, name) else default

def _lin_edges(lo: float, hi: float, n_bins: int):
    if n_bins <= 1 or hi <= lo:
        return np.array([lo, hi], dtype=float)
    return np.linspace(lo, hi, n_bins + 1, dtype=float)

def _log_edges(lo: float, hi: float, n_bins: int):
    # For positive ranges only
    lo = max(lo, 1e-12)
    hi = max(hi, lo * 1.000001)
    if n_bins <= 1:
        return np.array([lo, hi], dtype=float)
    return np.exp(np.linspace(np.log(lo), np.log(hi), n_bins + 1, dtype=float))

def _bucketize(v: float, edges: np.ndarray) -> int:
    # returns bin index in [0, len(edges)-2]
    if len(edges) < 2:
        return 0
    idx = int(np.searchsorted(edges, v, side="right") - 1)
    return int(np.clip(idx, 0, len(edges) - 2))

def _coeff_key(nu: float, k: float, E: float, nu_edges, k_edges, E_edges) -> str:
    ib = _bucketize(nu, nu_edges)
    kb = _bucketize(k,  k_edges)
    Eb = _bucketize(E,  E_edges)
    return f"nuB{ib}_kB{kb}_EB{Eb}"

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

# -----------------------------
# Metrics (diagnostics + label)
# -----------------------------
def compute_diagnostics(U, x, t_norm, sc: SimCfg, t_end_actual: float):
    """
    Return diagnostics that are *independent of regime thresholds*.
    """
    U = np.asarray(U)  # (Nt, Nx)
    peak_u = float(np.max(U))
    dx = float(x[1] - x[0]) if len(x) > 1 else 1.0

    # gradient proxy
    dU = np.diff(U, axis=1) / max(dx, 1e-12)
    g_abs = np.abs(dU)
    gmax_t = g_abs.max(axis=1)  # (Nt,)
    g_peak = float(gmax_t.max())

    # shock position (x at max gradient)
    idx = g_abs.argmax(axis=1)
    shock_pos_t = x[idx]

    # front speed (linear fit over active region)
    active_thr = getattr(sc, "grad_active_threshold", None)
    if active_thr is None:
        active_thr = 0.6 * g_peak if g_peak > 0 else np.inf
    mask = gmax_t >= float(active_thr)
    front_speed = np.nan
    if np.count_nonzero(mask) >= 3:
        tt = (t_norm[mask] * t_end_actual).astype(np.float64)
        xx = shock_pos_t[mask].astype(np.float64)
        t0 = tt.mean()
        x0 = xx.mean()
        denom = np.sum((tt - t0) ** 2)
        if denom > 0:
            front_speed = float(np.sum((tt - t0) * (xx - x0)) / denom)

    fs_abs = float(abs(front_speed)) if not np.isnan(front_speed) else 0.0

    # run-up time based on u threshold (optional diagnostic)
    runup_u = np.nan
    u_runup_thr = getattr(sc, "u_runup_threshold", None)
    if u_runup_thr is not None:
        for i, tn in enumerate(t_norm):
            if float(np.max(U[i])) >= float(u_runup_thr):
                runup_u = float(tn * t_end_actual)
                break

    # gradient run-up time (optional diagnostic)
    # grad_runup_thr = getattr(sc, "grad_runup_threshold", None)
    # if grad_runup_thr is None:
    #     grad_runup_thr = 0.25 * g_peak if g_peak > 0 else 0.0
    # runup_g = np.nan
    # for i, tn in enumerate(t_norm):
    #     if float(gmax_t[i]) >= float(grad_runup_thr):
    #         runup_g = float(tn * t_end_actual)
    #         break

    # gradient run-up time (optional diagnostic)
    # NEW: relative-to-initial growth rule (more physically consistent across conditions)
    grad_runup_mult = getattr(sc, "grad_runup_mult", 2.0)  # e.g., 2x growth from initial gradient
    g0 = float(gmax_t[0])
    g0 = max(g0, 1e-8)  # safety floor
    grad_runup_thr = grad_runup_mult * g0

    runup_g = np.nan
    for i, tn in enumerate(t_norm):
        if float(gmax_t[i]) >= float(grad_runup_thr):
            runup_g = float(tn * t_end_actual)
            break
        
    return dict(
        peak_u=peak_u,
        runup_time_u_s=runup_u,
        runup_time_g_s=runup_g,
        g_peak=g_peak,
        front_speed=front_speed,
        front_speed_abs=fs_abs,
    )

def assign_regime_by_coeff(d, th_bucket):
    """
    d: diagnostics dict containing g_peak, front_speed_abs
    th_bucket: dict containing refs & ratio thresholds for this coeff bucket.
      required keys:
        - g_ref, v_ref
        - g_no, v_no, g_det, v_det (ratio thresholds)
    """
    g = float(d["g_peak"])

    g_no_ref  = float(th_bucket["g_no_ref"])
    g_det_ref = float(th_bucket["g_det_ref"])

    if g >= g_det_ref:
        regime = "detonation_like"
    elif g <= g_no_ref:
        regime = "no_detonation"
    else:
        regime = "deflagration_like"

    # g_ratio/v_ratio는 더 이상 의미 없으니 그냥 참고용으로만 반환
    return regime, dict(g_ratio=np.nan, v_ratio=np.nan)

# -----------------------------
# Guided sampling for dTdx (keep)
# -----------------------------
def sample_dTdx_guided(rng, dc, target_label: str):
    lo, hi = float(dc.dTdx_min), float(dc.dTdx_max)
    sign = 1.0
    if lo < 0 < hi:
        sign = -1.0 if rng.random() < 0.5 else 1.0
        mag_hi = max(abs(lo), abs(hi))
        if target_label == "no_detonation":
            a, b = 0.0, 0.05 * mag_hi
        elif target_label == "deflagration_like":
            a, b = 0.05 * mag_hi, 0.55 * mag_hi
        else:
            a, b = 0.55 * mag_hi, 1.00 * mag_hi
        mag = float(rng.uniform(a, b))
        return float(np.clip(sign * mag, lo, hi))

    span = hi - lo
    if span <= 0:
        return float(lo)
    if target_label == "no_detonation":
        a, b = lo, lo + 0.05 * span
    elif target_label == "deflagration_like":
        a, b = lo + 0.05 * span, lo + 0.55 * span
    else:
        a, b = lo + 0.55 * span, hi
    return float(rng.uniform(a, b))

# -----------------------------
# Calibration (condition-aware)
# -----------------------------
def calibrate_thresholds_by_coeff(sc: SimCfg, dc: DataCfg, rng: np.random.Generator,
                                  n_samples: int = 4000,
                                  nu_bins: int = 3, k_bins: int = 2, E_bins: int = 2,
                                  ratio_no: float = 0.55, ratio_det: float = 0.90):
    """
    Build per-(nu,k,E) bucket reference statistics:
      g_ref = Q90(g_peak) and v_ref = Q90(|front_speed|)
    and store shared ratio thresholds for det/no.
    """

    # Define coefficient ranges
    nu_nom_lo = float(sc.nu) * float(_get(dc, "nu_scale_min", 1.0))
    nu_nom_hi = float(sc.nu) * float(_get(dc, "nu_scale_max", 1.0))
    # Allow optional mismatch ranges; fallback to point-mass at nu_ood/k_ood/E_ood
    nu_ood_lo = float(_get(dc, "nu_ood_min", _get(dc, "nu_ood", sc.nu)))
    nu_ood_hi = float(_get(dc, "nu_ood_max", _get(dc, "nu_ood", sc.nu)))
    k_ood_lo  = float(_get(dc, "k_ood_min",  _get(dc, "k_ood",  sc.k)))
    k_ood_hi  = float(_get(dc, "k_ood_max",  _get(dc, "k_ood",  sc.k)))
    E_ood_lo  = float(_get(dc, "E_ood_min",  _get(dc, "E_ood",  sc.E)))
    E_ood_hi  = float(_get(dc, "E_ood_max",  _get(dc, "E_ood",  sc.E)))

    # We'll calibrate over the union of nominal and mismatch ranges for nu,
    # and over union for k,E (nominal are fixed sc.k/sc.E unless user provides ranges).
    k_nom_lo = float(_get(dc, "k_nom_min", sc.k))
    k_nom_hi = float(_get(dc, "k_nom_max", sc.k))
    E_nom_lo = float(_get(dc, "E_nom_min", sc.E))
    E_nom_hi = float(_get(dc, "E_nom_max", sc.E))

    nu_lo = min(nu_nom_lo, nu_ood_lo)
    nu_hi = max(nu_nom_hi, nu_ood_hi)
    k_lo = min(k_nom_lo, k_ood_lo)
    k_hi = max(k_nom_hi, k_ood_hi)
    E_lo = min(E_nom_lo, E_ood_lo)
    E_hi = max(E_nom_hi, E_ood_hi)

    # Bin edges: nu on log scale (positive), k/E on linear scale
    nu_edges = _log_edges(nu_lo, nu_hi, nu_bins)
    k_edges  = _lin_edges(k_lo, k_hi,  k_bins)
    E_edges  = _lin_edges(E_lo, E_hi,  E_bins)

    # Collect diagnostics per bucket
    buckets = {}  # key -> {"gs": [...], "vs":[...], "n":int}
    t0 = time.time()
    print(f"[calibration_by_coeff] n_samples={n_samples} bins: nu={nu_bins},k={k_bins},E={E_bins}", flush=True)

    te_smin = float(_get(dc, "t_end_scale_min", 1.0))
    te_smax = float(_get(dc, "t_end_scale_max", 1.0))

    for i in range(int(n_samples)):
        # Sample coefficients across union range (this is what makes it "generalized")
        nu = float(np.exp(rng.uniform(np.log(max(nu_lo,1e-12)), np.log(max(nu_hi, nu_lo*1.000001)))))
        k  = float(rng.uniform(k_lo, k_hi)) if k_hi > k_lo else float(k_lo)
        E  = float(rng.uniform(E_lo, E_hi)) if E_hi > E_lo else float(E_lo)

        dTdx = float(rng.uniform(dc.dTdx_min, dc.dTdx_max))
        t_end_case = float(sc.t_end) * float(rng.uniform(te_smin, te_smax))

        x, t_norm, U, _ = simulate_case(
            L_mm=sc.L, Nx=sc.Nx, t_end=t_end_case, Nt_save=sc.Nt_save, CFL=sc.CFL,
            nu=nu, k=k, E=E, dTdx=dTdx, b_quad=0.0,
            seed=990000 + i
        )

        d = compute_diagnostics(U, x, t_norm, sc, t_end_actual=t_end_case)
        key = _coeff_key(nu, k, E, nu_edges, k_edges, E_edges)
        b = buckets.setdefault(key, {"gs": [], "vs": [], "n": 0})
        b["gs"].append(float(d["g_peak"]))
        b["vs"].append(float(d["front_speed_abs"]))
        b["n"] += 1

        if (i+1) % 100 == 0:
            print(f"[calibration_by_coeff] {i+1}/{n_samples} elapsed={time.time()-t0:.1f}s buckets={len(buckets)}",
                  flush=True)

    # Compute references per bucket
    th = {
        "meta": {
            "nu_edges": nu_edges.tolist(),
            "k_edges": k_edges.tolist(),
            "E_edges": E_edges.tolist(),
            "nu_bins": int(nu_bins),
            "k_bins": int(k_bins),
            "E_bins": int(E_bins),
            "ratio_thresholds": {"g_no": float(ratio_no), "v_no": float(ratio_no),
                                 "g_det": float(ratio_det), "v_det": float(ratio_det)},
            "q_ref": 0.90,
            "n_samples": int(n_samples),
        },
        "buckets": {}
    }

    for key, b in buckets.items():
        gs = np.asarray(b["gs"], dtype=float)
        vs = np.asarray(b["vs"], dtype=float)

        # 기존 유지(원하면 v_ref는 진짜 안 써도 됨)
        v_ref = float(np.quantile(vs, 0.90)) if vs.size else 1e-12

        # NEW: g-based refs
        g_no_ref  = float(np.quantile(gs, 0.20)) if gs.size else 1e-12
        g_det_ref = float(np.quantile(gs, 0.90)) if gs.size else 1e-12

        th["buckets"][key] = {
            "g_no_ref":  g_no_ref,
            "g_det_ref": g_det_ref,
            "v_ref": v_ref,
            "n": int(b["n"]),
        }

    _ensure_dir("data")
    out_path = "data/thresholds_by_coeff.json"
    with open(out_path, "w") as f:
        json.dump(th, f, indent=2)
    print(f"[calibration_by_coeff] wrote {out_path} with {len(th['buckets'])} buckets.", flush=True)
    return th

def load_thresholds_by_coeff():
    path = "data/thresholds_by_coeff.json"
    if not os.path.exists(path):
        raise RuntimeError(
            "thresholds_by_coeff.json not found. "
            "Run with --calibrate_by_coeff first."
        )
    with open(path, "r") as f:
        th = json.load(f)
    return th

def _nearest_bucket(key: str, th_buckets: dict) -> str:
    # Simple fallback: if exact key missing, choose bucket with max n
    # (You can refine later by param-space distance.)
    if key in th_buckets:
        return key
    best = None
    best_n = -1
    for k, v in th_buckets.items():
        n = int(v.get("n", 0))
        if n > best_n:
            best_n = n
            best = k
    return best if best is not None else list(th_buckets.keys())[0]

# -----------------------------
# Pool generation (label-aware, condition-aware)
# -----------------------------
def gen_pool_for_split(sc, dc, rng, split, n_target, th_all,
                       ood_profile=False, ood_mismatch=False,
                       max_tries_factor=80.0, log_every=200):
    """
    Similar to original, but regime label uses thresholds_by_coeff based on (nu,k,E) bucket.
    """
    det_min_frac = {
        "train": _get(dc, "det_min_frac_train", 0.10),
        "val": _get(dc, "det_min_frac_val", 0.10),
        "test_profile_ood": _get(dc, "det_min_frac_test_profile_ood", 0.10),
        "test_mismatch_ood": _get(dc, "det_min_frac_test_mismatch_ood", 0.10),
    }.get(split, 0.10)

    req_det = int(math.ceil(det_min_frac * n_target))
    req_other_each = int(_get(dc, "min_other_each", 1))
    if split == "test_mismatch_ood":
        # keep as originally relaxed, but now labels should not collapse
        req_other_each = int(_get(dc, "min_other_each_mismatch", 0))

    pool = {lab: [] for lab in LABELS}
    attempts = 0
    max_attempts = int(max(1, math.ceil(n_target * max_tries_factor)))
    t_start = time.time()

    def counts():
        return {lab: len(pool[lab]) for lab in LABELS}

    # Load edges for bucketization
    nu_edges = np.asarray(th_all["meta"]["nu_edges"], dtype=float)
    k_edges  = np.asarray(th_all["meta"]["k_edges"], dtype=float)
    E_edges  = np.asarray(th_all["meta"]["E_edges"], dtype=float)
    th_buckets = th_all["buckets"]

    # coeff ranges for mismatch sampling (generalized)
    nu_ood_lo = float(_get(dc, "nu_ood_min", _get(dc, "nu_ood", sc.nu)))
    nu_ood_hi = float(_get(dc, "nu_ood_max", _get(dc, "nu_ood", sc.nu)))
    k_ood_lo  = float(_get(dc, "k_ood_min",  _get(dc, "k_ood",  sc.k)))
    k_ood_hi  = float(_get(dc, "k_ood_max",  _get(dc, "k_ood",  sc.k)))
    E_ood_lo  = float(_get(dc, "E_ood_min",  _get(dc, "E_ood",  sc.E)))
    E_ood_hi  = float(_get(dc, "E_ood_max",  _get(dc, "E_ood",  sc.E)))

    nu_smin = float(_get(dc, "nu_scale_min", 1.0))
    nu_smax = float(_get(dc, "nu_scale_max", 1.0))
    te_smin = float(_get(dc, "t_end_scale_min", 1.0))
    te_smax = float(_get(dc, "t_end_scale_max", 1.0))

    while attempts < max_attempts:
        c = counts()
        total = sum(c.values())
        if (c["detonation_like"] >= req_det and
            c["no_detonation"] >= req_other_each and
            c["deflagration_like"] >= req_other_each and
            total >= n_target):
            break

        attempts += 1

        # Decide what label we most need right now
        need = []
        if c["detonation_like"] < req_det:
            need.append("detonation_like")
        if c["no_detonation"] < req_other_each:
            need.append("no_detonation")
        if c["deflagration_like"] < req_other_each:
            need.append("deflagration_like")
        target_label = rng.choice(need) if need else rng.choice(LABELS)

        dTdx = sample_dTdx_guided(rng, dc, target_label)

        b = 0.0
        if ood_profile:
            b = float(rng.uniform(dc.b_ood_min, dc.b_ood_max))

        # Sample coefficients
        if ood_mismatch:
            # generalized mismatch: sample within provided ranges (or point-mass if min=max)
            nu = float(rng.uniform(nu_ood_lo, nu_ood_hi)) if nu_ood_hi > nu_ood_lo else float(nu_ood_lo)
            k  = float(rng.uniform(k_ood_lo,  k_ood_hi))  if k_ood_hi  > k_ood_lo  else float(k_ood_lo)
            E  = float(rng.uniform(E_ood_lo,  E_ood_hi))  if E_ood_hi  > E_ood_lo  else float(E_ood_lo)
        else:
            nu = float(sc.nu) * float(rng.uniform(nu_smin, nu_smax))
            k  = float(sc.k)
            E  = float(sc.E)

        t_end_case = float(sc.t_end) * float(rng.uniform(te_smin, te_smax))

        x, t_norm, U, ic = simulate_case(
            L_mm=sc.L, Nx=sc.Nx, t_end=t_end_case, Nt_save=sc.Nt_save, CFL=sc.CFL,
            nu=nu, k=k, E=E, dTdx=dTdx, b_quad=b,
            seed=1000 + attempts + (12345 if split.startswith("test") else 0)
        )

        diag = compute_diagnostics(U, x, t_norm, sc, t_end_actual=t_end_case)

        key = _coeff_key(nu, k, E, nu_edges, k_edges, E_edges)
        key_use = _nearest_bucket(key, th_buckets)
        regime, ratios = assign_regime_by_coeff(diag, th_buckets[key_use])

        meta = {
            "dTdx": dTdx,
            "b_quad": b,
            "nu": float(nu),
            "k": float(k),
            "E": float(E),
            "L_mm": float(sc.L),
            "Nx": int(sc.Nx),
            "Nt": int(sc.Nt_save),
            "peak_u": float(diag["peak_u"]),
            "runup_time_s": float(diag["runup_time_u_s"]) if not np.isnan(diag["runup_time_u_s"]) else np.nan,
            "runup_time_g_s": float(diag["runup_time_g_s"]) if not np.isnan(diag["runup_time_g_s"]) else np.nan,
            "g_peak": float(diag["g_peak"]),
            "front_speed": float(diag["front_speed"]) if not np.isnan(diag["front_speed"]) else np.nan,
            "front_speed_abs": float(diag["front_speed_abs"]),
            "coeff_bucket": key_use,
            "regime": regime,
            "regime_id": LABEL2ID[regime],
            "t_end": float(t_end_case),
            "dt": float(t_end_case) / max(int(sc.Nt_save) - 1, 1),
        }
        meta.update(ic)
        pool[regime].append((U, meta))

        if (attempts % log_every) == 0:
            c = counts()
            elapsed = time.time() - t_start
            print(f"[{split}] attempts={attempts}/{max_attempts} counts={c} elapsed={elapsed:.1f}s", flush=True)

    final_counts = counts()
    total = sum(final_counts.values())
    ok = (final_counts["detonation_like"] >= req_det and
          final_counts["no_detonation"] >= req_other_each and
          final_counts["deflagration_like"] >= req_other_each and
          total >= n_target)

    return pool, req_det, req_other_each, attempts, max_attempts, ok, final_counts

def sample_from_pool(rng, pool, n_target, req_det, req_other_each):
    chosen = []

    def take(label, k):
        items = pool[label]
        if k <= 0:
            return
        if len(items) < k:
            raise RuntimeError(f"Not enough '{label}' samples in pool: have {len(items)}, need {k}")
        idx = rng.choice(len(items), size=k, replace=False)
        idx = sorted(idx, reverse=True)
        for i in idx:
            chosen.append(items.pop(i))

    take("detonation_like", req_det)
    take("no_detonation", req_other_each)
    take("deflagration_like", req_other_each)

    remaining = n_target - len(chosen)
    rest = []
    for lab in LABELS:
        rest.extend(pool[lab])

    if remaining > len(rest):
        raise RuntimeError(f"Pool too small after mandatory picks: need {remaining}, have {len(rest)}")

    idx = rng.choice(len(rest), size=remaining, replace=False)
    for i in idx:
        chosen.append(rest[i])

    rng.shuffle(chosen)
    return chosen

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate_by_coeff", action="store_true",
                        help="Build thresholds_by_coeff.json using coefficient-bucket calibration.")
    parser.add_argument("--n_calib", type=int, default=4000,
                        help="Number of calibration samples (default 4000).")
    parser.add_argument("--nu_bins", type=int, default=3)
    parser.add_argument("--k_bins", type=int, default=2)
    parser.add_argument("--E_bins", type=int, default=2)
    args = parser.parse_args()

    sc = SimCfg()
    dc = DataCfg()
    rng = np.random.default_rng(dc.seed)

    if args.calibrate_by_coeff:
        calibrate_thresholds_by_coeff(
            sc, dc, rng,
            n_samples=int(args.n_calib),
            nu_bins=int(args.nu_bins),
            k_bins=int(args.k_bins),
            E_bins=int(args.E_bins),
        )
        return

    th_all = load_thresholds_by_coeff()
    print("[thresholds_by_coeff loaded] buckets:", len(th_all["buckets"]), flush=True)

    _ensure_dir("data")

    # canonical grids (for convenience; note solver returns x normalized 0..1)
    x_grid = np.linspace(0.0, 1.0, int(sc.Nx), dtype=np.float32)
    t_norm_grid = np.linspace(0.0, 1.0, int(sc.Nt_save), dtype=np.float32)
    np.savez_compressed("data/grid.npz", x=x_grid, t=t_norm_grid)

    splits = [
        ("train", int(dc.n_train), False, False),
        ("val", int(dc.n_val), False, False),
        ("test_profile_ood", int(dc.n_test_profile_ood), True, False),
        ("test_mismatch_ood", int(dc.n_test_mismatch_ood), False, True),
    ]

    rows = []
    blobs = {}

    max_tries_factor = float(_get(dc, "det_max_tries_factor", 80.0))
    log_every = int(_get(dc, "det_log_every", 200))

    for split, n_target, ood_profile, ood_mismatch in splits:
        pool, req_det, req_other_each, attempts, max_attempts, ok, counts = gen_pool_for_split(
            sc, dc, rng, split, n_target, th_all,
            ood_profile=ood_profile, ood_mismatch=ood_mismatch,
            max_tries_factor=max_tries_factor,
            log_every=log_every
        )

        if not ok:
            print(f"[WARN] Could not fully satisfy constraints for split={split}. counts={counts}, "
                  f"req_det={req_det}, req_other_each={req_other_each}. Proceeding best-effort.",
                  flush=True)

        req_det_eff = req_det if counts["detonation_like"] >= req_det else max(0, counts["detonation_like"])
        chosen = sample_from_pool(rng, pool, n_target, req_det_eff, req_other_each)

        U_list = []
        for i, (U, meta) in enumerate(chosen):
            case_id = len(U_list)
            meta_row = {"case_id": case_id, "split": split}
            meta_row.update(meta)
            rows.append(meta_row)
            U_list.append(U.astype(np.float32))

        blobs[split] = np.stack(U_list, axis=0)  # (Ncases, Nt, Nx)

    # write files
    meta = pd.DataFrame(rows)
    meta.to_csv("data/meta.csv", index=False)

    # save per-split npz
    np.savez_compressed("data/u_train.npz", u=blobs["train"])
    np.savez_compressed("data/u_val.npz", u=blobs["val"])
    np.savez_compressed("data/u_test_profile_ood.npz", u=blobs["test_profile_ood"])
    np.savez_compressed("data/u_test_mismatch_ood.npz", u=blobs["test_mismatch_ood"])

    print("Saved dataset to data/")
    print("Splits:", [s[0] for s in splits])
    print("Labels:", LABELS)
    for split, *_ in splits:
        sub = meta[meta["split"] == split]
        vc = sub["regime"].value_counts().to_dict()
        print(f" {split} label counts:", vc)

if __name__ == "__main__":
    main()
