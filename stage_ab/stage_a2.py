"""Stage A2: difference-aware certificates and face-selective certified routing.

This module strengthens the Stage A certificate for periodic Burgers/Godunov
with a two-input ReLU numerical flux.  All claims are relative to the SAME-GRID,
EXACT-ARITHMETIC Godunov map S (never to the PDE solution); see
``docs/stage_a2_theory.md`` for the statements and proofs referenced below.

Pipeline for one time step (all certificate inputs are functions of the state
only, so routing is decided BEFORE any flux is evaluated):

1. per-face bounds  eps_N(f) >= |F_fp(x_f) - G(x_f)|        (table, Lemma A5)
                    eps_G     >= |G_fp(x_f) - G(x_f)|        (Lemma F2)
2. per-cell difference bound Delta_i >= |e_i - e_(i-1)| when both faces of
   cell i are neural (Theorem D1; table or local-box variant)
3. per-cell costs kappa_i(s_(i-1), s_i) for the four trust patterns
4. exact trust frontier C(k) = min certified cost with k neural faces (DP on
   the cycle, Theorem R1)
5. an online budget policy picks k; NN is evaluated only on trusted faces and
   Godunov only on the others; floating update; clip to [-M, M]
6. certified step bound  eta = lam*h*sum_i kappa_i + rho_upd  (Theorem M1)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction as Q
import json
from pathlib import Path
import time

import numpy as np

from certified_burgers.godunov import godunov_flux
from .burgers import FrozenFlux, exact_godunov
from .interval import Interval as I
from .vinterval import (VI, U, up, up_float, rigorous_nonneg_sum,
                        network_value_enclosure, network_value_and_gradient,
                        godunov_value_enclosure, godunov_gradient_cases,
                        hidden_slopes_degenerate)

ARITHMETIC = "numpy-float64-separate-multiply-add-v1"


# ---------------------------------------------------------------------------
# Immutable model wrapper
# ---------------------------------------------------------------------------
class CertifiedFlux:
    """Read-only copy of a FrozenFlux with a digest computed once (fixes A-2)."""

    def __init__(self, frozen: FrozenFlux):
        layers = []
        for w, b in frozen.layers:
            w, b = np.array(w, dtype=np.float64), np.array(b, dtype=np.float64)
            w.setflags(write=False)
            b.setflags(write=False)
            layers.append((w, b))
        self._frozen = FrozenFlux(layers)
        for w, b in self._frozen.layers:
            w.setflags(write=False)
            b.setflags(write=False)
        self.layers = self._frozen.layers
        self.digest = self._frozen.digest
        self.width = max(len(b) for _, b in self.layers)

    def predict(self, pairs):
        return self._frozen.predict(pairs)

    def real_predict(self, pairs):
        return self._frozen.real_predict(pairs)

    def fp_roundoff_bound(self, box):
        return self._frozen.fp_roundoff_bound(box)


# ---------------------------------------------------------------------------
# Floating-point constants (Lemmas F1, F2)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FPConstants:
    lam: float
    envelope: float
    eps_godunov: float          # >= |G_fp(a,b) - G(a,b)| for |a|,|b| <= M   (Lemma F2)
    upd_a: float                # >= u/(1-u)                                  (Lemma F1)
    upd_b: float                # >= lam(2u+u^2)/(1-u)
    upd_c: float                # absolute slack per cell (covers eta and underflow)

    @classmethod
    def build(cls, lam: float, envelope: float):
        lam_q, m = Q(float(lam)), Q(float(envelope))
        tiny = Q(1, 2**1075)
        eps_g = U*m*m/2 + (3*m+2)*tiny
        return cls(float(lam), float(envelope), up_float(eps_g), up_float(U/(1-U)),
                   up_float(lam_q*(2*U+U*U)/(1-U)), 2.0**-1072)

    def update_defect(self, p, dF, n_cells: int) -> Q:
        """Rigorous upper bound of h*sum_i |p_i - Ptilde_i| (Lemma F1 + F3)."""
        t = self.upd_a*np.abs(p) + self.upd_b*np.abs(dF) + self.upd_c
        # Each term was formed with <= 3 roundings of nonnegative values; the
        # summation adds <= n-1 more (Lemma F3); h = 1/n exactly.
        s = rigorous_nonneg_sum(t)
        return s/((1-U)**3)/n_cells

    def update_defect_apriori(self, flux_abs_max: float, n_cells: int) -> Q:
        """A priori upper bound of the value ``update_defect`` can return for any
        state in K whose face fluxes satisfy |flux| <= flux_abs_max (Lemma F1')."""
        lam, m, fmax = Q(self.lam), Q(self.envelope), Q(float(flux_abs_max))
        eta = Q(1, 2**1075)
        dF = 2*fmax*(1+U)
        p = (m + lam*dF*(1+U) + eta)*(1+U)
        base = Q(self.upd_a)*p + Q(self.upd_b)*dF + Q(self.upd_c)
        n = int(n_cells)
        # t_hat <= base (1+u)^3, fl-sum <= (1+u)^(n-1) sum, and (1+u)^k <= 1/(1-ku).
        return base/((1-(n+2)*U)*(1-(n-1)*U)*(1-U)**3)


# ---------------------------------------------------------------------------
# Offline certificate tables
# ---------------------------------------------------------------------------
def _box_edges(edges, lo_idx, hi_idx):
    return edges[np.minimum(lo_idx, hi_idx)], edges[np.maximum(lo_idx, hi_idx)+1]


@dataclass
class CertificateTables:
    """Face bounds delta (P x P), FP bounds eps_fp (P x P), and triple-box
    gradient enclosures of D = F_real - G for the difference certificate."""
    model_hash: str
    edges: np.ndarray
    subdivisions: int
    delta: np.ndarray
    eps_fp: np.ndarray
    grad_fa: np.ndarray = field(repr=False)     # (P,P,P,2) lo/hi of dF/da on Omega_pqr
    grad_fb: np.ndarray = field(repr=False)
    has_p: np.ndarray = field(repr=False)       # (P,P,P) bool
    grad_pa: np.ndarray = field(repr=False)     # (P,P,P,2) enclosure of a+ on branch p
    has_q: np.ndarray = field(repr=False)
    grad_qb: np.ndarray = field(repr=False)     # (P,P,P,2) enclosure of b- on branch q
    flux_abs_max: float = 0.0
    build_seconds: float = 0.0

    @property
    def bins(self):
        return len(self.edges)-1

    @property
    def envelope(self):
        return float(self.edges[-1])

    @classmethod
    def build(cls, model: CertifiedFlux, *, envelope=2., bins=16, subdivisions=1, triple=True):
        if not (np.isfinite(envelope) and envelope > 0 and bins >= 1 and subdivisions >= 1):
            raise ValueError("Invalid certificate partition")
        start = time.perf_counter()
        edges = np.linspace(-envelope, envelope, bins+1)
        if edges[0] != -envelope or edges[-1] != envelope:
            raise ArithmeticError("linspace did not reproduce the envelope endpoints")
        P = bins
        # --- delta table: enclosure of the FP network minus the exact Godunov range.
        delta = np.zeros((P, P))
        fmax = 0.0
        sub = sub_edges(edges, subdivisions)            # (P, S+1), exact shared endpoints
        for j in range(subdivisions):
            for k in range(subdivisions):
                A = VI(np.repeat(sub[:, j], P), np.repeat(sub[:, j+1], P))
                B = VI(np.tile(sub[:, k], P), np.tile(sub[:, k+1], P))
                val = network_value_enclosure(model.layers, A, B)
                g = godunov_value_enclosure(A, B)
                bound = np.maximum(up(val.hi - g.lo), up(g.hi - val.lo))
                delta = np.maximum(delta, bound.reshape(P, P))
                fmax = max(fmax, float(np.max(val.abs_upper())))
        # --- per-bin FP forward-error bound (exact rational, reviewed routine).
        eps_fp = np.zeros((P, P))
        for p in range(P):
            for q in range(P):
                box = [I(edges[p], edges[p+1]), I(edges[q], edges[q+1])]
                eps_fp[p, q] = up_float(model.fp_roundoff_bound(box))
        obj = cls(model.digest, edges, int(subdivisions), delta, eps_fp,
                  *([None]*6), flux_abs_max=up_float(Q(fmax)))
        if triple:
            obj._build_triples(model)
        obj.build_seconds = time.perf_counter()-start
        return obj

    def _build_triples(self, model):
        P, edges = self.bins, self.edges
        p, q, r = np.meshgrid(np.arange(P), np.arange(P), np.arange(P), indexing="ij")
        p, q, r = p.ravel(), q.ravel(), r.ravel()
        a_lo, a_hi = _box_edges(edges, p, q)
        b_lo, b_hi = _box_edges(edges, q, r)
        A, B = VI(a_lo, a_hi), VI(b_lo, b_hi)
        _, gfa, gfb = network_value_and_gradient(model.layers, A, B)
        has_p, gpa, has_q, gqb = godunov_gradient_cases(A, B)
        shape = (P, P, P)
        pack = lambda v: np.stack([v.lo, v.hi], axis=-1).reshape(shape+(2,))
        self.grad_fa, self.grad_fb = pack(gfa), pack(gfb)
        self.has_p, self.has_q = has_p.reshape(shape), has_q.reshape(shape)
        self.grad_pa, self.grad_qb = pack(gpa), pack(gqb)

    # ------------------------------------------------------------------ lookup
    def bin_index(self, v):
        v = np.asarray(v, dtype=np.float64)
        if not np.all(np.isfinite(v)) or np.any(v < self.edges[0]) or np.any(v > self.edges[-1]):
            raise ValueError("State outside the certified envelope")
        idx = np.searchsorted(self.edges, v, side="right")-1
        return np.clip(idx, 0, self.bins-1)

    # ------------------------------------------------------------ persistence
    def save(self, path):
        data = {"model_hash": self.model_hash, "edges": self.edges.tolist(),
                "subdivisions": self.subdivisions, "delta": self.delta.tolist(),
                "eps_fp": self.eps_fp.tolist(), "flux_abs_max": self.flux_abs_max,
                "build_seconds": self.build_seconds, "arithmetic": ARITHMETIC,
                "triple": self.grad_fa is not None}
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path, model: CertifiedFlux, *, verify=True):
        """Tables are rebuilt and compared on load; a hash alone authenticates nothing."""
        d = json.loads(Path(path).read_text())
        if d["model_hash"] != model.digest or d.get("arithmetic") != ARITHMETIC:
            raise ValueError("Certificate contract mismatch")
        edges = np.array(d["edges"])
        rebuilt = cls.build(model, envelope=float(edges[-1]), bins=len(edges)-1,
                            subdivisions=int(d["subdivisions"]), triple=bool(d["triple"]))
        stored_delta, stored_eps = np.array(d["delta"]), np.array(d["eps_fp"])
        if verify and (not np.array_equal(edges, rebuilt.edges)
                       or np.any(stored_delta < rebuilt.delta) or np.any(stored_eps < rebuilt.eps_fp)
                       or float(d["flux_abs_max"]) < rebuilt.flux_abs_max):
            raise ValueError("Unverified or tampered certificate table")
        return rebuilt


def sub_edges(edges, subdivisions):
    """Sub-bin edges S[p, 0..n] with S[p,0]=e_p and S[p,n]=e_(p+1) exactly.

    Consecutive closed sub-intervals share the same float endpoint, so their
    union is exactly the closed bin (no gaps), whatever the interior rounding.
    """
    n = int(subdivisions)
    lo, hi = edges[:-1], edges[1:]
    t = np.arange(n+1)/n
    S = lo[:, None] + t[None, :]*(hi-lo)[:, None]
    S[:, 0], S[:, -1] = lo, hi
    S = np.maximum.accumulate(S, axis=1)
    S = np.minimum(S, hi[:, None])
    if np.any(np.diff(S, axis=1) < 0) or not (np.array_equal(S[:, 0], lo) and np.array_equal(S[:, -1], hi)):
        raise ArithmeticError("Invalid sub-bin partition")
    return S


# ---------------------------------------------------------------------------
# Per-cell certified costs
# ---------------------------------------------------------------------------
def _interval_diff(x, y):
    """Outward enclosure of y - x for float arrays (exact reals)."""
    return VI.point(y) - VI.point(x)


def _hull_branches(has_p, jp: VI, has_q, jq: VI) -> VI:
    both = has_p & has_q
    lo = np.where(both, np.minimum(jp.lo, jq.lo), np.where(has_p, jp.lo, jq.lo))
    hi = np.where(both, np.maximum(jp.hi, jq.hi), np.where(has_p, jp.hi, jq.hi))
    return VI(lo, hi)


def difference_bounds(model: CertifiedFlux, tables: CertificateTables, v, *, mode="both"):
    """Per-cell bound Delta_i >= |e_i - e_(i-1)| for all-neural faces i-1, i.

    mode: 'table' (triple-box enclosure, O(1) lookups), 'local' (online
    interval AD on hull(x_(i-1), x_i)), or 'both' (elementwise minimum).
    """
    v = np.asarray(v, dtype=np.float64)
    vm, vp = np.roll(v, 1), np.roll(v, -1)
    da, db = _interval_diff(vm, v), _interval_diff(v, vp)
    idx = tables.bin_index(v)
    im, ip = np.roll(idx, 1), np.roll(idx, -1)
    eps_face = tables.eps_fp[idx, ip]                  # face i   = (v_i, v_(i+1))
    eps_left = np.roll(eps_face, 1)                    # face i-1 = (v_(i-1), v_i)
    out = []
    if mode in ("table", "both"):
        if tables.grad_fa is None:
            raise ValueError("Tables were built without triple gradients")
        sel = (im, idx, ip)
        g = lambda arr: VI(arr[sel][..., 0], arr[sel][..., 1])
        gfa, gfb, gpa, gqb = g(tables.grad_fa), g(tables.grad_fb), g(tables.grad_pa), g(tables.grad_qb)
        jp = (gfa - gpa)*da + gfb*db
        jq = gfa*da + (gfb - gqb)*db
        J = _hull_branches(tables.has_p[sel], jp, tables.has_q[sel], jq)
        out.append(J.abs_upper())
    if mode in ("local", "both"):
        A = VI(np.minimum(vm, v), np.maximum(vm, v))
        B = VI(np.minimum(v, vp), np.maximum(v, vp))
        _, gfa, gfb = network_value_and_gradient(model.layers, A, B)
        has_p, gpa, has_q, gqb = godunov_gradient_cases(A, B)
        jp = (gfa - gpa)*da + gfb*db
        jq = gfa*da + (gfb - gqb)*db
        out.append(_hull_branches(has_p, jp, has_q, jq).abs_upper())
    if not out:
        raise ValueError("mode must be 'table', 'local' or 'both'")
    j = out[0] if len(out) == 1 else np.minimum(out[0], out[1])
    return up(up(j + eps_left) + eps_face)


def face_bounds(model: CertifiedFlux, tables: CertificateTables, v, *, mode="both"):
    """Per-face bound eps_N(f) >= |F_fp(x_f) - G(x_f)|, x_f = (v_f, v_(f+1)).

    'table' / 'separable': the offline bin table (Lemma A5).
    'local' / 'both': additionally the outward enclosure of the FP network and of
    G at the single point x_f (Lemma A4 applied to a degenerate box), which is
    within a few ulps of the true face error; the minimum of sound bounds is sound.
    """
    v = np.asarray(v, dtype=np.float64)
    idx = tables.bin_index(v)
    delta = tables.delta[idx, np.roll(idx, -1)]
    if mode in ("local", "both"):
        a, b = VI.point(v), VI.point(np.roll(v, -1))
        val = network_value_enclosure(model.layers, a, b)
        g = godunov_value_enclosure(a, b)
        delta = np.minimum(delta, np.maximum(up(val.hi - g.lo), up(g.hi - val.lo)))
    return delta


def local_regularity(model: CertifiedFlux, v):
    """Regular cells of Theorem R3: on Omega_i = hull(x_(i-1), x_i) the network has a
    certified single activation pattern and exactly one Godunov branch is possible.
    Returns (regular mask, branch: 0 for p (a+), 1 for q (b-), -1 if irregular)."""
    v = np.asarray(v, dtype=np.float64)
    vm, vp = np.roll(v, 1), np.roll(v, -1)
    A = VI(np.minimum(vm, v), np.maximum(vm, v))
    B = VI(np.minimum(v, vp), np.maximum(v, vp))
    slopes = hidden_slopes_degenerate(model.layers, A, B)
    has_p, _, has_q, _ = godunov_gradient_cases(A, B)
    single = has_p ^ has_q
    regular = slopes & single
    branch = np.where(regular, np.where(has_p, 0, 1), -1)
    return regular, branch


def cell_costs(model: CertifiedFlux, tables: CertificateTables, consts: FPConstants, v,
               *, mode="both"):
    """kappa[i, s_left, s_right] >= |etilde_i - etilde_(i-1)| for each trust pattern.

    Face f is (v_f, v_(f+1)); cell i has left face i-1 and right face i.
    Returns an (N, 2, 2) float array; every entry is an upper bound of the
    real quantity (all additions are rounded upward).
    mode: 'separable' (original Stage A certificate: table face bounds only),
    'table', 'local' or 'both' (see face_bounds and difference_bounds).
    """
    if mode not in ("separable", "table", "local", "both"):
        raise ValueError("Unknown certificate mode")
    v = np.asarray(v, dtype=np.float64)
    delta_face = face_bounds(model, tables, v, mode=mode)
    dl, dr = np.roll(delta_face, 1), delta_face
    g = consts.eps_godunov
    n = len(v)
    k = np.empty((n, 2, 2))
    k[:, 0, 0] = up(np.full(n, g) + g)
    k[:, 1, 0] = up(dl + g)
    k[:, 0, 1] = up(dr + g)
    sep = up(dl + dr)
    if mode == "separable":
        k[:, 1, 1] = sep
    else:
        k[:, 1, 1] = np.minimum(sep, difference_bounds(model, tables, v, mode=mode))
    return k


# ---------------------------------------------------------------------------
# Exact trust frontier on the cycle (Theorem R1)
# ---------------------------------------------------------------------------
@dataclass
class Frontier:
    cost: np.ndarray            # (N+1,) float DP sums: min sum_i kappa_i with k trusted faces
    sigma: np.ndarray           # (N+1,) status of face N-1 in the optimal set
    back: dict = field(repr=False)

    def select(self, k: int) -> np.ndarray:
        n = self.back[0].shape[0]
        sigma = int(self.sigma[k])
        back = self.back[sigma]
        s = np.zeros(n, dtype=bool)
        t, K = sigma, int(k)
        for f in range(n-1, 0, -1):
            s[f] = bool(t)
            prev = int(back[f, t, K])
            K -= t
            t = prev
        s[0] = bool(t)
        K -= t
        if K != 0 or s[n-1] != bool(sigma):
            raise AssertionError("Inconsistent DP traceback")
        return s


def trust_frontier(costs) -> Frontier:
    """C(k) = min over s in {0,1}^N with |s| = k of sum_i kappa_i(s_(i-1), s_i).

    Exact dynamic program over the cycle: fix the status sigma of face N-1
    (the left face of cell 0), sweep faces 0..N-1 keeping (status, count),
    and close the cycle by requiring face N-1 to have status sigma.  O(N^2).
    """
    costs = np.asarray(costs, dtype=np.float64)
    n = costs.shape[0]
    if costs.shape != (n, 2, 2) or n < 2:
        raise ValueError("costs must have shape (N,2,2) with N >= 2")
    inf = np.inf
    best = np.full(n+1, inf)
    sigma_of = np.zeros(n+1, dtype=np.int8)
    backs = {}
    for sigma in (0, 1):
        B = np.full((2, n+1), inf)
        B[0, 0] = costs[0, sigma, 0]
        B[1, 1] = costs[0, sigma, 1]
        back = np.zeros((n, 2, n+1), dtype=np.int8)
        for f in range(1, n):
            nb = np.full((2, n+1), inf)
            # t = 0: count unchanged
            c0, c1 = B[0] + costs[f, 0, 0], B[1] + costs[f, 1, 0]
            take = c1 < c0
            nb[0] = np.where(take, c1, c0)
            back[f, 0] = take
            # t = 1: count increases by one
            c0 = np.full(n+1, inf); c1 = np.full(n+1, inf)
            c0[1:] = B[0, :-1] + costs[f, 0, 1]
            c1[1:] = B[1, :-1] + costs[f, 1, 1]
            take = c1 < c0
            nb[1] = np.where(take, c1, c0)
            back[f, 1] = take
            B = nb
        C = B[sigma]
        better = C < best
        best = np.where(better, C, best)
        sigma_of = np.where(better, sigma, sigma_of).astype(np.int8)
        backs[sigma] = back
    if not np.all(np.isfinite(best)):
        raise ArithmeticError("Nonfinite frontier")
    return Frontier(best, sigma_of, backs)


def pattern_cost(costs, s) -> np.ndarray:
    s = np.asarray(s, dtype=int)
    left = np.roll(s, 1)
    return np.asarray(costs)[np.arange(len(s)), left, s]


def rigorous_step_cost(costs, s, lam, n_cells) -> Q:
    """Rigorous lam*h*sum_i kappa_i(s) as an exact rational upper bound."""
    return Q(float(lam))*rigorous_nonneg_sum(pattern_cost(costs, s))/n_cells


def frontier_upper(frontier: Frontier, lam, n_cells) -> np.ndarray:
    """Float upper bounds of lam*h*(real sum along the DP path), for every k.

    The DP accumulates N nonnegative floats sequentially, so the real sum is at
    most cost/(1-(N-1)u) (Lemma F3); one more upward-rounded product scales it.
    """
    c = up_float(Q(float(lam))/n_cells/(1-(n_cells-1)*U))
    return up(frontier.cost*c)


# ---------------------------------------------------------------------------
# Mixed NN/Godunov step (Theorem M1)
# ---------------------------------------------------------------------------
def pairs(state):
    return np.column_stack((state, np.roll(state, -1)))


def mixed_step(model: CertifiedFlux, consts: FPConstants, v, trust):
    """Neural flux on trusted faces only, FP64 Godunov elsewhere, clipped update.

    Returns (new_state, rho_upd (Fraction), unclipped p, face fluxes).
    """
    v = np.asarray(v, dtype=np.float64)
    trust = np.asarray(trust, dtype=bool)
    x = pairs(v)
    flux = np.empty(len(v))
    if trust.any():
        flux[trust] = model.predict(x[trust])
    if (~trust).any():
        flux[~trust] = godunov_flux(x[~trust, 0], x[~trust, 1])
    if not np.all(np.isfinite(flux)):
        raise ArithmeticError("Nonfinite flux")
    dF = flux - np.roll(flux, 1)
    m = consts.lam*dF
    p = v - m
    if not np.all(np.isfinite(p)):
        raise ArithmeticError("Nonfinite update")
    rho = consts.update_defect(p, dF, len(v))
    M = consts.envelope
    return np.clip(p, -M, M), rho, p, flux


def exact_step_error(v, new, lam) -> Q:
    """Research oracle: h * sum_i |new_i - S(v)_i| in exact rationals."""
    vq = [Q(float(x)) for x in v]
    n = len(vq)
    f = [exact_godunov(vq[i], vq[(i+1) % n]) for i in range(n)]
    lq = Q(float(lam))
    return sum((abs(Q(float(new[i])) - (vq[i] - lq*(f[i]-f[i-1]))) for i in range(n)), Q(0))/n


def exact_actual_face_error(model, v, lam) -> Q:
    """Oracle: lam*h*sum|e_i - e_(i-1)| for the all-neural step in exact rationals
    (FP network output, exact Godunov), i.e. the error the certificate bounds
    before update roundoff."""
    vq = [Q(float(x)) for x in v]
    n = len(vq)
    pred = model.predict(pairs(np.asarray(v, dtype=np.float64)))
    e = [Q(float(pred[i])) - exact_godunov(vq[i], vq[(i+1) % n]) for i in range(n)]
    return Q(float(lam))*sum((abs(e[i]-e[i-1]) for i in range(n)), Q(0))/n


# ---------------------------------------------------------------------------
# Certified rollout with an online budget policy (Theorems M1, M2, K1-K3)
# ---------------------------------------------------------------------------
def godunov_flux_abs_max(envelope) -> float:
    m = Q(float(envelope))
    return up_float(m*m/2*(1+2*U) + Q(1, 2**1070))


def per_step_reserve(tables: CertificateTables, consts: FPConstants, n_cells: int):
    """(c0max, rho_bar): a priori bounds on the all-Godunov frontier value and on
    the update-defect certificate, valid for every state in K."""
    n = int(n_cells)
    two_g = float(up(np.array(consts.eps_godunov) + consts.eps_godunov))
    c = up_float(Q(consts.lam)/n/(1-(n-1)*U))
    # frontier(0) <= n*two_g/(1-(n-1)u) (Lemma 3.6); up(fl(y)) <= y(1+u)(1+2u) for normal y.
    c0max = Q(n)*Q(two_g)/(1-(n-1)*U)*Q(c)*(1+4*U)**2
    fmax = max(tables.flux_abs_max, godunov_flux_abs_max(consts.envelope))
    return up_float(c0max), consts.update_defect_apriori(fmax, n)


def budget_parameters(tables: CertificateTables, lam, n_cells, steps, total_budget, cmin_fraction=0.05):
    """Excess budget B, charge floor c_min and the density range [L, U] used by
    ThresholdPolicy (Section 5 of the theory note).  Every charged option has
    density <= U = 1/c_min, and <= 1/(2 lam h delta_max + c_min) =: L only
    fails if the table under-estimates, which Lemma A5 excludes; L is in any
    case a free parameter of Theorem K1."""
    consts = FPConstants.build(lam, tables.envelope)
    c0max, rho_bar = per_step_reserve(tables, consts, n_cells)
    excess = Q(float(total_budget)) - steps*(Q(c0max) + rho_bar)
    if excess < 0:
        raise ValueError("Total budget below the unavoidable roundoff reserve")
    excess_f = float(excess)
    if Q(excess_f) > excess:
        excess_f = float(np.nextafter(excess_f, -np.inf))
    c_min = cmin_fraction*excess_f/(n_cells*steps)
    per_face = 2*float(lam)/n_cells*float(np.max(tables.delta))*(1+1e-9)
    L = 1.0/(per_face + c_min)
    U = 1.0/c_min if c_min > 0 else np.inf
    return {"excess": excess_f, "c_min": c_min, "L": L, "U": U, "theta": U/L,
            "alpha": 1+np.log(U/L), "reserve_per_step": float(Q(c0max)+rho_bar)}


def certified_rollout_a2(model: CertifiedFlux, tables: CertificateTables, v0, *, steps, lam,
                         total_budget, policy, mode="both", cmin_fraction=0.05,
                         audit=False, record=True):
    """Face-selective certified rollout.

    Guarantee (Theorem M2): for every n, ||v_n - u_n||_(1,h) <= spent_n <= total_budget,
    where u_n is the exact-arithmetic same-grid Godunov trajectory from v0.
    """
    M = tables.envelope
    if model.digest != tables.model_hash:
        raise ValueError("Model/certificate fingerprint mismatch")
    if Q(float(lam))*Q(M) > 1 or lam <= 0:
        raise ValueError("Certificate requires 0 < lam*M <= 1")
    v = np.asarray(v0, dtype=np.float64).copy()
    n = len(v)
    tables.bin_index(v)
    consts = FPConstants.build(lam, M)
    c0max, rho_bar = per_step_reserve(tables, consts, n)
    reserve = Q(c0max) + rho_bar
    total = Q(float(total_budget))
    excess = total - steps*reserve
    if excess < 0:
        raise ValueError("Total budget below the unavoidable roundoff reserve")
    excess_f = float(excess)
    if Q(excess_f) > excess:
        excess_f = np.nextafter(excess_f, -np.inf)
    c_min = cmin_fraction*excess_f/(n*steps) if excess_f > 0 else 0.0
    policy.reset(excess_f, steps)
    excess_q, charged = Q(excess_f), Q(0)
    values = np.arange(n+1, dtype=np.float64)
    spent = Q(0)
    ref = v.copy() if audit else None
    ref_spent = Q(0)
    rows, menus, raw_menus = [], [], []
    timing = {"certificate": 0.0, "frontier": 0.0, "policy": 0.0, "flux_update": 0.0}
    for step in range(steps):
        t0 = time.perf_counter()
        costs = cell_costs(model, tables, consts, v, mode=mode)
        t1 = time.perf_counter()
        fr = trust_frontier(costs)
        cbar = frontier_upper(fr, lam, n)
        t2 = time.perf_counter()
        # Charged weights are rounded UP so that c0max + w_j >= cbar[j] >= certified cost.
        if cbar[0] > c0max:
            raise AssertionError("All-Godunov frontier exceeds its a priori reserve")
        diff = np.where(cbar > c0max, up(cbar - c0max), 0.0)
        weights = np.where(values > 0, up(diff + up(values*c_min)), 0.0) if c_min > 0 else diff
        weights[0] = 0.0
        j = policy.propose(values, weights)
        # The budget is enforced in exact rationals, independently of the
        # policy's own floating bookkeeping (Theorem 4.2).
        overridden = charged + Q(float(weights[j])) > excess_q
        if overridden:
            j = 0
        trust = fr.select(j)
        t3 = time.perf_counter()
        new, rho, p, flux = mixed_step(model, consts, v, trust)
        t4 = time.perf_counter()
        cost = min(rigorous_step_cost(costs, trust, lam, n), Q(float(cbar[j])))
        eta = cost + rho
        if cost > Q(float(c0max)) + Q(float(weights[j])) or rho > rho_bar:
            raise AssertionError("Charged weight does not dominate the certified cost")
        spent += eta
        charged += Q(float(weights[j]))
        policy.commit(j, weights[j])
        if spent > total - (steps-step-1)*reserve:
            raise AssertionError("Budget invariant violated")
        timing["certificate"] += t1-t0
        timing["frontier"] += t2-t1
        timing["policy"] += t3-t2
        timing["flux_update"] += t4-t3
        row = {"step": step, "trusted": int(j), "fraction": j/n, "cost": float(cost),
               "budget_override": bool(overridden),
               "rho": float(rho), "eta": up_float(eta), "spent": up_float(spent)}
        if audit:
            actual = exact_step_error(v, new, lam)
            ref_new, ref_rho, _, _ = mixed_step(model, consts, ref, np.zeros(n, dtype=bool))
            ref_cost = rigorous_step_cost(cell_costs(model, tables, consts, ref, mode="separable"),
                                          np.zeros(n, dtype=bool), lam, n)
            ref_spent += ref_cost + ref_rho
            ref = ref_new
            measured = Q(0)
            for a, b in zip(new, ref):
                measured += abs(Q(float(a))-Q(float(b)))
            measured /= n
            row.update(actual_step_error=float(actual), step_certificate_holds=actual <= eta,
                       error_vs_fp_godunov=float(measured),
                       global_bound=up_float(spent+ref_spent),
                       global_certificate_holds=measured <= spent+ref_spent)
        rows.append(row)
        if record:
            menus.append((values.copy(), weights.copy()))
            raw_menus.append(diff.copy())
        v = new
    return {"state": v, "rows": rows, "spent": spent, "total_budget": total,
            "excess_budget": excess_f, "c_min": c_min, "c0max": c0max,
            "rho_bar": float(rho_bar), "menus": menus, "raw_excess": raw_menus, "timing": timing,
            "mean_trusted_fraction": float(np.mean([r["fraction"] for r in rows])) if rows else 0.0,
            "policy": policy.name}


# ---------------------------------------------------------------------------
# Better-fitted flux (review item A-4)
# ---------------------------------------------------------------------------
def sample_training_pairs(rng, count, envelope, diag_frac=0.4, sonic_frac=0.2):
    """Uniform pairs plus pairs near the diagonal a=b (smooth data) and near
    the sonic/shock lines a=0, b=0, a=-b where G is nonsmooth."""
    M = float(envelope)
    n_diag = int(diag_frac*count)
    n_son = int(sonic_frac*count)
    n_uni = count - n_diag - n_son
    uni = rng.uniform(-M, M, (n_uni, 2))
    a = rng.uniform(-M, M, n_diag)
    diag = np.column_stack((a, np.clip(a + rng.normal(0, 0.05*M, n_diag), -M, M)))
    kind = rng.integers(0, 3, n_son)
    s = rng.uniform(-M, M, n_son)
    t = rng.normal(0, 0.03*M, n_son)
    son = np.where(kind[:, None] == 0, np.column_stack((t, s)),
                   np.where(kind[:, None] == 1, np.column_stack((s, t)),
                            np.column_stack((s, np.clip(-s + t, -M, M)))))
    return np.clip(np.vstack((uni, diag, son)), -M, M)


def train_flux_v2(*, width=16, steps=6000, batch=4096, envelope=2., seed=0, lr=3e-3):
    """Mini-batch Adam with cosine decay on freshly sampled pairs each step."""
    import torch
    from .burgers import PairFlux
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = PairFlux(width=width, dropout=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr*1e-2)
    losses = []
    for it in range(steps):
        x = sample_training_pairs(rng, batch, envelope)
        y = godunov_flux(x[:, 0], x[:, 1])
        xt = torch.from_numpy(x.astype(np.float32))
        yt = torch.from_numpy(y.astype(np.float32))
        opt.zero_grad(set_to_none=True)
        loss = (model(xt)-yt).square().mean()
        loss.backward()
        opt.step()
        sched.step()
        if it % 50 == 0 or it == steps-1:
            losses.append(float(loss.detach()))
    model.eval()
    return model, losses


def fit_report(model: CertifiedFlux, envelope=2., grid=401):
    g = np.linspace(-envelope, envelope, grid)
    A, B = np.meshgrid(g, g, indexing="ij")
    x = np.column_stack((A.ravel(), B.ravel()))
    err = np.abs(model.predict(x) - godunov_flux(x[:, 0], x[:, 1]))
    diag = np.column_stack((g, g))
    derr = np.abs(model.predict(diag) - godunov_flux(g, g))
    return {"sup_error_grid": float(err.max()), "mean_error_grid": float(err.mean()),
            "sup_consistency_defect_diag": float(derr.max())}
