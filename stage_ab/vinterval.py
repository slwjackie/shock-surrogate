"""Vectorized outward-rounded binary64 intervals and interval forward-mode AD.

Every elementary operation is evaluated once in round-to-nearest by a NumPy
element-wise ufunc and the result is widened by one ulp with ``np.nextafter``.
Under the same arithmetic contract as :mod:`stage_ab.interval` (IEEE-754
binary64, correctly rounded element-wise ufuncs, gradual underflow, no FMA
contraction) the widened interval contains the exact real result.  See
``docs/stage_a2_theory.md``, Lemma V1.

Nothing here uses BLAS, reductions or fused kernels: products are formed
element-wise and accumulated in an explicit, documented order.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np

U = Fraction(1, 2**53)             # unit roundoff of binary64
ETA = Fraction(1, 2**1075)         # half the smallest subnormal
_NEG, _POS = -np.inf, np.inf


def _finite(*arrays):
    for a in arrays:
        if not np.all(np.isfinite(a)):
            raise ArithmeticError("Nonfinite interval operation; certificate unavailable")


def down(x):
    x = np.asarray(x, dtype=np.float64)
    _finite(x)
    return np.nextafter(x, _NEG)


def up(x):
    x = np.asarray(x, dtype=np.float64)
    _finite(x)
    return np.nextafter(x, _POS)


def up_float(q: Fraction) -> float:
    """Smallest-effort float >= the rational q (rounded upward)."""
    y = float(q)
    if not math.isfinite(y):
        raise ArithmeticError("Bound overflow")
    return math.nextafter(y, math.inf) if Fraction(y) < q else y


def down_float(q: Fraction) -> float:
    y = float(q)
    if not math.isfinite(y):
        raise ArithmeticError("Bound overflow")
    return math.nextafter(y, -math.inf) if Fraction(y) > q else y


def rigorous_nonneg_sum(terms) -> Fraction:
    """Exact rational upper bound of the real sum of nonnegative float terms.

    NumPy's (pairwise) summation of n nonnegative addends is below the real sum
    by at most the factor (1-u)^(n-1) >= 1-(n-1)u, independently of the order
    (Lemma F3).  The division is done in exact rationals.
    """
    t = np.asarray(terms, dtype=np.float64).ravel()
    if t.size == 0:
        return Fraction(0)
    _finite(t)
    if np.any(t < 0):
        raise ValueError("rigorous_nonneg_sum requires nonnegative terms")
    n = t.size
    if n*U >= Fraction(1, 2):
        raise ArithmeticError("Too many terms for the summation bound")
    s = float(np.sum(t))
    _finite(np.array(s))
    return Fraction(s)/(1-(n-1)*U)


@dataclass(frozen=True)
class VI:
    """Array of closed intervals [lo, hi] (broadcastable NumPy arrays)."""
    lo: np.ndarray
    hi: np.ndarray

    def __post_init__(self):
        lo = np.asarray(self.lo, dtype=np.float64)
        hi = np.asarray(self.hi, dtype=np.float64)
        _finite(lo, hi)
        if np.any(lo > hi):
            raise ArithmeticError("Invalid interval with lo > hi")
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)

    @staticmethod
    def point(x):
        x = np.asarray(x, dtype=np.float64)
        return VI(x, x)

    @staticmethod
    def hull_of(a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        return VI(np.minimum(a, b), np.maximum(a, b))

    def __add__(self, other):
        o = other if isinstance(other, VI) else VI.point(other)
        return VI(down(self.lo + o.lo), up(self.hi + o.hi))

    __radd__ = __add__

    def __neg__(self):
        return VI(-self.hi, -self.lo)

    def __sub__(self, other):
        o = other if isinstance(other, VI) else VI.point(other)
        return self + (-o)

    def __rsub__(self, other):
        return VI.point(other) - self

    def mul_point(self, w):
        """Interval times a float array (exact real w), outward rounded."""
        w = np.asarray(w, dtype=np.float64)
        a, b = self.lo*w, self.hi*w
        return VI(down(np.minimum(a, b)), up(np.maximum(a, b)))

    def __mul__(self, other):
        if not isinstance(other, VI):
            return self.mul_point(other)
        c = [self.lo*other.lo, self.lo*other.hi, self.hi*other.lo, self.hi*other.hi]
        lo = np.minimum(np.minimum(c[0], c[1]), np.minimum(c[2], c[3]))
        hi = np.maximum(np.maximum(c[0], c[1]), np.maximum(c[2], c[3]))
        return VI(down(lo), up(hi))

    __rmul__ = __mul__

    def relu(self):
        return VI(np.maximum(self.lo, 0.0), np.maximum(self.hi, 0.0))

    def abs_upper(self):
        return np.maximum(np.abs(self.lo), np.abs(self.hi))

    def hull(self, other):
        return VI(np.minimum(self.lo, other.lo), np.maximum(self.hi, other.hi))

    def where(self, mask, other):
        return VI(np.where(mask, self.lo, other.lo), np.where(mask, self.hi, other.hi))

    def __getitem__(self, idx):
        return VI(self.lo[idx], self.hi[idx])

    @property
    def shape(self):
        return np.broadcast(self.lo, self.hi).shape


def relu_slope(pre: VI) -> VI:
    """Enclosure of every ReLU slope that can occur on the box: {1}, {0} or [0,1]."""
    one = pre.lo > 0.0
    zero = pre.hi < 0.0
    lo = np.where(one, 1.0, 0.0)
    hi = np.where(zero, 0.0, 1.0)
    return VI(lo, hi)


def scale_by_slope(g: VI, s: VI) -> VI:
    """Exact product of an interval g with a slope interval s in {[0,0],[1,1],[0,1]}."""
    zero = s.hi == 0.0
    one = s.lo == 1.0
    lo = np.where(zero, 0.0, np.where(one, g.lo, np.minimum(g.lo, 0.0)))
    hi = np.where(zero, 0.0, np.where(one, g.hi, np.maximum(g.hi, 0.0)))
    return VI(lo, hi)


def network_value_enclosure(layers, a: VI, b: VI) -> VI:
    """Enclose both the real network value and the frozen FP64 evaluation.

    The accumulation order (bias first, then inputs j = 0, 1, ... in order)
    mirrors ``FrozenFlux.predict`` exactly, so Lemma V1 applies to the FP
    computation as well as to the real one (Lemma A4 of the Stage A review).
    Shapes: a, b are (M,); returns (M,).
    """
    x = [VI(a.lo[:, None], a.hi[:, None]), VI(b.lo[:, None], b.hi[:, None])]
    for k, (w, bias) in enumerate(layers):
        acc = VI.point(np.broadcast_to(bias[None, :], (x[0].shape[0], len(bias))))
        for j in range(w.shape[1]):
            acc = acc + x[j].mul_point(w[None, :, j])
        if k+1 < len(layers):
            acc = acc.relu()
            x = [acc[:, jj:jj+1] for jj in range(acc.shape[1])]
        else:
            return acc[:, 0]
    raise ValueError("empty network")


def network_value_and_gradient(layers, a: VI, b: VI):
    """Forward-mode interval AD of the REAL ReLU network on boxes a x b.

    Returns (value, grad_a, grad_b) enclosures of shape (M,).  For every point
    x of the box and every ReLU activation pattern that is consistent with the
    sign information of the pre-activation enclosures, the corresponding
    piecewise gradient lies in (grad_a, grad_b) (Lemma G1).
    """
    m = a.shape[0]
    val = [VI(a.lo[:, None], a.hi[:, None]), VI(b.lo[:, None], b.hi[:, None])]
    ga = [VI.point(np.ones((m, 1))), VI.point(np.zeros((m, 1)))]
    gb = [VI.point(np.zeros((m, 1))), VI.point(np.ones((m, 1)))]
    for k, (w, bias) in enumerate(layers):
        width = len(bias)
        acc = VI.point(np.broadcast_to(bias[None, :], (m, width)))
        dacc_a = VI.point(np.zeros((m, width)))
        dacc_b = VI.point(np.zeros((m, width)))
        for j in range(w.shape[1]):
            wj = w[None, :, j]
            acc = acc + val[j].mul_point(wj)
            dacc_a = dacc_a + ga[j].mul_point(wj)
            dacc_b = dacc_b + gb[j].mul_point(wj)
        if k+1 < len(layers):
            slope = relu_slope(acc)
            out = acc.relu()
            dacc_a = scale_by_slope(dacc_a, slope)
            dacc_b = scale_by_slope(dacc_b, slope)
            val = [out[:, jj:jj+1] for jj in range(width)]
            ga = [dacc_a[:, jj:jj+1] for jj in range(width)]
            gb = [dacc_b[:, jj:jj+1] for jj in range(width)]
        else:
            return acc[:, 0], dacc_a[:, 0], dacc_b[:, 0]
    raise ValueError("empty network")


def _half_square_bounds(x_lo, x_hi):
    """Outward bounds of s(x)=x^2/2 for scalars x_lo <= x_hi with x_lo >= 0."""
    lo = down(down(x_lo*x_lo)*0.5)
    hi = up(up(x_hi*x_hi)*0.5)
    return lo, hi


def godunov_value_enclosure(a: VI, b: VI) -> VI:
    """G(a,b)=max(p(a),q(b)), p(a)=(a+)^2/2 nondecreasing, q(b)=(b-)^2/2 nonincreasing."""
    ap_lo, ap_hi = np.maximum(a.lo, 0.0), np.maximum(a.hi, 0.0)
    bm_lo, bm_hi = np.minimum(b.lo, 0.0), np.minimum(b.hi, 0.0)
    p_lo, _ = _half_square_bounds(ap_lo, ap_lo)
    _, p_hi = _half_square_bounds(ap_hi, ap_hi)
    # |b-| is nonincreasing in b: the largest q at b.lo, the smallest at b.hi.
    q_lo, _ = _half_square_bounds(-bm_hi, -bm_hi)
    _, q_hi = _half_square_bounds(-bm_lo, -bm_lo)
    return VI(np.maximum(p_lo, q_lo), np.maximum(p_hi, q_hi))


def godunov_gradient_cases(a: VI, b: VI):
    """Enclosure of the a.e. gradients of G on the box, split by active branch.

    Returns (has_p, grad_p_a, has_q, grad_q_b):
      branch p active -> gradient (a+, 0) with a+ in grad_p_a;
      branch q active -> gradient (0, b-) with b- in grad_q_b.
    ``has_p``/``has_q`` are False only when the other branch strictly dominates
    on the whole box (Lemma G2).
    """
    ap_lo, ap_hi = np.maximum(a.lo, 0.0), np.maximum(a.hi, 0.0)
    bm_lo, bm_hi = np.minimum(b.lo, 0.0), np.minimum(b.hi, 0.0)
    p_min, _ = _half_square_bounds(ap_lo, ap_lo)
    _, p_max = _half_square_bounds(ap_hi, ap_hi)
    q_min, _ = _half_square_bounds(-bm_hi, -bm_hi)
    _, q_max = _half_square_bounds(-bm_lo, -bm_lo)
    only_p = p_min > q_max
    only_q = q_min > p_max
    return ~only_q, VI(ap_lo, ap_hi), ~only_p, VI(bm_lo, bm_hi)


def difference_enclosure(layers, a: VI, b: VI, da: VI, db: VI):
    """Enclose D(y) - D(x) = int_0^1 grad D(x+t d).d dt for any x, y=x+d in the box.

    D = F_real - G.  Returns an interval array J (Theorem D1).
    """
    _, gfa, gfb = network_value_and_gradient(layers, a, b)
    has_p, gpa, has_q, gqb = godunov_gradient_cases(a, b)
    j_p = (gfa - gpa)*da + gfb*db
    j_q = gfa*da + (gfb - gqb)*db
    # Hull over the branches that can be active; at least one always is.
    lo = np.where(has_p & has_q, np.minimum(j_p.lo, j_q.lo), np.where(has_p, j_p.lo, j_q.lo))
    hi = np.where(has_p & has_q, np.maximum(j_p.hi, j_q.hi), np.where(has_p, j_p.hi, j_q.hi))
    return VI(lo, hi)


def hidden_slopes_degenerate(layers, a: VI, b: VI):
    """True where every hidden pre-activation enclosure is strictly signed on the box,
    i.e. the network is affine on the box with a certified activation pattern."""
    m = a.shape[0]
    ok = np.ones(m, dtype=bool)
    x = [VI(a.lo[:, None], a.hi[:, None]), VI(b.lo[:, None], b.hi[:, None])]
    for k, (w, bias) in enumerate(layers[:-1]):
        acc = VI.point(np.broadcast_to(bias[None, :], (m, len(bias))))
        for j in range(w.shape[1]):
            acc = acc + x[j].mul_point(w[None, :, j])
        ok &= np.all((acc.lo > 0) | (acc.hi < 0), axis=1)
        acc = acc.relu()
        x = [acc[:, jj:jj+1] for jj in range(acc.shape[1])]
    return ok
