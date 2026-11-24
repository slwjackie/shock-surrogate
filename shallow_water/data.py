"""Wet 1-D shallow-water interface data, bisection labels and Decimal audits.

The stored columns are h_L, u_L, h_R, u_R in metres and metres/second.
The independent oracle solves the two Rankine-Hugoniot/rarefaction wave
curves. All training and accuracy work is outside the native timing loops.
"""
from decimal import Decimal, localcontext

import numpy as np

G = 9.81
EVALUATION_SEEDS = {"iid": 50001, "shift": 50002, "smooth": 50003}


def _log_uniform(rng, low, high, n):
    return np.exp(rng.uniform(np.log(low), np.log(high), n))


def sample(n, seed, split="train"):
    """Draw n strictly wet, non-vacuum interfaces from a reproducible mixture.

    For IID data, draw each depth log-uniformly from 0.25 to 6 m. At 30% of
    interfaces the neighboring states are close. The smooth split makes all
    interfaces close. The shift split pairs depths from 0.018–0.23 m and
    5.5–15 m; left/right ordering is random. Relative velocity jumps remain
    below the dry-middle threshold of 2*(sqrt(g*hL)+sqrt(g*hR)).
    """
    if n < 0 or split not in ("train", "iid", "shift", "smooth"):
        raise ValueError("Provide a nonnegative count and a known split")
    rng = np.random.default_rng(seed)
    if split == "shift":
        low = _log_uniform(rng, .018, .23, n)
        high = _log_uniform(rng, 5.5, 15., n)
        flip = rng.random(n) < .5
        hl, hr = np.where(flip, high, low), np.where(flip, low, high)
        velocity_jump = rng.uniform(-2.2, 1.93, n)
    else:
        hl = _log_uniform(rng, .25, 6., n)
        hr = _log_uniform(rng, .25, 6., n)
        weak = np.ones(n, dtype=bool) if split == "smooth" else rng.random(n) < .30
        hr[weak] = np.clip(hl[weak] * np.exp(rng.uniform(-.02, .02, weak.sum())), .25, 6.)
        velocity_jump = rng.uniform(-1.2, 1.45, n)
        velocity_jump[weak] = rng.uniform(-.012, .012, weak.sum())
    cl, cr = np.sqrt(G * hl), np.sqrt(G * hr)
    du = velocity_jump * (cl + cr)
    advection = rng.uniform(-1, 1, n) * .5 * (cl + cr)
    return np.column_stack((hl, advection - .5*du, hr, advection + .5*du))


def _wave(h, side, g):
    """Shock curve above side, rationalized rarefaction curve below side."""
    # reference() evaluates positive arguments only.
    factor = np.where(h > side, np.sqrt(.5*g*(1/h + 1/side)),
                      2*np.sqrt(g)/(np.sqrt(h) + np.sqrt(side)))
    return (h-side)*factor


def reference(states, g=G):
    """Vectorized FP64 bracket and bisection oracle for positive star depths."""
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 4:
        raise ValueError("States must have four columns [hl, ul, hr, ur]")
    if not np.isfinite(states).all() or not np.isfinite(g) or g <= 0:
        raise ValueError("Require finite inputs and positive gravity")
    hl, ul, hr, ur = states.T
    if np.any(hl <= 0) or np.any(hr <= 0):
        raise ValueError("Initial depths must be positive")
    du = ur-ul
    cl, cr = np.sqrt(g*hl), np.sqrt(g*hr)
    if np.any(du >= 2*(cl+cr)):
        raise ValueError("A dry middle state is outside this experiment")
    def phi(x):
        return _wave(x, hl, g) + _wave(x, hr, g) + du

    # When both waves are rarefactions, the closed-form estimate also gives
    # a useful initial scale near vacuum. The sign is independently checked.
    two_rare_root = (.5*(cl+cr) - .25*du)**2/g
    upper = np.minimum(np.maximum(hl, hr), 2*two_rare_root)
    if np.any(upper <= 0) or not np.isfinite(upper).all():
        raise ArithmeticError("Root bracket outside the FP64 oracle range")
    for _ in range(1024):
        signs = phi(upper)
        if not np.isfinite(signs).all():
            raise ArithmeticError("Nonfinite oracle residual")
        expand = signs < 0
        if not expand.any():
            break
        upper[expand] *= 2
    else:
        raise ArithmeticError("FP64 oracle failed to bracket a positive root")
    lower = np.zeros_like(upper)
    for _ in range(72):
        middle = lower + .5*(upper-lower)
        negative = phi(middle) < 0
        lower = np.where(negative, middle, lower)
        upper = np.where(negative, upper, middle)
    return lower + .5*(upper-lower)


def decimal_audit(state, candidate, tol, g=G):
    """Independent 90-digit residual-sign audit at exact-binary input values.

    This is an independent numerical audit, with Decimal square roots and
    operations at 90 digits; sw_certify supplies the online certificate.
    """
    if not np.isfinite([*state, candidate, tol, g]).all():
        return False
    with localcontext() as ctx:
        ctx.prec = 90
        hl, ul, hr, ur, guess, eps, grav = (Decimal.from_float(float(v))
                                           for v in (*state, candidate, tol, g))
        if min(hl, hr, guess, grav) <= 0 or eps < 0:
            return False
        def phi(x):
            def term(side):
                if x > side:
                    return (x-side)*(grav*(1/x + 1/side)/2).sqrt()
                return 2*((grav*x).sqrt() - (grav*side).sqrt())
            return term(hl) + term(hr) + ur-ul
        if phi(Decimal(0)) >= 0:
            return False
        return phi(max(Decimal(0), guess-eps)) <= 0 <= phi(guess+eps)
