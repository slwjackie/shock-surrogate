"""A deterministic three-stage control for the learned Burgers macro-step."""
from __future__ import annotations

import numpy as np

from .macro_flux import H, LAM, T, MacroFlux


def matched_macro():
    """Match three Taylor moments of the eight original Euler updates.

    For a smooth spatial operator F, expand each F evaluated by the eight
    Godunov substeps about the initial state. The average coefficients for
    F'F, F''(F,F)/2 and F'(F'F) are mean(j*h), mean((j*h)**2) and
    mean(j*(j-1)*h**2/2), respectively, for j=0,...,7 and h=dt/dx.
    At transitions of the piecewise Burgers flux no such expansion is needed
    to run this method; accuracy there is tested empirically.
    """
    indices = np.arange(H, dtype=np.float64)
    first = np.mean(indices * LAM)
    second = np.mean((indices * LAM) ** 2)
    chain = np.mean(indices * (indices - 1) * LAM**2 / 2)

    # Fixed abscissae behind the learned method's initial guess.
    a, s = 0.65, 1.15
    # Solve w1*a + w2*s = first, w1*a**2 + w2*s**2 = second,
    # and w2*c*a = chain. The first output weight is 1-w1-w2.
    w2 = (second - a * first) / (s * (s - a))
    w1 = (first - w2 * s) / a
    c = chain / (w2 * a)
    return MacroFlux(np.array([a, s - c, c, T * w1, T * w2]))


fixed_macro_step = matched_macro().step
