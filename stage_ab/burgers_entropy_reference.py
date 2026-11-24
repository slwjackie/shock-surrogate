"""Offline entropy-solution cell averages for inviscid Burgers on a unit circle."""
from __future__ import annotations

import math

import numpy as np


def entropy_averages(state, t):
    """Evolve the supplied piecewise constant cell values to physical time t.

    The Hopf–Lax formula minimizes U0(y) + (x-y)^2/(2t) at each cell
    boundary x, where U0 is a primitive of the periodic initial data. The
    minimization is analytic on each initial cell. Differencing this
    potential gives the entropy-solution cell averages, up to FP64 roundoff.
    This offline reference performs no finite-volume time steps.
    """
    values = np.asarray(state, dtype=np.float64)
    t = float(t)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("state must be a nonempty finite 1-D array")
    if not math.isfinite(t) or t < 0:
        raise ValueError("t must be finite and nonnegative")
    if t == 0 or np.all(values == values[0]):
        return values.copy()

    n = len(values)
    primitive = np.r_[0.0, np.cumsum(values) / n]
    mean = primitive[-1]
    # U0(y+k) = U0(y)+k*mean. Among integer copies of any y only the
    # nearest to x-t*mean can minimize. Finite wave speeds tighten this
    # window further at short times. Include a neighboring cell at ties.
    lo = max(-t * values.max(), -t * mean - 0.5)
    hi = min(-t * values.min(), -t * mean + 0.5)
    offsets = np.arange(math.floor(n * lo) - 1, math.ceil(n * hi) + 1)
    potential = np.empty(n + 1)
    batch_size = max(1, 262144 // len(offsets))
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        edge = np.arange(start, stop)[:, None]
        periods, cells = np.divmod(edge + offsets, n)
        slope = values[cells]
        # Write the minimizer as y = x + d on this closed initial cell.
        left = offsets / n
        d = np.clip(-t * slope, left, (offsets + 1) / n)
        initial = primitive[cells] + periods * mean + slope * (d - left)
        potential[start:stop] = np.min(initial + 0.5 * d * (d / t), axis=1)
    potential[-1] = potential[0] + mean
    return np.diff(potential) * n
