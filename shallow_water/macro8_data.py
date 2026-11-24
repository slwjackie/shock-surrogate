"""Disjoint whole trajectories for a 1D wet, flat-bed shallow-water operator.

Each training or evaluation item represents an independent initial profile and
one coarse-grid Godunov evolution. Refinement projects the *same supplied*
piecewise-constant initial state onto a finer grid at equal physical time.
"""
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .rollout_study import compile_rollout, exact_cell_averages

N = 256
G = 9.81
LAMBDA = 0.07  # dt/dx in seconds per metre, unit-length domain
STEPS = 8
DURATION = STEPS * LAMBDA / N
TOL = 1e-8  # local Godunov middle-depth tolerance, metres

SPLIT_SEEDS = {"train": 68111, "dev": 68213, "mine": 68329, "calibrate": 68657,
               "test": 68437, "shift": 68543}
FAMILIES = ("smooth", "dam", "interacting", "evolved")


class ReferenceSolver:
    """Compile the existing C finite-volume solver once per context."""
    def __enter__(self):
        self._tmp = TemporaryDirectory()
        try:
            self.kernel = compile_rollout(Path(self._tmp.name))
        except BaseException:
            self._tmp.cleanup()
            raise
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        self._tmp.cleanup()

    def advance(self, q, steps=STEPS, mode=6, tol=TOL, duration=None):
        """Return the endpoint. Default dt/dx is fixed independently of grid size."""
        out = np.array(q, dtype=np.float64, order="C", copy=True)
        if out.ndim != 2 or out.shape[1] != 2 or out.shape[0] < 2:
            raise ValueError("Expected a (cells, 2) array of depth and discharge")
        if not isinstance(steps, (int, np.integer)) or steps < 1:
            raise ValueError("Use a positive integer step count")
        n = len(out)
        t = steps * LAMBDA / n if duration is None else float(duration)
        status = self.kernel(n, out, 1., t, int(steps), tol, mode)
        if status:
            raise RuntimeError(f"Native rollout returned {status} on N={n}, "
                               f"steps={steps}, duration={t}, mode={mode}")
        return out

    def refined(self, q, refinement=4, blocks=1):
        """Fine Godunov endpoint, restricted by conservative cell averaging."""
        if not isinstance(refinement, (int, np.integer)) or refinement < 1:
            raise ValueError("Use a positive integer refinement")
        if not isinstance(blocks, (int, np.integer)) or blocks < 1:
            raise ValueError("Use a positive integer block count")
        lifted = np.repeat(q, refinement, axis=0)
        end = self.advance(lifted, steps=STEPS*refinement*blocks)
        return end.reshape(len(q), refinement, 2).mean(axis=1)


@dataclass(frozen=True)
class Batch:
    q: np.ndarray
    target: np.ndarray
    families: tuple
    ids: tuple


def _conservative(h, u):
    return np.column_stack((h, h*u))


def _smooth(rng, shift):
    x = (np.arange(N) + .5)/N
    # Compact support protects the outflow boundaries; sine-squared tapers
    # have zero first derivative where they meet the uniform exterior.
    start, stop = rng.uniform(.09, .16), rng.uniform(.84, .91)
    y = np.clip((x-start)/(stop-start), 0., 1.)
    taper = np.where((x > start) & (x < stop), np.sin(np.pi*y)**2, 0.)
    modes = rng.integers(1, 6 if shift else 4, size=2)
    phases = rng.uniform(-np.pi, np.pi, size=3)
    w1 = np.sin(2*np.pi*modes[0]*y + phases[0])
    w2 = np.sin(2*np.pi*modes[1]*y + phases[1])
    w3 = np.sin(2*np.pi*modes[0]*y + phases[2])
    base = rng.uniform(1.2, 3.3 if shift else 2.5)
    amp = rng.uniform(.25,.43) if shift else rng.uniform(.09,.32)
    h = base*(1 + amp*taper*(.65*w1+.35*w2))
    u = rng.uniform(-.5,.5) + rng.uniform(.4,1.2 if shift else .85)*taper*w3
    return _conservative(h, u)


def _dam(rng, shift):
    # Common background motion preserves one shock and one rarefaction.
    shallow = rng.uniform(.42, .95) if shift else rng.uniform(.65,1.45)
    ratio = rng.uniform(2.8,4.7) if shift else rng.uniform(1.25,2.9)
    states = [shallow, shallow*ratio]
    if rng.random() < .5:
        states.reverse()
    jump = rng.integers(N//4, 3*N//4+1)
    h = np.full(N, states[1])
    h[:jump] = states[0]
    return _conservative(h, rng.uniform(-.7,.7))


def _interacting(rng, shift):
    # Lower centre depth creates inward-propagating bores. Some narrow
    # central shelves disappear inside the eight-step observation interval.
    centre = rng.integers(N//3, 2*N//3)
    span = rng.integers(3,12)
    hl, hr = rng.uniform(1.7, 3.8 if shift else 2.8, size=2)
    hm = rng.uniform(.40,.85) if shift else rng.uniform(.85,1.45)
    drift = rng.uniform(-.45,.45)
    ul = drift + rng.uniform(.1,1.35 if shift else .85)
    ur = drift - rng.uniform(.1,1.35 if shift else .85)
    h = np.full(N, hr)
    u = np.full(N, ur)
    h[:centre], u[:centre] = hl, ul
    h[centre:centre+span], u[centre:centre+span] = hm, drift
    return _conservative(h,u)


def make_split(split, count, solver):
    """Return a reproducible prefix of independent, interleaved trajectory families."""
    if split not in SPLIT_SEEDS:
        raise ValueError(f"Choose a split in {tuple(SPLIT_SEEDS)}")
    if not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("Use a positive number of trajectories")
    q = np.empty((count, N, 2))
    target = np.empty_like(q)
    families, ids = [], []
    shift = split == "shift"
    for i in range(count):
        family = FAMILIES[i % len(FAMILIES)]
        tid = f"{split}-{i:06d}"
        rng = np.random.default_rng(np.random.SeedSequence([SPLIT_SEEDS[split], i]))
        try:
            if family == "smooth":
                state = _smooth(rng, shift)
            elif family == "dam":
                state = _dam(rng, shift)
            elif family == "interacting":
                state = _interacting(rng, shift)
            else:
                profile = _dam if rng.random() < .5 else _interacting
                state = solver.advance(profile(rng,shift), steps=int(rng.integers(8,33)))
            q[i] = state
            target[i] = solver.advance(state)
        except RuntimeError as exc:
            # Do not replace failed examples with easier random draws.
            raise RuntimeError(f"Reference failed on {tid} ({family})") from exc
        families.append(family)
        ids.append(tid)
    return Batch(q, target, tuple(families), tuple(ids))


def analytical_dam(q, blocks=1):
    """Exact Riemann cell averages for one grid-aligned jump before boundary hits."""
    q = np.asarray(q)
    if q.ndim != 2 or q.shape[1] != 2 or not np.isfinite(q).all() or np.min(q[:,0]) <= 0:
        raise ValueError("Expected finite wet conserved cells")
    changes = np.flatnonzero(np.any(q[1:] != q[:-1], axis=1))
    if len(changes) != 1:
        raise ValueError("Expected a single two-state Riemann discontinuity")
    if not isinstance(blocks, (int, np.integer)) or blocks < 1:
        raise ValueError("Use a positive integer block count")
    (hl, ml), (hr, mr) = q[[0,-1]]
    jump = int(changes[0]+1)
    duration = STEPS*LAMBDA*blocks/len(q)
    exact, _ = exact_cell_averages((hl, ml/hl, hr, mr/hr), len(q), duration, jump)
    return exact
