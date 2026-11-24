"""Direct conservative H=8 predictor for periodic Burgers states at dt/dx=.2.

Train with OPENBLAS_NUM_THREADS=1 python -m stage_ab.macro_network.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np

WEIGHTS = Path(__file__).resolve().parents[1] / "reports/macro8/net_h8.npz"
H = 8
LAM = 0.2


@lru_cache(maxsize=32)
def indices(n):
    return (np.arange(n)[:, None] + np.arange(-3, 5)) % n


def features(state):
    """Eight neighboring cell values determine the features at each face."""
    cells = state[..., indices(state.shape[-1])]
    slopes = np.diff(cells, axis=-1)
    scale = np.max(np.abs(slopes), axis=-1)
    x = np.empty((*scale.shape, 10))
    x[..., 0:2] = cells[..., 3:5]
    x[..., 2] = scale
    x[..., 3:] = slopes / np.where(scale > 0, scale, 1)[..., None]
    base = 0.5 * np.maximum(np.maximum(x[..., 0], -x[..., 1]), 0)**2
    return x, scale, base


def load_network(path=WEIGHTS):
    """Return a single-call macro predictor for one state or a batch."""
    with np.load(path) as f:
        w, b, v, c = (f[k].copy() for k in ("W", "b", "v", "c"))

    def predict(state):
        u = np.asarray(state, dtype=np.float64)
        x, scale, base = features(u)
        flux = base + scale * (np.maximum(x @ w + b, 0) @ v + c)
        return u - (H * LAM) * (flux - np.roll(flux, 1, axis=-1))

    return predict


def reference(state):
    u = state.copy()
    total = np.zeros_like(state)
    for _ in range(H):
        right = np.roll(u, -1, axis=-1)
        flux = 0.5 * np.maximum(np.maximum(u, -right), 0)**2
        total += flux
        u -= LAM * (flux - np.roll(flux, 1, axis=-1))
    return u, total / H


def generate(seed, count):
    from .experiments_a2 import ic_family, discretize
    from certified_burgers.initial_conditions import sample_states

    rng = np.random.default_rng(seed)
    family = np.stack([discretize(ic, 128) for ic in ic_family(seed, count)])
    ood = sample_states(count, 128, seed=seed + 1, strong_ood=True)
    u = np.concatenate([family, ood])
    samples, labels = [], []
    for _ in range(8):
        x, scale, base = features(u)
        u, avg = reference(u)
        rows = scale > 1e-7
        samples.append(x[rows])
        labels.append((avg[rows] - base[rows]) / scale[rows])
    x = np.concatenate(samples)
    y = np.concatenate(labels)
    # Thin abundant gentle slopes to balance the less common shock faces.
    keep = (x[:, 2] > .075) | (rng.random(len(x)) < .23)
    return x[keep], y[keep]


def train(path=WEIGHTS, seed=78351, steps=7500):
    """Fit both the hidden and output layers using Adam on labelled fluxes."""
    rng = np.random.default_rng(seed)
    x, y = generate(seed, 180)
    width = 16
    w = rng.normal(0, .26, (10, width))
    b = rng.uniform(-.3, .3, width)
    v = rng.normal(0, .08, width)
    c = np.zeros(1)
    params = (w, b, v, c)
    first = [np.zeros_like(p) for p in params]
    second = [np.zeros_like(p) for p in params]
    # Mix equal-probability sampling over faces and over feature-scale bins.
    bins = np.digitize(x[:, 2], [.025, .065, .16, .36, .8])
    groups = [np.flatnonzero(bins == i) for i in range(6)]
    for i in range(1, steps + 1):
        idx = np.concatenate([rng.choice(g, 80) for g in groups] +
                             [rng.integers(len(x), size=160)])
        xb = x[idx]
        act = np.maximum(xb @ w + b, 0)
        delta = (act @ v + c - y[idx]) / len(idx)
        dz = (delta[:, None] * v) * (act > 0)
        grads = (xb.T @ dz, dz.sum(axis=0), act.T @ delta, delta.sum(keepdims=True))
        rate = .003 * (.08 + .92 * (1 - i / steps))
        for j, (p, grad) in enumerate(zip(params, grads)):
            first[j] *= .9
            first[j] += .1 * grad
            second[j] *= .999
            second[j] += .001 * grad**2
            p -= rate * (first[j] / (1 - .9**i)) / (np.sqrt(second[j] / (1 - .999**i)) + 1e-8)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, W=w, b=b, v=v, c=c[0], seed=seed, steps=steps)
    return load_network(path)


if __name__ == "__main__":
    from .experiments_a2 import ic_family, discretize
    predict = train()
    u = np.stack([discretize(ic, 128) for ic in ic_family(51771, 60)])
    exact, _ = reference(u)
    errors = np.mean(np.abs(predict(u) - exact), axis=1)
    print(f"Validation across 60 fresh states: mean L1 {errors.mean():.6g}, max {errors.max():.6g}")
