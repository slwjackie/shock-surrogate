"""A three-stage learned conservative macro-step for H=8, dt/dx=0.2.

The five scalar weights are fitted offline to S^8, the eight-fold discrete
Godunov update. Each nonlinear spatial layer uses the Burgers face activation.
The output weights sum to H*dt/dx. Inference uses exactly three layers.
"""
from dataclasses import dataclass
from pathlib import Path
import argparse

import numpy as np

H = 8
LAM = 0.2
T = H * LAM
WEIGHTS = Path(__file__).resolve().parents[1] / 'reports/macro8/model.npz'


def flux_difference(u):
    """Periodic left-face minus right-face flux; the final axis is space."""
    right = np.roll(u, -1, axis=-1)
    speed = np.maximum(np.maximum(u, -right), 0.0)
    face = 0.5 * speed * speed
    return np.roll(face, 1, axis=-1) - face


@dataclass(frozen=True)
class MacroFlux:
    weights: np.ndarray

    def step(self, state):
        u = np.asarray(state, dtype=np.float64)
        a, b, c, d, e = self.weights
        f0 = flux_difference(u)
        f1 = flux_difference(u + a * f0)
        f2 = flux_difference(u + b * f0 + c * f1)
        return u + (T - d - e) * f0 + d * f1 + e * f2

    @classmethod
    def load(cls, path=WEIGHTS):
        with np.load(path) as data:
            if int(data['h']) != H or float(data['lam']) != LAM:
                raise ValueError('Checkpoint horizon or time-to-space ratio differs')
            return cls(data['weights'].copy())


def labels(u):
    for _ in range(H):
        u = u + LAM * flux_difference(u)
    return u


def training_set(seed, count):
    from .experiments_a2 import ic_family, discretize
    from certified_burgers.initial_conditions import sample_states

    family = np.stack([discretize(ic, 128) for ic in ic_family(seed, count)])
    u = np.concatenate([family, sample_states(count, 128, seed=seed + 1, strong_ood=True)])
    states, targets = [], []
    for _ in range(3):
        states.append(u)
        u = labels(u)
        targets.append(u)
    return np.concatenate(states), np.concatenate(targets)


def train(path=WEIGHTS, seed=1100, count=54):
    from scipy.optimize import least_squares

    u, target = training_set(seed, count)
    f0 = flux_difference(u)
    # Balance shocks against small nonconstant inputs, while retaining the
    # influence of larger-amplitude examples on the global absolute error.
    scale = 0.02 + np.max(np.abs(target - u), axis=1, keepdims=True)
    def residual(w):
        a, b, c, d, e = w
        f1 = flux_difference(u + a * f0)
        f2 = flux_difference(u + b * f0 + c * f1)
        estimate = (T - d - e) * f0 + d * f1 + e * f2
        return ((estimate - (target-u)) / scale).ravel()

    initial = [.65, .14, 1.01, .52, .68]
    solution = least_squares(residual, initial, bounds=([.1, -2, .1, -1, .1],
                                                        [1.5, 2, 2.5, 2, 2]),
                             max_nfev=100, ftol=1e-10, xtol=1e-10)
    if not solution.success:
        raise RuntimeError(solution.message)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, weights=solution.x, h=H, lam=LAM, seed=seed, count=count)
    return MacroFlux(solution.x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=WEIGHTS)
    parser.add_argument('--seed', type=int, default=1100)
    args = parser.parse_args()
    model = train(args.out, args.seed)
    print('Fitted scalar layer weights:', model.weights)
