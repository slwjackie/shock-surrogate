"""Frozen FP64 flux evaluator and reference used for inference and certification."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path

import numpy as np

from .interval import Interval as I


def exact_godunov(a: Q, b: Q) -> Q:
    """Exact Burgers numerical flux: max((a_+)^2, (b_-)^2)/2."""
    return max(max(a, Q(0))**2, min(b, Q(0))**2)/2


@dataclass
class FrozenFlux:
    layers: list[tuple[np.ndarray, np.ndarray]]
    arithmetic: str = "numpy-float64-separate-multiply-add-v1"

    def __post_init__(self):
        previous = 2
        clean = []
        for w, b in self.layers:
            w, b = np.asarray(w, dtype=np.float64), np.asarray(b, dtype=np.float64)
            if w.ndim != 2 or w.shape[1] != previous or b.shape != (w.shape[0],):
                raise ValueError("Invalid frozen network shape")
            if not (np.isfinite(w).all() and np.isfinite(b).all()):
                raise ValueError("Nonfinite model weights")
            clean.append((w.copy(), b.copy()))
            previous = len(b)
        if not clean or previous != 1:
            raise ValueError("Flux network must have one scalar output")
        self.layers = clean

    def payload(self):
        return {"arithmetic": self.arithmetic,
                "layers": [{"weight": w.tolist(), "bias": b.tolist()} for w, b in self.layers]}

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(self.payload(), sort_keys=True, allow_nan=False).encode()).hexdigest()

    def predict(self, pairs):
        """Use precisely the elementary operation order enclosed offline."""
        x = np.asarray(pairs, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all():
            raise ValueError("Expected finite [batch,2] pairs")
        for k, (w, b) in enumerate(self.layers):
            y = np.broadcast_to(b, (len(x), len(b))).copy()
            for j in range(w.shape[1]):
                product = np.multiply(x[:, j:j+1], w[None, :, j])
                y = np.add(y, product)
            x = np.maximum(y, 0) if k+1 < len(self.layers) else y
        return x[:, 0]

    def enclosure(self, box):
        x = list(box)
        for k, (w, b) in enumerate(self.layers):
            y = []
            for row, bias in zip(w, b):
                accum = I(bias)
                for coefficient, item in zip(row, x):
                    accum = accum + I(coefficient)*item
                y.append(accum.relu() if k+1 < len(self.layers) else accum)
            x = y
        return x[0]

    def real_predict(self, pairs):
        """Exact rational network values, used only for proof/audit tests."""
        out = []
        for pair in pairs:
            x = [Q(float(v)) for v in pair]
            for k, (w, b) in enumerate(self.layers):
                x = [sum((Q(float(a))*v for a, v in zip(row, x)), Q(float(bias)))
                     for row, bias in zip(w, b)]
                if k+1 < len(self.layers):
                    x = [max(Q(0), v) for v in x]
            out.append(x[0])
        return out

    def lipschitz_bound(self):
        """Real ReLU network Lipschitz bound in the infinity norm, exact rational."""
        result = Q(1)
        for w, _ in self.layers:
            result *= max(sum((abs(Q(float(x))) for x in row), Q(0)) for row in w)
        return result

    def fp_roundoff_bound(self, box):
        """Uniform forward-error bound between the frozen FP64 and real network.

        Gamma_(2d) dot-product estimate plus an absolute subnormal allowance;
        component errors propagate through |W| and 1-Lipschitz ReLU.
        """
        x, errors = list(box), [Q(0), Q(0)]
        u, tiny = Q(1, 2**53), Q(1, 2**1074)
        for layer_index, (w, b) in enumerate(self.layers):
            values, new_errors = [], []
            count = 2*w.shape[1]
            gamma = count*u/(1-count*u)
            for row, bias in zip(w, b):
                accum = I(bias)
                magnitude = abs(Q(float(bias)))
                propagated = Q(0)
                for coefficient, item, error in zip(row, x, errors):
                    accum = accum+I(coefficient)*item
                    magnitude += abs(Q(float(coefficient)))*Q(float(item.abs_upper()))
                    propagated += abs(Q(float(coefficient)))*error
                values.append(accum.relu() if layer_index+1 < len(self.layers) else accum)
                new_errors.append(propagated+gamma*magnitude+count*tiny/(1-count*u))
            x, errors = values, new_errors
        return errors[0]

    def save(self, path):
        Path(path).write_text(json.dumps(self.payload(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data["arithmetic"] != "numpy-float64-separate-multiply-add-v1":
            raise ValueError("Unknown certified arithmetic contract")
        return cls([(np.array(x["weight"]), np.array(x["bias"])) for x in data["layers"]])
