"""Certified signed flux-error queries using offline anchors and derivatives.

For a face x in a bin and its representable center c, the segment from c to x
lies in the closed bin. Integrating the directional derivative of the
piecewise smooth D = F_real - G along this segment encloses D(x) in
D(c) + grad_a(D) * (x_a-c_a) + grad_b(D) * (x_b-c_b). Consistent ReLU slopes
and all possible active Godunov branches are included on the whole bin.
The anchor interval is widened offline by the frozen evaluator's uniform
forward-error bound to enclose F_fp(x)-G(x) at online queries.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
import time

import numpy as np

from .vinterval import (VI, godunov_gradient_cases, godunov_value_enclosure,
                        network_value_and_gradient, network_value_enclosure)


def _immutable(array):
    """Use immutable backing storage even if a caller requests write access."""
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class AnchorTable:
    model_hash: str
    edges: np.ndarray
    centers: np.ndarray
    # (left bin, right bin, term, endpoint). Terms: anchor, grad_a, grad_b.
    coefficients: np.ndarray
    build_seconds: float

    @property
    def bins(self):
        return len(self.centers)

    @property
    def nbytes(self):
        return self.edges.nbytes + self.centers.nbytes + self.coefficients.nbytes

    @classmethod
    def build(cls, model, tables, *, bins=128, chunk_size=512):
        if model.digest != tables.model_hash:
            raise ValueError("Model/certificate fingerprint mismatch")
        if not isinstance(bins, Integral) or bins < 1 or not isinstance(chunk_size, Integral) or chunk_size < 1:
            raise ValueError("Positive integer bin and chunk counts are required")
        start = time.perf_counter()
        edges = np.linspace(tables.edges[0], tables.edges[-1], bins+1)
        if not np.all(np.diff(edges) > 0.0):
            raise ValueError("Anchor partition contains empty bins")
        centers = np.clip(0.5*edges[:-1] + 0.5*edges[1:], edges[:-1], edges[1:])
        # Works even if the anchor grid does not align with the base grid.
        eps = float(np.max(tables.eps_fp))
        fp_error = VI(-eps, eps)
        coefficients = np.empty((bins*bins, 3, 2))
        for begin in range(0, len(coefficients), chunk_size):
            end = min(begin + chunk_size, len(coefficients))
            i, j = np.divmod(np.arange(begin, end), bins)
            ca, cb = VI.point(centers[i]), VI.point(centers[j])
            anchor = network_value_enclosure(model.layers, ca, cb) - godunov_value_enclosure(ca, cb)
            a, b = VI(edges[i], edges[i+1]), VI(edges[j], edges[j+1])
            _, grad_fa, grad_fb = network_value_and_gradient(model.layers, a, b)
            has_p, pa, has_q, qb = godunov_gradient_cases(a, b)
            # p'(a) >= 0, q'(b) <= 0. The inactive coordinate has slope zero.
            grad_ga = VI(np.where(has_q, 0., pa.lo), np.where(has_p, pa.hi, 0.))
            grad_gb = VI(np.where(has_q, qb.lo, 0.), np.where(has_p, 0., qb.hi))
            for k, bound in enumerate((anchor + fp_error, grad_fa - grad_ga, grad_fb - grad_gb)):
                coefficients[begin:end, k, 0] = bound.lo
                coefficients[begin:end, k, 1] = bound.hi
        return cls(model.digest, _immutable(edges), _immutable(centers),
                   _immutable(coefficients.reshape(bins, bins, 3, 2)), time.perf_counter()-start)

    def check_contract(self, model, tables):
        if self.model_hash != model.digest or self.model_hash != tables.model_hash:
            raise ValueError("Anchor table/model fingerprint mismatch")
        if self.edges[0] > tables.edges[0] or self.edges[-1] < tables.edges[-1]:
            raise ValueError("Anchor table does not cover the certified envelope")

    def signed_errors(self, left, right):
        """Online enclosure for each FP flux error using table reads and interval arithmetic."""
        left = np.asarray(left, dtype=np.float64)
        right = np.asarray(right, dtype=np.float64)
        if left.shape != right.shape:
            raise ValueError("Face input shapes must agree")
        for x in (left, right):
            if not np.all(np.isfinite(x)) or np.any(x < self.edges[0]) or np.any(x > self.edges[-1]):
                raise ValueError("Face outside the anchor envelope")
        i = np.clip(np.searchsorted(self.edges, left, side="right")-1, 0, self.bins-1)
        j = np.clip(np.searchsorted(self.edges, right, side="right")-1, 0, self.bins-1)
        a = VI.point(left) - VI.point(self.centers[i])
        b = VI.point(right) - VI.point(self.centers[j])
        record = self.coefficients[i, j]
        anchor = VI(record[..., 0, 0], record[..., 0, 1])
        ga = VI(record[..., 1, 0], record[..., 1, 1])
        gb = VI(record[..., 2, 0], record[..., 2, 1])
        return anchor + ga*a + gb*b
