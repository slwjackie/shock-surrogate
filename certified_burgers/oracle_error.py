"""Oracle quantities used for diagnostics, not deployable verification."""
from __future__ import annotations
import numpy as np

def l1_error(a, b, dx: float):
    aa = np.asarray(a); bb = np.asarray(b)
    if aa.shape != bb.shape:
        raise ValueError("a and b must have identical shapes.")
    return float(dx) * np.sum(np.abs(aa - bb), axis=-1)

def oracle_gate(prediction, reference, dx: float, threshold: float):
    eta = l1_error(prediction, reference, dx)
    accept = eta <= float(threshold)
    hybrid = np.where(np.expand_dims(accept, -1), prediction, reference)
    return hybrid, accept, eta
