"""Cheap conservation diagnostic; a proxy, not by itself a certificate."""
from __future__ import annotations
import numpy as np

def conservation_defect(current, prediction, dx: float):
    u0=np.asarray(current); u1=np.asarray(prediction)
    if u0.shape!=u1.shape: raise ValueError("current and prediction must have identical shapes.")
    per_channel=float(dx)*np.abs(np.sum(u1-u0,axis=-1))
    return np.sum(per_channel,axis=-1) if per_channel.ndim>1 else per_channel
