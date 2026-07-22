"""Torch-facing adapter for WENO fallback actions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from sim.solver_burgers_weno import advance_state


def _batch_values(value: Any, batch: int, default: float) -> np.ndarray:
    if value is None:
        return np.full(batch, default, dtype=np.float64)
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if arr.ndim == 0:
        return np.full(batch, float(arr), dtype=np.float64)
    arr = arr.reshape(arr.shape[0], -1)[:, 0].astype(np.float64)
    if arr.size == 1 and batch > 1:
        arr = np.repeat(arr, batch)
    return arr


class WENOSolverAdapter:
    def __init__(self, cfl: float = 0.45):
        self.cfl = float(cfl)

    def advance(self, u_last, x_normalized, params: Mapping[str, Any]):
        del x_normalized
        batch = u_last.shape[0]
        values = {
            "dt": _batch_values(params.get("dt"), batch, 1.0),
            "L_mm": _batch_values(params.get("L_mm"), batch, 20.0),
            "nu": _batch_values(params.get("nu"), batch, 0.002),
            "k": _batch_values(params.get("k"), batch, 1.5),
            "E": _batch_values(params.get("E"), batch, 6.0),
            "dTdx": _batch_values(params.get("dTdx"), batch, 0.0),
            "b_quad": _batch_values(params.get("b_quad"), batch, 0.0),
        }
        outputs, step_counts = [], []
        states = u_last.detach().cpu().numpy()
        for i in range(batch):
            state, steps = advance_state(
                states[i], dt_total=values["dt"][i], L_mm=values["L_mm"][i],
                CFL=self.cfl, nu=values["nu"][i], k=values["k"][i],
                E=values["E"][i], dTdx=values["dTdx"][i], b_quad=values["b_quad"][i],
            )
            outputs.append(state)
            step_counts.append(int(steps))
        return torch.as_tensor(np.stack(outputs), device=u_last.device, dtype=u_last.dtype), step_counts
