"""High-resolution Godunov reference used only as a discretization-error proxy."""
from __future__ import annotations

import numpy as np

from .godunov import advance


def prolong_piecewise_constant(state: np.ndarray, factor: int) -> np.ndarray:
    if int(factor) < 1:
        raise ValueError("refinement factor must be at least one.")
    return np.repeat(np.asarray(state, dtype=np.float64), int(factor), axis=-1)


def restrict_cell_average(fine_state: np.ndarray, factor: int) -> np.ndarray:
    fine = np.asarray(fine_state, dtype=np.float64)
    factor = int(factor)
    if factor < 1 or fine.shape[-1] % factor:
        raise ValueError("fine-grid size must be divisible by the refinement factor.")
    shape = fine.shape[:-1] + (fine.shape[-1] // factor, factor)
    return fine.reshape(shape).mean(axis=-1)


def fine_godunov_reference(
    state: np.ndarray,
    *,
    dx: float,
    dt: float,
    coarse_steps: int,
    factor: int = 4,
    boundary: str = "periodic",
) -> np.ndarray:
    """Advance a piecewise-constant prolongation for the same physical time."""

    factor = int(factor)
    fine = prolong_piecewise_constant(state, factor)
    fine, _ = advance(
        fine,
        float(dx) / factor,
        int(coarse_steps) * factor,
        dt=float(dt) / factor,
        boundary=boundary,
    )
    return restrict_cell_average(fine, factor)
