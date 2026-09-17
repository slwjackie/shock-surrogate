"""Numerical validation checks for the trusted first-order Godunov scheme."""
from __future__ import annotations

import numpy as np

from .godunov import advance, godunov_step
from .initial_conditions import cell_centers, riemann_state


def periodic_total_variation(state: np.ndarray) -> np.ndarray:
    values = np.asarray(state, dtype=np.float64)
    return np.sum(np.abs(values - np.roll(values, 1, axis=-1)), axis=-1)


def _shock_location(state: np.ndarray, dx: float) -> float:
    jump = np.abs(np.diff(np.asarray(state, dtype=np.float64)))
    face = int(np.argmax(jump)) + 1
    return float(face * dx)


def godunov_validation_suite(*, n_cells: int = 256, cfl: float = 0.8, seed: int = 0) -> dict:
    if n_cells < 32:
        raise ValueError("Use at least 32 cells for the Riemann validation suite.")
    x, dx = cell_centers(n_cells)
    dt = cfl * dx

    rng = np.random.default_rng(seed)
    random_states = rng.uniform(-1.0, 1.0, size=(64, n_cells))
    updated, _ = godunov_step(random_states, dx, dt=dt)
    mass_error = np.max(np.abs(dx * np.sum(updated - random_states, axis=-1)))
    maximum_principle_violations = int(
        np.sum(
            (np.min(updated, axis=-1) < np.min(random_states, axis=-1) - 1e-12)
            | (np.max(updated, axis=-1) > np.max(random_states, axis=-1) + 1e-12)
        )
    )
    tv_before = periodic_total_variation(random_states)
    tv_after = periodic_total_variation(updated)

    shock_x0 = 0.3
    shock_initial = riemann_state(n_cells, 1.0, 0.0, location=shock_x0)
    shock_steps = max(1, int(0.12 / dt))
    shock_numerical, shock_time = advance(
        shock_initial, dx, shock_steps, dt=dt, boundary="outflow"
    )
    shock_exact_location = shock_x0 + 0.5 * shock_time
    shock_location_error = abs(_shock_location(shock_numerical, dx) - shock_exact_location)

    rare_x0 = 0.5
    rare_initial = riemann_state(n_cells, -1.0, 1.0, location=rare_x0)
    rare_steps = max(1, int(0.1 / dt))
    rare_numerical, rare_time = advance(
        rare_initial, dx, rare_steps, dt=dt, boundary="outflow"
    )
    similarity = (x - rare_x0) / rare_time
    rare_exact = np.where(similarity <= -1.0, -1.0, np.where(similarity >= 1.0, 1.0, similarity))
    rarefaction_l1_error = float(dx * np.sum(np.abs(rare_numerical - rare_exact)))

    return {
        "n_cells": int(n_cells),
        "cfl": float(cfl),
        "periodic_mass_max_abs_error": float(mass_error),
        "maximum_principle_violations": maximum_principle_violations,
        "tvd_violations": int(np.sum(tv_after > tv_before + 1e-12)),
        "max_tv_ratio": float(np.max(tv_after / np.maximum(tv_before, 1e-15))),
        "shock": {
            "elapsed_time": float(shock_time),
            "exact_speed": 0.5,
            "location_abs_error": float(shock_location_error),
            "location_error_in_cells": float(shock_location_error / dx),
        },
        "rarefaction": {
            "elapsed_time": float(rare_time),
            "l1_error_vs_exact_entropy_solution": rarefaction_l1_error,
        },
    }
