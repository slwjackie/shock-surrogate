"""Conservative one-dimensional remapping between monotone cell-center grids."""

from __future__ import annotations

import numpy as np


def _centers(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        raise ValueError("A grid requires at least two cell centers")
    if not np.all(np.isfinite(arr)) or np.any(np.diff(arr) <= 0):
        raise ValueError("Grid centers must be finite and strictly increasing")
    return arr


def cell_edges_from_centers(centers) -> np.ndarray:
    x = _centers(centers)
    edges = np.empty(x.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = x[0] - 0.5 * (x[1] - x[0])
    edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
    return edges


def is_uniform_grid(centers, rtol: float = 1e-5, atol: float = 1e-10) -> bool:
    x = _centers(centers)
    spacing = np.diff(x)
    return bool(np.allclose(spacing, spacing.mean(), rtol=rtol, atol=atol))


def uniform_centers_for_domain(source_centers, n_cells: int | None = None) -> np.ndarray:
    source = _centers(source_centers)
    n = int(n_cells or source.size)
    if n < 2:
        raise ValueError("n_cells must be at least two")
    source_edges = cell_edges_from_centers(source)
    target_edges = np.linspace(source_edges[0], source_edges[-1], n + 1)
    return 0.5 * (target_edges[:-1] + target_edges[1:])


def cell_integral(values, centers) -> float:
    values_arr = np.asarray(values, dtype=np.float64).reshape(-1)
    edges = cell_edges_from_centers(centers)
    if values_arr.size != edges.size - 1:
        raise ValueError("Values and grid size do not match")
    return float(np.sum(values_arr * np.diff(edges)))


def conservative_remap_1d(values, source_centers, target_centers) -> np.ndarray:
    """Piecewise-constant conservative remap of cell averages.

    Source and target outer cell edges must agree. The overlap integral is
    computed exactly for the piecewise-constant representation, so total mass is
    preserved to floating-point precision.
    """
    source = _centers(source_centers)
    target = _centers(target_centers)
    values_arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if values_arr.size != source.size:
        raise ValueError("Values and source grid size do not match")

    source_edges = cell_edges_from_centers(source)
    target_edges = cell_edges_from_centers(target)
    domain_scale = max(abs(source_edges[-1] - source_edges[0]), 1.0)
    tolerance = 1e-10 * domain_scale
    if not (
        abs(source_edges[0] - target_edges[0]) <= tolerance
        and abs(source_edges[-1] - target_edges[-1]) <= tolerance
    ):
        raise ValueError("Source and target grids must span the same outer cell-edge domain")

    accum = np.zeros(target.size, dtype=np.float64)
    i = 0
    j = 0
    while i < source.size and j < target.size:
        left = max(source_edges[i], target_edges[j])
        right = min(source_edges[i + 1], target_edges[j + 1])
        if right > left:
            accum[j] += values_arr[i] * (right - left)
        if source_edges[i + 1] <= target_edges[j + 1] + tolerance:
            i += 1
        else:
            j += 1

    widths = np.diff(target_edges)
    return accum / widths
