"""Reproducible, role-separated data splits with exact-overlap auditing."""
from __future__ import annotations

import hashlib

import numpy as np

from .initial_conditions import sample_states


def _row_digest(row: np.ndarray) -> str:
    canonical = np.ascontiguousarray(row, dtype="<f8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _array_digest(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array, dtype="<f8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def make_data_splits(
    *,
    n_cells: int,
    train_samples: int,
    calib_samples: int,
    test_samples: int,
    rollout_cases: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict]:
    specs = {
        "train": (train_samples, seed, 1.0, False),
        "calibration": (calib_samples, seed + 10, 1.0, False),
        "test_id": (test_samples, seed + 20, 1.0, False),
        "test_ood": (test_samples, seed + 30, 1.8, True),
        "rollout_id": (rollout_cases, seed + 50, 1.0, False),
        "rollout_ood": (rollout_cases, seed + 60, 1.8, True),
    }
    splits = {
        name: sample_states(count, n_cells, seed=split_seed, max_abs=max_abs, strong_ood=ood)
        for name, (count, split_seed, max_abs, ood) in specs.items()
    }

    fingerprints = {name: {_row_digest(row) for row in values} for name, values in splits.items()}
    overlap_counts: dict[str, int] = {}
    names = list(splits)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap_counts[f"{left}__{right}"] = len(fingerprints[left] & fingerprints[right])
    if any(overlap_counts.values()):
        raise RuntimeError(f"Exact state leakage detected across data roles: {overlap_counts}")

    manifest = {
        "roles": {
            name: {
                "count": int(specs[name][0]),
                "seed": int(specs[name][1]),
                "max_abs": float(specs[name][2]),
                "strong_ood": bool(specs[name][3]),
                "sha256": _array_digest(values),
            }
            for name, values in splits.items()
        },
        "exact_cross_split_overlap_counts": overlap_counts,
        "all_exact_overlap_counts_zero": not any(overlap_counts.values()),
    }
    return splits, manifest
