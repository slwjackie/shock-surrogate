"""Controlled corruptions that expose blind spots in cheap verifier scores."""
from __future__ import annotations

import numpy as np

from .oracle_error import l1_error
from .verifiers import conservation_defect, weak_residual_score


def verifier_counterexample_candidates(reference: np.ndarray) -> dict[str, np.ndarray]:
    ref = np.asarray(reference, dtype=np.float64)
    n = ref.shape[-1]

    shifted = np.roll(ref, max(1, n // 64), axis=-1)
    smoothed = 0.25 * np.roll(ref, 1, axis=-1) + 0.5 * ref + 0.25 * np.roll(ref, -1, axis=-1)

    oscillatory = ref.copy()
    gradient = np.abs(ref - np.roll(ref, 1, axis=-1))
    center = int(np.argmax(gradient))
    width = min(8, n - (n % 2))
    indices = (center + np.arange(width) - width // 2) % n
    amplitude = max(0.02, 0.05 * float(np.ptp(ref)))
    oscillatory[..., indices] += amplitude * np.where(np.arange(width) % 2 == 0, 1.0, -1.0)

    scale = max(0.02, 0.05 * max(1.0, float(np.max(np.abs(ref)))))
    biased = ref + scale
    return {
        "mass_preserving_shock_shift": shifted,
        "mass_preserving_over_smoothing": smoothed,
        "mass_preserving_local_oscillation": oscillatory,
        "global_bias": biased,
    }


def audit_verifier_counterexamples(
    current: np.ndarray,
    reference: np.ndarray,
    *,
    dx: float,
    elapsed_time: float,
    thresholds: dict[str, float] | None = None,
    ensemble_members: int = 8,
) -> dict:
    thresholds = thresholds or {}
    rows = []
    candidates = verifier_counterexample_candidates(reference)
    for name, candidate in candidates.items():
        eta = float(l1_error(candidate[None], reference[None], dx)[0])
        conservation = float(conservation_defect(current[None], candidate[None], dx)[0])
        weak = float(
            weak_residual_score(
                current[None], candidate[None], dx=dx, dt=elapsed_time
            )[0]
        )
        rows.append(
            {
                "name": name,
                "oracle_eta_vs_same_grid_solver": eta,
                "conservation": conservation,
                "weak_residual": weak,
                "accepted_by_conservation_threshold": (
                    bool(conservation <= thresholds["conservation"])
                    if "conservation" in thresholds
                    else None
                ),
                "accepted_by_residual_threshold": (
                    bool(weak <= thresholds["residual"])
                    if "residual" in thresholds
                    else None
                ),
            }
        )

    shared_bias = candidates["mass_preserving_shock_shift"]
    shared_draws = np.repeat(shared_bias[None], int(ensemble_members), axis=0)
    shared_std = float(shared_draws.std(axis=0).mean())
    shared_eta = float(l1_error(shared_bias[None], reference[None], dx)[0])
    return {
        "purpose": "Controlled failure probes; they are not samples from the trained surrogate.",
        "cases": rows,
        "shared_bias_ensemble_probe": {
            "members": int(ensemble_members),
            "mean_member_spread": shared_std,
            "oracle_eta_vs_same_grid_solver": shared_eta,
            "interpretation": "Identically biased members can have zero spread and nonzero error.",
        },
    }
