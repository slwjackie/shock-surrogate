"""Residual-Calibrated Clamp Policy (RCCP)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

from calibration.residual_calibrator import EmpiricalRiskCalibrator


class ClampAction(str, Enum):
    ACCEPT = "accept_surrogate"
    CORRECT = "residual_correct"
    FALLBACK = "weno_fallback"


@dataclass(frozen=True)
class ClampDecision:
    action: ClampAction
    score: float
    tau_low: float
    tau_high: float
    group: str | None = None
    reason: str = "calibrated_score"
    threshold_strategy: str = "soft_conservative"


class ResidualCalibratedClampPolicy:
    """Map calibrated risk to accept/correct/fallback actions.

    Group thresholds are no longer tied by default to one hard argmax class.
    ``soft_conservative`` averages regime thresholds with class probabilities and
    blends toward the safest group as classification entropy increases.
    """

    VALID_GROUP_STRATEGIES = {
        "global",
        "predicted",
        "soft_conservative",
        "worst_case",
    }

    def __init__(
        self,
        calibrator: EmpiricalRiskCalibrator,
        threshold_scale: float = 1.0,
        hard_guard_multiplier: float = 2.0,
        group_strategy: str = "soft_conservative",
        group_safety_blend: float = 0.25,
    ):
        if group_strategy not in self.VALID_GROUP_STRATEGIES:
            raise ValueError(
                f"Unknown group_strategy={group_strategy!r}; "
                f"expected one of {sorted(self.VALID_GROUP_STRATEGIES)}"
            )
        self.calibrator = calibrator
        self.threshold_scale = float(threshold_scale)
        self.hard_guard_multiplier = float(hard_guard_multiplier)
        self.group_strategy = group_strategy
        self.group_safety_blend = float(np.clip(group_safety_blend, 0.0, 1.0))

    def _worst_case_thresholds(self) -> tuple[float, float]:
        pairs = list(self.calibrator.all_thresholds(include_global=True).values())
        return min(pair[0] for pair in pairs), min(pair[1] for pair in pairs)

    def _soft_thresholds(
        self,
        probabilities: Sequence[float] | np.ndarray | None,
    ) -> tuple[float, float]:
        global_pair = self.calibrator.thresholds()
        if probabilities is None:
            return global_pair
        probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        probs = np.clip(probs, 0.0, None)
        total = probs.sum()
        if total <= 0:
            return global_pair
        probs = probs / total

        pairs = []
        for class_index in range(len(probs)):
            pairs.append(self.calibrator.thresholds(str(class_index)))
        lows = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
        highs = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
        weighted = (float(probs @ lows), float(probs @ highs))
        worst = self._worst_case_thresholds()

        if len(probs) > 1:
            entropy = -float(np.sum(probs * np.log(probs + 1e-12))) / np.log(len(probs))
        else:
            entropy = 0.0
        conservative_weight = self.group_safety_blend + (
            1.0 - self.group_safety_blend
        ) * float(np.clip(entropy, 0.0, 1.0))
        return (
            (1.0 - conservative_weight) * weighted[0] + conservative_weight * worst[0],
            (1.0 - conservative_weight) * weighted[1] + conservative_weight * worst[1],
        )

    def _base_thresholds(
        self,
        group: str | int | None,
        group_probabilities: Sequence[float] | np.ndarray | None,
    ) -> tuple[float, float]:
        if self.group_strategy == "global":
            return self.calibrator.thresholds()
        if self.group_strategy == "predicted":
            return self.calibrator.thresholds(group)
        if self.group_strategy == "worst_case":
            return self._worst_case_thresholds()
        return self._soft_thresholds(group_probabilities)

    def _scaled_thresholds(
        self,
        group: str | int | None,
        group_probabilities: Sequence[float] | np.ndarray | None,
    ) -> tuple[float, float]:
        low, high = self._base_thresholds(group, group_probabilities)
        scale = max(self.threshold_scale, 0.0)
        low = min(max(low * scale, 0.0), 1.0)
        high = min(max(high * scale, low), 1.0)
        return low, high

    def _hard_guard_reason(self, components: Mapping[str, float]):
        if self.hard_guard_multiplier <= 0:
            return None
        residual_ref = max(self.calibrator.raw_quantile("residual", 0.99), 1e-8)
        if float(components.get("residual", 0.0)) > self.hard_guard_multiplier * residual_ref:
            return "residual_outside_calibration_support"
        coeff_ref = max(self.calibrator.raw_quantile("coeff_ood", 0.99), 0.5)
        if float(components.get("coeff_ood", 0.0)) > self.hard_guard_multiplier * coeff_ref:
            return "coefficient_ood_outside_calibration_support"
        uncertainty_ref = max(self.calibrator.raw_quantile("uncertainty", 0.99), 1e-8)
        if (
            uncertainty_ref > 1e-8
            and float(components.get("uncertainty", 0.0))
            > self.hard_guard_multiplier * uncertainty_ref
        ):
            return "uncertainty_outside_calibration_support"
        return None

    def decide(
        self,
        components: Mapping[str, float],
        group=None,
        group_probabilities: Sequence[float] | np.ndarray | None = None,
    ) -> ClampDecision:
        score = self.calibrator.score_one(components)
        low, high = self._scaled_thresholds(group, group_probabilities)
        guard_reason = self._hard_guard_reason(components)
        if guard_reason is not None:
            action, reason = ClampAction.FALLBACK, guard_reason
        elif score <= low:
            action, reason = ClampAction.ACCEPT, "score_below_low_threshold"
        elif score <= high:
            action, reason = ClampAction.CORRECT, "score_between_thresholds"
        else:
            action, reason = ClampAction.FALLBACK, "score_above_high_threshold"
        return ClampDecision(
            action=action,
            score=score,
            tau_low=low,
            tau_high=high,
            group=None if group is None else str(group),
            reason=reason,
            threshold_strategy=self.group_strategy,
        )
