"""Residual-Calibrated Clamp Policy (RCCP)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from collections.abc import Mapping

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


class ResidualCalibratedClampPolicy:
    """Map calibrated risk to accept/correct/fallback actions.

    ``threshold_scale < 1`` is conservative; ``threshold_scale > 1`` trusts
    advice more. A raw-tail guard catches residual or coefficient shifts that
    lie well outside validation support.
    """

    def __init__(self, calibrator, threshold_scale=1.0, hard_guard_multiplier=2.0):
        self.calibrator = calibrator
        self.threshold_scale = float(threshold_scale)
        self.hard_guard_multiplier = float(hard_guard_multiplier)

    def _scaled_thresholds(self, group):
        low, high = self.calibrator.thresholds(group)
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
        return None

    def decide(self, components: Mapping[str, float], group=None):
        score = self.calibrator.score_one(components)
        low, high = self._scaled_thresholds(group)
        guard_reason = self._hard_guard_reason(components)
        if guard_reason is not None:
            action, reason = ClampAction.FALLBACK, guard_reason
        elif score <= low:
            action, reason = ClampAction.ACCEPT, "score_below_low_threshold"
        elif score <= high:
            action, reason = ClampAction.CORRECT, "score_between_thresholds"
        else:
            action, reason = ClampAction.FALLBACK, "score_above_high_threshold"
        return ClampDecision(action, score, low, high, None if group is None else str(group), reason)
