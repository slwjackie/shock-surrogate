import numpy as np

from calibration.residual_calibrator import COMPONENT_NAMES, EmpiricalRiskCalibrator
from policies.clamp_policy import ClampAction, ResidualCalibratedClampPolicy


def _calibrator():
    base = np.linspace(0.0, 1.0, 200)
    calibrator = EmpiricalRiskCalibrator(q_low=0.6, q_high=0.9).fit(
        {name: base.copy() for name in COMPONENT_NAMES}
    )
    calibrator.global_thresholds = (0.60, 0.90)
    calibrator.group_thresholds = {
        "0": (0.70, 0.95),
        "1": (0.20, 0.30),
        "2": (0.50, 0.75),
    }
    return calibrator


def test_soft_strategy_becomes_conservative_when_classifier_is_uncertain():
    policy = ResidualCalibratedClampPolicy(
        _calibrator(), group_strategy="soft_conservative", group_safety_blend=0.25,
        hard_guard_multiplier=0.0,
    )
    components = {name: 0.4 for name in COMPONENT_NAMES}
    uncertain = policy.decide(components, group=0, group_probabilities=[1/3, 1/3, 1/3])
    confident = policy.decide(components, group=0, group_probabilities=[0.99, 0.005, 0.005])
    assert uncertain.tau_low <= confident.tau_low
    assert uncertain.tau_high <= confident.tau_high
    assert uncertain.threshold_strategy == "soft_conservative"


def test_uncertainty_raw_tail_guard_forces_fallback():
    calibrator = _calibrator()
    # Give uncertainty a small but nonzero calibration scale.
    calibrator.component_quantiles["uncertainty"] = np.linspace(0.0, 0.1, 101)
    policy = ResidualCalibratedClampPolicy(
        calibrator, hard_guard_multiplier=2.0, group_strategy="global"
    )
    components = {name: 0.0 for name in COMPONENT_NAMES}
    components["uncertainty"] = 1.0
    decision = policy.decide(components)
    assert decision.action == ClampAction.FALLBACK
    assert decision.reason == "uncertainty_outside_calibration_support"
