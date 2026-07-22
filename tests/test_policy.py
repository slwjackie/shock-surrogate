import numpy as np

from calibration.residual_calibrator import COMPONENT_NAMES, EmpiricalRiskCalibrator
from policies.clamp_policy import ClampAction, ResidualCalibratedClampPolicy


def test_calibrated_policy_has_three_actions():
    n = 200
    base = np.linspace(0.0, 1.0, n)
    components = {name: base.copy() for name in COMPONENT_NAMES}
    calibrator = EmpiricalRiskCalibrator(q_low=0.6, q_high=0.9).fit(components)
    policy = ResidualCalibratedClampPolicy(calibrator)
    low = policy.decide({name: 0.05 for name in COMPONENT_NAMES})
    medium = policy.decide({name: 0.75 for name in COMPONENT_NAMES})
    high = policy.decide({name: 1.5 for name in COMPONENT_NAMES})
    assert low.action == ClampAction.ACCEPT
    assert medium.action in {ClampAction.CORRECT, ClampAction.FALLBACK}
    assert high.action == ClampAction.FALLBACK
