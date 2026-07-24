import numpy as np

from calibration.residual_calibrator import COMPONENT_NAMES, EmpiricalRiskCalibrator, ParameterOODScorer


def test_calibration_json_roundtrip(tmp_path):
    values = np.linspace(0.0, 1.0, 64)
    groups = [str(i % 3) for i in range(64)]
    calibrator = EmpiricalRiskCalibrator(q_low=0.7, q_high=0.9)
    calibrator.ood_scorer = ParameterOODScorer.fit(
        np.tile(np.array([[0.002, 1.5, 6.0, 0.0, 0.0, 0.01, 20.0]]), (8, 1))
    )
    calibrator.fit(
        {name: values + index * 0.01 for index, name in enumerate(COMPONENT_NAMES)},
        groups=groups,
        min_group_size=10,
    )
    path = tmp_path / "calibration.json"
    calibrator.save(path)
    loaded = EmpiricalRiskCalibrator.load(path)
    assert loaded.global_thresholds == calibrator.global_thresholds
    assert loaded.group_thresholds == calibrator.group_thresholds
    assert np.allclose(loaded.ood_scorer.mean, calibrator.ood_scorer.mean)
    sample = {name: 0.3 for name in COMPONENT_NAMES}
    assert np.isclose(
        loaded.score_one(sample),
        calibrator.score_one(sample),
        rtol=0.0,
        atol=1e-14,
    )
