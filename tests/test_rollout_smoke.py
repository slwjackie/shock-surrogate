import numpy as np
import torch
import torch.nn as nn

from calibration.residual_calibrator import COMPONENT_NAMES, EmpiricalRiskCalibrator
from evaluation.rollout_policy_eval import rollout_case
from policies.clamp_policy import ResidualCalibratedClampPolicy
from sim.solver_burgers_weno import simulate_case


class LastStateBackbone(nn.Module):
    def forward(self, x, history, parameters=None):
        del x, parameters
        prediction = history[:, :, -1]
        logits = torch.zeros(prediction.shape[0], 3, device=prediction.device)
        return prediction, logits


def test_end_to_end_policy_rollout_smoke():
    x, _, trajectory, _ = simulate_case(
        L_mm=20.0, Nx=32, t_end=0.02, Nt_save=7,
        CFL=0.3, nu=0.002, k=1.5, E=6.0, seed=2,
    )
    base = np.linspace(0, 100, 128)
    calibrator = EmpiricalRiskCalibrator(q_low=0.8, q_high=0.95).fit(
        {name: base.copy() for name in COMPONENT_NAMES}
    )
    policy = ResidualCalibratedClampPolicy(
        calibrator, threshold_scale=1.5, hard_guard_multiplier=0.0,
        group_strategy="global",
    )
    result = rollout_case(
        LastStateBackbone(), trajectory, x,
        {"dt": 0.02 / 6, "L_mm": 20.0, "nu": 0.002, "k": 1.5,
         "E": 6.0, "dTdx": 0.0, "b_quad": 0.0},
        history=3, device=torch.device("cpu"), policy=policy, mc_samples=1,
    )
    assert result["n_rollout_steps"] == 4
    assert np.isfinite(result["mse"])
    assert sum(result["actions"].values()) == 4
