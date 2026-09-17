"""Long-rollout evaluation for pure surrogate and RCCP-controlled advice."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from calibration.residual_calibrator import EmpiricalRiskCalibrator
from physics.discrete_residual import centered_abs_gradient, physical_grid, risk_components_torch
from physics.residual_projection import InfeasibleProjectionError, ResidualProjectionConfig, project_prediction
from policies.clamp_policy import ClampAction, ResidualCalibratedClampPolicy
from solvers.weno_adapter import WENOSolverAdapter
from uncertainty.predictive import predict_distribution


@dataclass
class RolloutCosts:
    surrogate: float = 1.0
    correction: float = 4.0
    fallback: float = 50.0


def scalar_params_to_tensors(params, device, dtype):
    return {
        key: torch.tensor([float(value)], device=device, dtype=dtype)
        for key, value in params.items()
    }


def _raw_components(
    prediction,
    last,
    x,
    params_t,
    calibrator: EmpiricalRiskCalibrator | None,
    uncertainty: torch.Tensor | None = None,
):
    coeff_ood = 0.0
    if calibrator is not None and calibrator.ood_scorer is not None:
        mapping = {key: value.detach().cpu().numpy() for key, value in params_t.items()}
        coeff_ood = float(calibrator.ood_scorer.score_mapping(mapping)[0])
    components = risk_components_torch(
        prediction,
        last,
        x,
        params_t,
        coeff_ood=torch.tensor(
            [coeff_ood], device=prediction.device, dtype=prediction.dtype
        ),
        uncertainty=uncertainty,
    )
    return {name: float(value[0].detach().item()) for name, value in components.items()}


def rollout_case(
    model,
    trajectory: np.ndarray,
    x_array: np.ndarray,
    params: Mapping[str, float],
    history: int,
    device: torch.device,
    policy: ResidualCalibratedClampPolicy | None = None,
    solver: WENOSolverAdapter | None = None,
    projection_config: ResidualProjectionConfig | None = None,
    costs: RolloutCosts | None = None,
    mc_samples: int = 1,
):
    costs = costs or RolloutCosts()
    solver = solver or WENOSolverAdapter()
    truth = torch.as_tensor(trajectory, device=device, dtype=torch.float32)
    x = torch.as_tensor(x_array, device=device, dtype=torch.float32).unsqueeze(0)
    params_t = scalar_params_to_tensors(params, device, truth.dtype)
    current_history = truth[:history].transpose(0, 1).unsqueeze(0).clone()
    predictions = [truth[i].detach().cpu() for i in range(history)]
    action_counts = Counter()
    scores: list[float] = []
    selected_residuals: list[float] = []
    uncertainty_values: list[float] = []
    mass_drifts: list[float] = []
    total_cost = 0.0
    weno_substeps = 0

    for _time_index in range(history, truth.shape[0]):
        last = current_history[:, :, -1]
        distribution = predict_distribution(
            model,
            x,
            current_history,
            params_t,
            mc_samples=mc_samples,
        )
        advice = distribution.mean
        logits = distribution.logits
        uncertainty_values.append(float(distribution.uncertainty[0].item()))
        # One normalized surrogate cost is charged per stochastic forward pass.
        total_cost += costs.surrogate * distribution.samples

        chosen = advice
        action = ClampAction.ACCEPT
        if policy is not None:
            raw = _raw_components(
                advice,
                last,
                x,
                params_t,
                policy.calibrator,
                uncertainty=distribution.uncertainty,
            )
            group = None
            group_probabilities = None
            if logits is not None:
                probabilities = torch.softmax(logits, dim=1)[0]
                group_probabilities = probabilities.detach().cpu().numpy()
                group = int(probabilities.argmax().item())
            decision = policy.decide(
                raw,
                group=group,
                group_probabilities=group_probabilities,
            )
            action = decision.action
            scores.append(decision.score)
            if action == ClampAction.CORRECT:
                try:
                    chosen, projection_info = project_prediction(
                        advice, last, x, params_t, config=projection_config
                    )
                    mass_drifts.append(float(projection_info["max_mass_drift"]))
                    total_cost += costs.correction
                except InfeasibleProjectionError:
                    # Positivity and conservation cannot both be met; escalate to
                    # the trusted numerical fallback instead of violating either.
                    action = ClampAction.FALLBACK
                    chosen, steps = solver.advance(last, x, params_t)
                    weno_substeps += sum(steps)
                    total_cost += costs.fallback
            elif action == ClampAction.FALLBACK:
                chosen, steps = solver.advance(last, x, params_t)
                weno_substeps += sum(steps)
                total_cost += costs.fallback

        action_counts[action.value] += 1
        selected_raw = _raw_components(
            chosen,
            last,
            x,
            params_t,
            policy.calibrator if policy is not None else None,
            uncertainty=distribution.uncertainty,
        )
        selected_residuals.append(selected_raw["residual"])
        predictions.append(chosen[0].detach().cpu())
        current_history = torch.cat(
            [current_history[:, :, 1:], chosen.unsqueeze(-1)], dim=-1
        )

    predicted = torch.stack(predictions).to(device)
    evaluated_pred, evaluated_truth = predicted[history:], truth[history:]
    diff = evaluated_pred - evaluated_truth
    x_batch = x.expand(evaluated_pred.shape[0], -1)
    L_values = torch.full(
        (evaluated_pred.shape[0],), float(params.get("L_mm", 20.0)), device=device
    )
    x_physical = physical_grid(x_batch, L_values)
    pred_gradient = centered_abs_gradient(evaluated_pred, x_physical)
    truth_gradient = centered_abs_gradient(evaluated_truth, x_physical)
    pred_peak, truth_peak = pred_gradient.max(dim=1).values, truth_gradient.max(dim=1).values
    pred_shock, truth_shock = pred_gradient.argmax(dim=1).float(), truth_gradient.argmax(dim=1).float()
    n_steps = max(truth.shape[0] - history, 1)
    return {
        "mse": float(diff.square().mean().item()),
        "rmse": float(diff.square().mean().sqrt().item()),
        "mae": float(diff.abs().mean().item()),
        "final_mse": float(diff[-1].square().mean().item()),
        "peak_gradient_error": float((pred_peak - truth_peak).abs().mean().item()),
        "shock_position_error": float(
            ((pred_shock - truth_shock).abs() / max(truth.shape[1] - 1, 1)).mean().item()
        ),
        "mean_selected_residual": float(np.mean(selected_residuals)),
        "mean_risk_score": float(np.mean(scores)) if scores else 0.0,
        "mean_predictive_uncertainty": float(np.mean(uncertainty_values)),
        "max_projection_mass_drift": float(max(mass_drifts)) if mass_drifts else 0.0,
        "normalized_cost_per_step": float(total_cost / n_steps),
        "weno_substeps": int(weno_substeps),
        "actions": dict(action_counts),
        "n_rollout_steps": int(n_steps),
        "mc_samples": int(mc_samples),
        "cost_model": asdict(costs),
    }


def aggregate_case_metrics(cases):
    if not cases:
        return {}
    scalar_keys = [
        "mse", "rmse", "mae", "final_mse", "peak_gradient_error",
        "shock_position_error", "mean_selected_residual", "mean_risk_score",
        "mean_predictive_uncertainty", "max_projection_mass_drift",
        "normalized_cost_per_step", "weno_substeps",
    ]
    output = {}
    for key in scalar_keys:
        values = np.asarray([case[key] for case in cases], dtype=np.float64)
        output[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    action_totals = Counter()
    total_steps = 0
    for case in cases:
        action_totals.update(case["actions"])
        total_steps += int(case["n_rollout_steps"])
    output["actions"] = dict(action_totals)
    output["action_rates"] = {
        name: count / max(total_steps, 1) for name, count in action_totals.items()
    }
    output["n_cases"] = len(cases)
    return output
