"""Strictly conservative local residual projection for the middle-cost action."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from physics.discrete_residual import (
    cell_widths,
    physical_grid,
    shock_aware_residual_loss,
    shock_sensor,
)


class InfeasibleProjectionError(RuntimeError):
    """Raised when positivity and conservation cannot be satisfied together."""


@dataclass
class ResidualProjectionConfig:
    steps: int = 4
    step_size: float = 2e-2
    trust_weight: float = 5.0
    positivity_weight: float = 10.0
    shock_beta: float = 2.0
    shock_quantile: float = 0.75
    dilation_radius: int = 3
    max_update: float = 0.25
    positivity_floor: float = 0.0
    local_only: bool = True


def _dilated_shock_mask(sensor: torch.Tensor, quantile: float, radius: int) -> torch.Tensor:
    threshold = torch.quantile(sensor.detach(), float(quantile), dim=1, keepdim=True)
    mask = (sensor >= threshold).to(sensor.dtype).unsqueeze(1)
    if radius > 0:
        kernel = 2 * int(radius) + 1
        mask = F.max_pool1d(mask, kernel_size=kernel, stride=1, padding=radius)
    return mask.squeeze(1).clamp(0.0, 1.0)


def _weighted_positive_projection(
    values: torch.Tensor,
    weights: torch.Tensor,
    target_mass: torch.Tensor,
    iterations: int = 64,
) -> torch.Tensor:
    """Project onto x>=0 and sum(weights*x)=target_mass by KKT bisection."""
    ratio = values / weights.clamp_min(1e-12)
    weighted_square_sum = weights.square().sum(dim=1, keepdim=True).clamp_min(1e-12)
    low = torch.minimum(ratio.min(dim=1, keepdim=True).values, torch.zeros_like(target_mass))
    low = low - target_mass.abs() / weighted_square_sum - 1.0
    high = torch.maximum(ratio.max(dim=1, keepdim=True).values, torch.zeros_like(target_mass)) + 1.0
    for _ in range(int(iterations)):
        mid = 0.5 * (low + high)
        candidate = torch.clamp(values - mid * weights, min=0.0)
        mass = (candidate * weights).sum(dim=1, keepdim=True)
        # Mass is monotone decreasing in lambda.
        low = torch.where(mass > target_mass, mid, low)
        high = torch.where(mass > target_mass, high, mid)
    result = torch.clamp(values - high * weights, min=0.0)
    # One final conservative rescale removes the residual bisection error.
    result_mass = (result * weights).sum(dim=1, keepdim=True).clamp_min(1e-12)
    scale = torch.where(target_mass > 0, target_mass / result_mass, torch.zeros_like(target_mass))
    return result * scale


def _positive_mass_reference(
    base: torch.Tensor,
    weights: torch.Tensor,
    floor: float,
) -> torch.Tensor:
    shifted = base - float(floor)
    target_mass = (shifted * weights).sum(dim=1, keepdim=True)
    if torch.any(target_mass < -1e-8):
        raise InfeasibleProjectionError(
            "Advice has insufficient conserved mass for the requested positivity floor"
        )
    target_mass = target_mass.clamp_min(0.0)
    projected = _weighted_positive_projection(shifted, weights, target_mass)
    return projected + float(floor)


def _zero_mass_masked_update(
    raw_delta: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    weighted_mask = weights * mask
    denominator = weighted_mask.sum(dim=1, keepdim=True).clamp_min(1e-12)
    mean = (weighted_mask * raw_delta).sum(dim=1, keepdim=True) / denominator
    return mask * (raw_delta - mean)


def _bound_update(update: torch.Tensor, max_update: float) -> torch.Tensor:
    peak = update.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    scale = (float(max_update) / peak).clamp(max=1.0)
    return update * scale


def _positivity_preserving_scale(
    reference: torch.Tensor,
    update: torch.Tensor,
    floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    negative = update < 0
    ratios = torch.where(
        negative,
        (reference - float(floor)).clamp_min(0.0) / (-update).clamp_min(1e-12),
        torch.full_like(update, float("inf")),
    )
    alpha = ratios.min(dim=1, keepdim=True).values.clamp(max=1.0)
    alpha = torch.where(torch.isfinite(alpha), 0.999999 * alpha, torch.ones_like(alpha))
    return reference + alpha * update, alpha


def project_prediction(
    u_pred: torch.Tensor,
    u_last: torch.Tensor,
    x_normalized: torch.Tensor,
    params: Mapping[str, Any],
    config: ResidualProjectionConfig | None = None,
):
    """Reduce residual while preserving the physical cell-integrated mass."""
    cfg = config or ResidualProjectionConfig()
    with torch.enable_grad():
        original = u_pred.detach()
        x = x_normalized.detach()
        x_physical = physical_grid(x, params.get("L_mm", 20.0))
        weights = cell_widths(x_physical).detach()
        base = _positive_mass_reference(
            original, weights, cfg.positivity_floor
        ).detach()
        last = u_last.detach()
        sensor = shock_sensor(base, x, params)
        mask = (
            _dilated_shock_mask(sensor, cfg.shock_quantile, cfg.dilation_radius)
            if cfg.local_only else torch.ones_like(base)
        )
        raw_delta = torch.zeros_like(base, requires_grad=True)
        with torch.no_grad():
            initial_loss, _ = shock_aware_residual_loss(
                base, last, x, params, shock_beta=cfg.shock_beta
            )

        for _ in range(max(0, int(cfg.steps))):
            update = _bound_update(
                _zero_mass_masked_update(raw_delta, mask, weights), cfg.max_update
            )
            candidate = base + update
            residual_loss, _ = shock_aware_residual_loss(
                candidate, last, x, params, shock_beta=cfg.shock_beta
            )
            trust = (weights * update.square()).sum() / weights.sum().clamp_min(1e-12)
            positivity = torch.relu(float(cfg.positivity_floor) - candidate).square().mean()
            objective = (
                residual_loss
                + float(cfg.trust_weight) * trust
                + float(cfg.positivity_weight) * positivity
            )
            grad = torch.autograd.grad(objective, raw_delta, create_graph=False)[0]
            with torch.no_grad():
                raw_delta -= float(cfg.step_size) * grad
            raw_delta.requires_grad_(True)

        update = _bound_update(
            _zero_mass_masked_update(raw_delta.detach(), mask, weights), cfg.max_update
        )
        corrected, alpha = _positivity_preserving_scale(
            base, update, cfg.positivity_floor
        )
        with torch.no_grad():
            final_loss, _ = shock_aware_residual_loss(
                corrected, last, x, params, shock_beta=cfg.shock_beta
            )

    mass_before = (original * weights).sum(dim=1)
    mass_after = (corrected * weights).sum(dim=1)
    return corrected, {
        "initial_residual_loss": float(initial_loss.item()),
        "final_residual_loss": float(final_loss.item()),
        "corrected_fraction": float(mask.mean().item()),
        "max_mass_drift": float((mass_after - mass_before).abs().max().item()),
        "minimum_value": float(corrected.min().item()),
        "mean_positivity_scale": float(alpha.mean().item()),
    }
