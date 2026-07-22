"""Local residual projection used as the middle-cost correction action."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from collections.abc import Mapping

import torch
import torch.nn.functional as F

from physics.discrete_residual import shock_aware_residual_loss, shock_sensor


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
    local_only: bool = True


def _dilated_shock_mask(sensor, quantile, radius):
    threshold = torch.quantile(sensor.detach(), float(quantile), dim=1, keepdim=True)
    mask = (sensor >= threshold).to(sensor.dtype).unsqueeze(1)
    if radius > 0:
        kernel = 2 * int(radius) + 1
        mask = F.max_pool1d(mask, kernel_size=kernel, stride=1, padding=radius)
    return mask.squeeze(1).clamp(0.0, 1.0)


def project_prediction(u_pred, u_last, x_normalized, params: Mapping[str, Any], config=None):
    cfg = config or ResidualProjectionConfig()
    with torch.enable_grad():
        base, last, x = u_pred.detach(), u_last.detach(), x_normalized.detach()
        sensor = shock_sensor(base, x, params)
        mask = _dilated_shock_mask(sensor, cfg.shock_quantile, cfg.dilation_radius) if cfg.local_only else torch.ones_like(base)
        delta = torch.zeros_like(base, requires_grad=True)
        with torch.no_grad():
            initial_loss, _ = shock_aware_residual_loss(base, last, x, params, shock_beta=cfg.shock_beta)
        for _ in range(max(0, int(cfg.steps))):
            candidate = base + mask * delta
            residual_loss, _ = shock_aware_residual_loss(candidate, last, x, params, shock_beta=cfg.shock_beta)
            trust = (mask * delta).square().mean()
            positivity = torch.relu(-candidate).square().mean()
            objective = residual_loss + float(cfg.trust_weight) * trust + float(cfg.positivity_weight) * positivity
            grad = torch.autograd.grad(objective, delta, create_graph=False)[0]
            with torch.no_grad():
                delta -= float(cfg.step_size) * grad
                delta.clamp_(-float(cfg.max_update), float(cfg.max_update))
            delta.requires_grad_(True)
        corrected = (base + mask * delta.detach()).clamp_min(0.0)
        with torch.no_grad():
            final_loss, _ = shock_aware_residual_loss(corrected, last, x, params, shock_beta=cfg.shock_beta)
    return corrected, {
        "initial_residual_loss": float(initial_loss.item()),
        "final_residual_loss": float(final_loss.item()),
        "corrected_fraction": float(mask.mean().item()),
    }
