"""Differentiable conservative residuals and risk diagnostics on grid fields."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F


def as_batch_column(value: Any, like: torch.Tensor, default: float = 0.0) -> torch.Tensor:
    if value is None:
        value = default
    if isinstance(value, torch.Tensor):
        out = value.to(device=like.device, dtype=like.dtype)
    else:
        out = torch.as_tensor(value, device=like.device, dtype=like.dtype)
    if out.ndim == 0:
        out = out.view(1, 1).expand(like.shape[0], 1)
    elif out.ndim == 1:
        out = out[:, None]
    else:
        out = out.reshape(out.shape[0], -1)[:, :1]
    if out.shape[0] == 1 and like.shape[0] > 1:
        out = out.expand(like.shape[0], 1)
    return out


def forcing_temperature(x_normalized, dTdx, b_quad=None):
    dtdx = as_batch_column(dTdx, x_normalized)
    bq = as_batch_column(b_quad, x_normalized) if b_quad is not None else 0.0
    return 1.0 + 0.35 * dtdx * x_normalized + 0.40 * bq * x_normalized.square()


def physical_grid(x_normalized, L_mm):
    return x_normalized * as_batch_column(L_mm, x_normalized, default=20.0)


def cell_widths(x_physical: torch.Tensor) -> torch.Tensor:
    """Finite-volume cell widths reconstructed from monotone cell centers."""
    if x_physical.shape[1] < 2:
        return torch.ones_like(x_physical)
    edges = torch.empty(
        x_physical.shape[0], x_physical.shape[1] + 1,
        device=x_physical.device, dtype=x_physical.dtype,
    )
    edges[:, 1:-1] = 0.5 * (x_physical[:, :-1] + x_physical[:, 1:])
    edges[:, 0] = x_physical[:, 0] - 0.5 * (x_physical[:, 1] - x_physical[:, 0])
    edges[:, -1] = x_physical[:, -1] + 0.5 * (x_physical[:, -1] - x_physical[:, -2])
    widths = edges[:, 1:] - edges[:, :-1]
    if torch.any(widths <= 0):
        raise ValueError("Physical coordinates must be strictly increasing")
    return widths


def mean_dx(x_physical):
    return cell_widths(x_physical).mean(dim=1, keepdim=True).clamp_min(1e-12)


def rusanov_flux_divergence(u, x_physical):
    if u.shape[1] < 2:
        return torch.zeros_like(u)
    widths = cell_widths(x_physical)
    u_left, u_right = u[:, :-1], u[:, 1:]
    f_left, f_right = 0.5 * u_left.square(), 0.5 * u_right.square()
    wave_speed = torch.maximum(u_left.abs(), u_right.abs())
    interior_faces = 0.5 * (f_left + f_right) - 0.5 * wave_speed * (u_right - u_left)
    left_face = 0.5 * u[:, :1].square()
    right_face = 0.5 * u[:, -1:].square()
    faces = torch.cat([left_face, interior_faces, right_face], dim=1)
    return (faces[:, 1:] - faces[:, :-1]) / widths


def laplacian_neumann(u, x_physical):
    if u.shape[1] < 2:
        return torch.zeros_like(u)
    # Nonuniform centered second derivative. Replicated boundary states impose a
    # zero-normal-gradient closure at the two outer faces.
    left_x = torch.cat([x_physical[:, :1] - (x_physical[:, 1:2] - x_physical[:, :1]), x_physical[:, :-1]], dim=1)
    right_x = torch.cat([x_physical[:, 1:], x_physical[:, -1:] + (x_physical[:, -1:] - x_physical[:, -2:-1])], dim=1)
    left_u = torch.cat([u[:, :1], u[:, :-1]], dim=1)
    right_u = torch.cat([u[:, 1:], u[:, -1:]], dim=1)
    h_left = (x_physical - left_x).clamp_min(1e-12)
    h_right = (right_x - x_physical).clamp_min(1e-12)
    slope_right = (right_u - u) / h_right
    slope_left = (u - left_u) / h_left
    return 2.0 * (slope_right - slope_left) / (h_left + h_right)


def centered_abs_gradient(u, x_physical):
    if u.shape[1] < 2:
        return torch.zeros_like(u)
    left_x = torch.cat([x_physical[:, :1] - (x_physical[:, 1:2] - x_physical[:, :1]), x_physical[:, :-1]], dim=1)
    right_x = torch.cat([x_physical[:, 1:], x_physical[:, -1:] + (x_physical[:, -1:] - x_physical[:, -2:-1])], dim=1)
    left_u = torch.cat([u[:, :1], u[:, :-1]], dim=1)
    right_u = torch.cat([u[:, 1:], u[:, -1:]], dim=1)
    return ((right_u - left_u) / (right_x - left_x).clamp_min(1e-12)).abs()


def total_variation(u):
    if u.shape[1] < 2:
        return torch.zeros(u.shape[0], device=u.device, dtype=u.dtype)
    return (u[:, 1:] - u[:, :-1]).abs().mean(dim=1)


def burgers_reaction_residual_fd(u_pred, u_last, x_normalized, params: Mapping[str, Any], interior: int = 2):
    dt = as_batch_column(params.get("dt"), u_pred, default=1.0)
    nu = as_batch_column(params.get("nu"), u_pred)
    k = as_batch_column(params.get("k"), u_pred)
    activation = as_batch_column(params.get("E"), u_pred)
    x_physical = physical_grid(x_normalized, params.get("L_mm", 20.0))
    u_t = (u_pred - u_last) / (dt + 1e-12)
    convection = rusanov_flux_divergence(u_pred, x_physical)
    diffusion = nu * laplacian_neumann(u_pred, x_physical)
    temperature = forcing_temperature(x_normalized, params.get("dTdx", 0.0), params.get("b_quad", 0.0))
    reaction = k * (1.0 - u_pred) * torch.exp(-activation / torch.clamp(temperature, min=1e-6))
    residual = u_t + convection - diffusion - reaction
    if interior > 0 and residual.shape[1] > 2 * interior:
        residual = residual[:, interior:-interior]
    return residual


def shock_aware_residual_loss(u_pred, u_last, x_normalized, params, shock_beta=2.0, eps=1e-3, interior=2):
    residual = burgers_reaction_residual_fd(u_pred, u_last, x_normalized, params, interior=interior)
    gradient = centered_abs_gradient(u_pred, physical_grid(x_normalized, params.get("L_mm", 20.0)))
    if interior > 0 and gradient.shape[1] > 2 * interior:
        gradient = gradient[:, interior:-interior]
    normalized_gradient = gradient / (gradient.mean(dim=1, keepdim=True).detach() + 1e-12)
    weight = (1.0 + float(shock_beta) * normalized_gradient.detach()).clamp(max=1.0 + 4.0 * float(shock_beta))
    weight = weight / (weight.mean(dim=1, keepdim=True).detach() + 1e-12)
    charbonnier = torch.sqrt(residual.square() + float(eps) ** 2)
    return (weight * charbonnier).mean(), residual


def shock_sensor(u, x_normalized, params):
    gradient = centered_abs_gradient(u, physical_grid(x_normalized, params.get("L_mm", 20.0)))
    scale = gradient.mean(dim=1, keepdim=True).detach().clamp_min(1e-12)
    return gradient / scale


def risk_components_torch(u_pred, u_last, x_normalized, params, coeff_ood=None, uncertainty=None):
    residual = burgers_reaction_residual_fd(u_pred, u_last, x_normalized, params)
    residual_score = residual.abs().mean(dim=1)
    tv_pred, tv_last = total_variation(u_pred), total_variation(u_last)
    tv_growth = (tv_pred - tv_last).abs() / (tv_last.abs() + 1e-6)
    x_physical = physical_grid(x_normalized, params.get("L_mm", 20.0))
    grad_pred = centered_abs_gradient(u_pred, x_physical)
    grad_last = centered_abs_gradient(u_last, x_physical)
    shock_shift = (grad_pred.argmax(dim=1).float() - grad_last.argmax(dim=1).float()).abs() / max(u_pred.shape[1] - 1, 1)
    coeff = as_batch_column(coeff_ood, u_pred).squeeze(1) if coeff_ood is not None else torch.zeros_like(residual_score)
    unc = as_batch_column(uncertainty, u_pred).squeeze(1) if uncertainty is not None else torch.zeros_like(residual_score)
    return {"residual": residual_score, "tv_growth": tv_growth, "shock_shift": shock_shift, "coeff_ood": coeff, "uncertainty": unc}
