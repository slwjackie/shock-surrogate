#!/usr/bin/env python3
"""Backbone factory and legacy continuous-coordinate PINN residual.

The default training path uses ``physics.discrete_residual`` because the
surrogates predict grid fields. The autograd residual is retained for ablation
and backward compatibility.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from backbones.registry import build_backbone


def make_model(arch: str, n_classes: int, causal: bool = True, history: int = 5, **kwargs) -> nn.Module:
    return build_backbone(arch=arch, n_classes=n_classes, causal=causal, history=history, **kwargs)


def forcing_T(x, dTdx, b_quad=None):
    if dTdx.dim() == 1:
        dTdx = dTdx[:, None]
    temperature = 1.0 + 0.35 * dTdx * x
    if b_quad is not None:
        if b_quad.dim() == 1:
            b_quad = b_quad[:, None]
        temperature = temperature + 0.40 * b_quad * x.square()
    return temperature


def _grad(y, x, create_graph):
    return torch.autograd.grad(
        outputs=y, inputs=x, grad_outputs=torch.ones_like(y),
        create_graph=create_graph, retain_graph=True, only_inputs=True,
    )[0]


def physics_residual_hybrid(u_pred, u_last, x, dt, nu, k, E, dTdx, b_quad=None):
    if isinstance(dt, (float, int)):
        dt_t = torch.full((u_pred.shape[0], 1), float(dt), device=u_pred.device, dtype=u_pred.dtype)
    else:
        dt_t = dt.to(device=u_pred.device, dtype=u_pred.dtype)
        if dt_t.dim() == 1:
            dt_t = dt_t[:, None]
    u_t = (u_pred - u_last) / (dt_t + 1e-12)
    u_x = _grad(u_pred, x, create_graph=True)
    u_xx = _grad(u_x, x, create_graph=True)
    if nu.dim() == 1:
        nu = nu[:, None]
    if k.dim() == 1:
        k = k[:, None]
    if E.dim() == 1:
        E = E[:, None]
    temperature = forcing_T(x, dTdx, b_quad=b_quad)
    reaction = k * (1.0 - u_pred) * torch.exp(-E / torch.clamp(temperature, min=1e-6))
    return u_t + u_pred * u_x - nu * u_xx - reaction
