#!/usr/bin/env python3
"""
Model helpers for Hybrid Temporal+Spatial Transformer, including PINN-style residual
matched to the data generator (sim/solver_burgers_weno.py).

Solver (proxy reactive viscous Burgers):
  u_t + (0.5 u^2)_x = nu u_xx + k*(1-u)*exp(-E/T(x))

We compute:
  u_t  via FD: (u_pred - u_last)/dt
  u_x, u_xx via autograd w.r.t continuous x input

Residual used for soft physics regularization:
  r = u_t + u_pred*u_x - nu*u_xx - k*(1-u_pred)*exp(-E/T(x))

T(x) proxy matches solver:
  T = 1 + 0.35*dTdx*x + 0.40*b_quad*x^2
where x is normalized coordinate in [0,1].
"""

from __future__ import annotations
import torch
import torch.nn as nn

from models.arches_hybrid_temporal_spatial import build_model


def make_model(arch: str, n_classes: int, causal: bool = True, **kwargs) -> nn.Module:
    return build_model(arch=arch, n_classes=n_classes, causal=causal, **kwargs)


def forcing_T(x: torch.Tensor, dTdx: torch.Tensor, b_quad: torch.Tensor | None = None) -> torch.Tensor:
    """Return proxy temperature field T(x) matching solver.

    x: (B, Nx)
    dTdx: (B,) or (B,1)
    b_quad: (B,) or (B,1) optional
    returns T(x): (B, Nx)
    """
    if dTdx.dim() == 1:
        dTdx = dTdx[:, None]
    T = 1.0 + 0.35 * dTdx * x
    if b_quad is not None:
        if b_quad.dim() == 1:
            b_quad = b_quad[:, None]
        T = T + 0.40 * b_quad * (x ** 2)
    return T


def _grad(y: torch.Tensor, x: torch.Tensor, create_graph: bool) -> torch.Tensor:
    return torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=create_graph,
        retain_graph=True,
        only_inputs=True,
    )[0]


def physics_residual_hybrid(
    u_pred: torch.Tensor,
    u_last: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor | float,
    nu: torch.Tensor,
    k: torch.Tensor,
    E: torch.Tensor,
    dTdx: torch.Tensor,
    b_quad: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute residual r for the solver PDE.

    u_pred: (B, Nx)
    u_last: (B, Nx)
    x: (B, Nx) with requires_grad=True
    dt: float or tensor (B,) or (B,1)
    nu,k,E,dTdx,b_quad: (B,) or (B,1)
    """
    # dt -> (B,1)
    if isinstance(dt, (float, int)):
        dt_t = torch.tensor(float(dt), device=u_pred.device, dtype=u_pred.dtype).view(1, 1)
        dt_t = dt_t.expand(u_pred.shape[0], 1)
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

    T = forcing_T(x, dTdx, b_quad=b_quad)
    react = k * (1.0 - u_pred) * torch.exp(-E / torch.clamp(T, min=1e-6))

    return u_t + u_pred * u_x - nu * u_xx - react
