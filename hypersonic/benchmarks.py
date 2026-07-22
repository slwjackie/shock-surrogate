"""Initial conditions for strong-shock and hypersonic benchmarks."""

from __future__ import annotations

import math

import torch

from hypersonic.state import primitive_to_conservative


def uniform_flow_1d(nx: int = 128, rho: float = 1.0, u: float = 1.0, p: float = 1.0) -> torch.Tensor:
    shape = (1, nx)
    return primitive_to_conservative(torch.full(shape, rho), torch.full(shape, u), p=torch.full(shape, p))


def sod_shock_tube(nx: int = 256, gamma: float = 1.4) -> torch.Tensor:
    x = torch.linspace(0.0, 1.0, nx).view(1, nx)
    rho = torch.where(x < 0.5, 1.0, 0.125)
    u = torch.zeros_like(x)
    p = torch.where(x < 0.5, 1.0, 0.1)
    return primitive_to_conservative(rho, u, p=p, gamma=gamma)


def strong_blast_1d(nx: int = 512, gamma: float = 1.4) -> torch.Tensor:
    x = torch.linspace(0.0, 1.0, nx).view(1, nx)
    rho = torch.ones_like(x)
    u = torch.zeros_like(x)
    p = torch.where(x < 0.1, 1000.0, torch.where(x > 0.9, 100.0, 0.01))
    return primitive_to_conservative(rho, u, p=p, gamma=gamma)


def normal_shock_downstream(mach: float, rho1: float = 1.0, p1: float = 1.0, gamma: float = 1.4) -> tuple[float, float, float]:
    M1 = float(mach)
    g = float(gamma)
    if M1 <= 1.0:
        raise ValueError("Upstream normal Mach number must exceed one")
    a1 = math.sqrt(g * p1 / rho1)
    un1 = M1 * a1
    density_ratio = ((g + 1.0) * M1 * M1) / ((g - 1.0) * M1 * M1 + 2.0)
    pressure_ratio = 1.0 + 2.0 * g / (g + 1.0) * (M1 * M1 - 1.0)
    return rho1 * density_ratio, un1 / density_ratio, p1 * pressure_ratio


def planar_hypersonic_shock_2d(nx: int = 128, ny: int = 96, mach: float = 10.0, shock_normal_angle_deg: float = 0.0, offset: float = 0.5, gamma: float = 1.4) -> torch.Tensor:
    theta = math.radians(float(shock_normal_angle_deg))
    nxn, nyn = math.cos(theta), math.sin(theta)
    rho1, p1 = 1.0, 1.0
    a1 = math.sqrt(gamma * p1 / rho1)
    un1 = float(mach) * a1
    rho2, un2, p2 = normal_shock_downstream(mach, rho1, p1, gamma)
    y, x = torch.meshgrid(torch.linspace(0.0, 1.0, ny), torch.linspace(0.0, 1.0, nx), indexing="ij")
    upstream = nxn * x + nyn * y < float(offset)
    rho = torch.where(upstream, torch.tensor(rho1), torch.tensor(rho2)).unsqueeze(0)
    un = torch.where(upstream, torch.tensor(un1), torch.tensor(un2)).unsqueeze(0)
    u, v = un * nxn, un * nyn
    p = torch.where(upstream, torch.tensor(p1), torch.tensor(p2)).unsqueeze(0)
    return primitive_to_conservative(rho, u, v, p, gamma=gamma)


def four_quadrant_riemann_2d(nx: int = 128, ny: int = 128, gamma: float = 1.4) -> torch.Tensor:
    y, x = torch.meshgrid(torch.linspace(0.0, 1.0, ny), torch.linspace(0.0, 1.0, nx), indexing="ij")
    q1 = (x >= 0.5) & (y >= 0.5)
    q2 = (x < 0.5) & (y >= 0.5)
    q3 = (x < 0.5) & (y < 0.5)
    rho = torch.where(q1, 1.5, torch.where(q2, 0.5323, torch.where(q3, 0.138, 0.5323))).unsqueeze(0)
    u = torch.where(q1, 0.0, torch.where(q2, 1.206, torch.where(q3, 1.206, 0.0))).unsqueeze(0)
    v = torch.where(q1, 0.0, torch.where(q2, 0.0, torch.where(q3, 1.206, 1.206))).unsqueeze(0)
    p = torch.where(q1, 1.5, torch.where(q2, 0.3, torch.where(q3, 0.029, 0.3))).unsqueeze(0)
    return primitive_to_conservative(rho, u, v, p, gamma=gamma)


def uniform_flow_3d(nx: int = 16, ny: int = 12, nz: int = 8, rho: float = 1.0, u: float = 1.0, v: float = 0.0, w: float = 0.0, p: float = 1.0, gamma: float = 1.4) -> torch.Tensor:
    from hypersonic.state import primitive_to_conservative_3d
    shape = (1, nz, ny, nx)
    fill = lambda value: torch.full(shape, float(value))
    return primitive_to_conservative_3d(fill(rho), fill(u), fill(v), fill(w), fill(p), gamma=gamma)


def planar_hypersonic_shock_3d(nx: int = 32, ny: int = 20, nz: int = 16, mach: float = 8.0, normal: tuple[float, float, float] = (1.0, 0.0, 0.0), offset: float = 0.5, gamma: float = 1.4) -> torch.Tensor:
    from hypersonic.state import primitive_to_conservative_3d
    normal_t = torch.tensor(normal, dtype=torch.float32)
    normal_t = normal_t / torch.linalg.vector_norm(normal_t).clamp_min(1e-12)
    rho1, p1 = 1.0, 1.0
    a1 = math.sqrt(gamma * p1 / rho1)
    un1 = float(mach) * a1
    rho2, un2, p2 = normal_shock_downstream(mach, rho1, p1, gamma)
    z, y, x = torch.meshgrid(torch.linspace(0.0, 1.0, nz), torch.linspace(0.0, 1.0, ny), torch.linspace(0.0, 1.0, nx), indexing="ij")
    upstream = normal_t[0] * x + normal_t[1] * y + normal_t[2] * z < float(offset)
    rho = torch.where(upstream, torch.tensor(rho1), torch.tensor(rho2)).unsqueeze(0)
    un = torch.where(upstream, torch.tensor(un1), torch.tensor(un2)).unsqueeze(0)
    u, v, w = un * normal_t[0], un * normal_t[1], un * normal_t[2]
    p = torch.where(upstream, torch.tensor(p1), torch.tensor(p2)).unsqueeze(0)
    return primitive_to_conservative_3d(rho, u, v, w, p, gamma=gamma)
