"""State conversion and thermodynamic helpers for a calorically perfect gas."""

from __future__ import annotations

from typing import Tuple

import torch


def primitive_to_conservative(
    rho: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor | None = None,
    p: torch.Tensor | None = None,
    gamma: float = 1.4,
) -> torch.Tensor:
    """Convert primitive variables to 1-D or 2-D conservative variables."""
    if p is None:
        if v is None:
            raise ValueError("Pressure must be supplied")
        p = v
        v = None
    kinetic = 0.5 * rho * u.square()
    if v is not None:
        kinetic = kinetic + 0.5 * rho * v.square()
    E = p / (float(gamma) - 1.0) + kinetic
    if v is None:
        return torch.stack((rho, rho * u, E), dim=1)
    return torch.stack((rho, rho * u, rho * v, E), dim=1)


def primitive_to_conservative_3d(
    rho: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    p: torch.Tensor,
    gamma: float = 1.4,
) -> torch.Tensor:
    kinetic = 0.5 * rho * (u.square() + v.square() + w.square())
    E = p / (float(gamma) - 1.0) + kinetic
    return torch.stack((rho, rho * u, rho * v, rho * w, E), dim=1)


def conservative_to_primitive(
    U: torch.Tensor,
    gamma: float = 1.4,
    rho_floor: float = 1e-12,
) -> Tuple[torch.Tensor, ...]:
    """Convert 1-D, 2-D, or 3-D conservative states to primitive variables."""
    if U.ndim < 3:
        raise ValueError(f"Expected channel-first batched state, got {tuple(U.shape)}")
    c = U.shape[1]
    if c not in (3, 4, 5):
        raise ValueError(f"Expected 3, 4, or 5 conservative channels, got {c}")
    rho = U[:, 0].clamp_min(float(rho_floor))
    u = U[:, 1] / rho
    if c == 3:
        kinetic = 0.5 * rho * u.square()
        p = (float(gamma) - 1.0) * (U[:, 2] - kinetic)
        return rho, u, p
    v = U[:, 2] / rho
    if c == 4:
        kinetic = 0.5 * rho * (u.square() + v.square())
        p = (float(gamma) - 1.0) * (U[:, 3] - kinetic)
        return rho, u, v, p
    w = U[:, 3] / rho
    kinetic = 0.5 * rho * (u.square() + v.square() + w.square())
    p = (float(gamma) - 1.0) * (U[:, 4] - kinetic)
    return rho, u, v, w, p


def pressure(U: torch.Tensor, gamma: float = 1.4, rho_floor: float = 1e-12) -> torch.Tensor:
    return conservative_to_primitive(U, gamma=gamma, rho_floor=rho_floor)[-1]


def sound_speed(
    U: torch.Tensor,
    gamma: float = 1.4,
    rho_floor: float = 1e-12,
    p_floor: float = 1e-12,
) -> torch.Tensor:
    rho = U[:, 0].clamp_min(float(rho_floor))
    p = pressure(U, gamma=gamma, rho_floor=rho_floor).clamp_min(float(p_floor))
    return torch.sqrt(float(gamma) * p / rho)


def mach_number(
    U: torch.Tensor,
    gamma: float = 1.4,
    rho_floor: float = 1e-12,
    p_floor: float = 1e-12,
) -> torch.Tensor:
    prim = conservative_to_primitive(U, gamma=gamma, rho_floor=rho_floor)
    if U.shape[1] == 3:
        speed = prim[1].abs()
    elif U.shape[1] == 4:
        speed = torch.sqrt(prim[1].square() + prim[2].square())
    else:
        speed = torch.sqrt(prim[1].square() + prim[2].square() + prim[3].square())
    return speed / sound_speed(U, gamma=gamma, rho_floor=rho_floor, p_floor=p_floor)


def euler_flux_x(U: torch.Tensor, gamma: float = 1.4) -> torch.Tensor:
    if U.shape[1] == 3:
        rho, u, p = conservative_to_primitive(U, gamma=gamma)
        return torch.stack((rho * u, rho * u.square() + p, (U[:, 2] + p) * u), dim=1)
    if U.shape[1] == 4:
        rho, u, v, p = conservative_to_primitive(U, gamma=gamma)
        return torch.stack((rho * u, rho * u.square() + p, rho * u * v, (U[:, 3] + p) * u), dim=1)
    rho, u, v, w, p = conservative_to_primitive(U, gamma=gamma)
    return torch.stack(
        (rho * u, rho * u.square() + p, rho * u * v, rho * u * w, (U[:, 4] + p) * u),
        dim=1,
    )


def euler_flux_y(U: torch.Tensor, gamma: float = 1.4) -> torch.Tensor:
    if U.shape[1] == 4:
        rho, u, v, p = conservative_to_primitive(U, gamma=gamma)
        return torch.stack((rho * v, rho * u * v, rho * v.square() + p, (U[:, 3] + p) * v), dim=1)
    if U.shape[1] != 5:
        raise ValueError("y-direction flux requires a 2-D/3-D state")
    rho, u, v, w, p = conservative_to_primitive(U, gamma=gamma)
    return torch.stack(
        (rho * v, rho * u * v, rho * v.square() + p, rho * v * w, (U[:, 4] + p) * v),
        dim=1,
    )


def euler_flux_z(U: torch.Tensor, gamma: float = 1.4) -> torch.Tensor:
    if U.shape[1] != 5:
        raise ValueError("z-direction flux requires a 3-D five-channel state")
    rho, u, v, w, p = conservative_to_primitive(U, gamma=gamma)
    return torch.stack(
        (rho * w, rho * u * w, rho * v * w, rho * w.square() + p, (U[:, 4] + p) * w),
        dim=1,
    )
