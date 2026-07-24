"""Positivity preservation for density and thermodynamic pressure.

Neural operators may predict non-admissible conservative states. The main
routine blends a candidate state toward a known admissible reference state
using one scalar per batch item. If the candidate increment has zero spatial
mean, this global convex blend retains exact global conservation while
guaranteeing density and pressure floors.
"""

from __future__ import annotations

import torch

from hypersonic.state import pressure


def admissible_mask(
    U: torch.Tensor,
    gamma: float = 1.4,
    rho_floor: float = 1e-6,
    p_floor: float = 1e-6,
) -> torch.Tensor:
    rho_ok = U[:, 0] >= float(rho_floor)
    p_ok = pressure(U, gamma=gamma) >= float(p_floor)
    return rho_ok & p_ok


def is_admissible(
    U: torch.Tensor,
    gamma: float = 1.4,
    rho_floor: float = 1e-6,
    p_floor: float = 1e-6,
) -> torch.Tensor:
    mask = admissible_mask(U, gamma=gamma, rho_floor=rho_floor, p_floor=p_floor)
    return mask.reshape(mask.shape[0], -1).all(dim=1)


def positivity_preserving_blend(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    gamma: float = 1.4,
    rho_floor: float = 1e-6,
    p_floor: float = 1e-6,
    iterations: int = 40,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Blend candidate toward reference until all cells are physically admissible.

    Returns ``(safe_state, theta)`` with ``theta`` shaped ``(B,)`` and
    ``safe_state = reference + theta * (candidate-reference)``. ``theta=1``
    leaves an admissible candidate unchanged.
    """
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate must have identical shapes")
    if not bool(is_admissible(reference, gamma, rho_floor, p_floor).all()):
        raise ValueError("reference state must be admissible")

    batch = reference.shape[0]
    low = torch.zeros(batch, device=reference.device, dtype=reference.dtype)
    high = torch.ones_like(low)
    candidate_ok = is_admissible(candidate, gamma, rho_floor, p_floor)
    low = torch.where(candidate_ok, torch.ones_like(low), low)
    view_shape = (batch,) + (1,) * (reference.ndim - 1)

    for _ in range(int(iterations)):
        mid = 0.5 * (low + high)
        trial = reference + mid.view(view_shape) * (candidate - reference)
        ok = is_admissible(trial, gamma, rho_floor, p_floor)
        low = torch.where(ok, mid, low)
        high = torch.where(ok, high, mid)

    theta = low
    safe = reference + theta.view(view_shape) * (candidate - reference)
    return safe, theta


def primitive_floor_projection(
    U: torch.Tensor,
    gamma: float = 1.4,
    rho_floor: float = 1e-6,
    p_floor: float = 1e-6,
) -> torch.Tensor:
    """Local emergency projection; prefer global blending when conservation matters."""
    from hypersonic.state import (
        conservative_to_primitive,
        primitive_to_conservative,
        primitive_to_conservative_3d,
    )

    prim = conservative_to_primitive(U, gamma=gamma, rho_floor=rho_floor)
    if U.shape[1] == 3:
        rho, u, p = prim
        return primitive_to_conservative(rho.clamp_min(rho_floor), u, p=p.clamp_min(p_floor), gamma=gamma)
    if U.shape[1] == 4:
        rho, u, v, p = prim
        return primitive_to_conservative(rho.clamp_min(rho_floor), u, v, p.clamp_min(p_floor), gamma=gamma)
    rho, u, v, w, p = prim
    return primitive_to_conservative_3d(
        rho.clamp_min(rho_floor), u, v, w, p.clamp_min(p_floor), gamma=gamma
    )
