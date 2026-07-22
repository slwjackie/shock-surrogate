"""Structured finite-volume solver for compressible Euler/Navier--Stokes flows.

The solver uses Rusanov fluxes, SSP-RK3, density/pressure positivity blending,
and optional laminar viscous/heat-conduction terms. It supports 1-D, 2-D, and
3-D Cartesian states; the 3-D implementation is suitable for small validation
and data-generation cases rather than production-scale DNS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from hypersonic.positivity import positivity_preserving_blend
from hypersonic.state import (
    conservative_to_primitive,
    euler_flux_x,
    euler_flux_y,
    euler_flux_z,
    sound_speed,
)

Boundary = Literal["periodic", "outflow", "reflective"]
Equation = Literal["euler", "navier_stokes"]


@dataclass
class CompressibleConfig:
    gamma: float = 1.4
    cfl: float = 0.35
    equation: Equation = "euler"
    viscosity: float = 0.0
    prandtl: float = 0.72
    gas_constant: float = 1.0
    rho_floor: float = 1e-6
    p_floor: float = 1e-6
    bc_x: Boundary = "outflow"
    bc_y: Boundary = "outflow"
    bc_z: Boundary = "outflow"
    max_substeps: int = 100000


def _pad_axis(U: torch.Tensor, axis: int, bc: Boundary, momentum_channel: int) -> torch.Tensor:
    if bc == "periodic":
        return torch.cat((U.narrow(axis, U.shape[axis] - 1, 1), U, U.narrow(axis, 0, 1)), dim=axis)
    if bc == "outflow":
        if U.ndim == 3:
            return F.pad(U, (1, 1), mode="replicate")
        if U.ndim == 4:
            return F.pad(U, (1, 1, 0, 0), mode="replicate") if axis == -1 else F.pad(U, (0, 0, 1, 1), mode="replicate")
        if U.ndim == 5:
            if axis == -1:
                return F.pad(U, (1, 1, 0, 0, 0, 0), mode="replicate")
            if axis == -2:
                return F.pad(U, (0, 0, 1, 1, 0, 0), mode="replicate")
            return F.pad(U, (0, 0, 0, 0, 1, 1), mode="replicate")
    if bc == "reflective":
        left = U.narrow(axis, 0, 1).clone()
        right = U.narrow(axis, U.shape[axis] - 1, 1).clone()
        left[:, momentum_channel] *= -1.0
        right[:, momentum_channel] *= -1.0
        return torch.cat((left, U, right), dim=axis)
    raise ValueError(f"Unsupported boundary condition: {bc}")


def _pad_x(U: torch.Tensor, bc: Boundary) -> torch.Tensor:
    return _pad_axis(U, -1, bc, 1)


def _pad_y(U: torch.Tensor, bc: Boundary) -> torch.Tensor:
    return _pad_axis(U, -2, bc, 2)


def _pad_z(U: torch.Tensor, bc: Boundary) -> torch.Tensor:
    return _pad_axis(U, -3, bc, 3)


def _rusanov(UL: torch.Tensor, UR: torch.Tensor, direction: str, gamma: float) -> torch.Tensor:
    if direction == "x":
        FL, FR = euler_flux_x(UL, gamma), euler_flux_x(UR, gamma)
        velocity_index = 1
    elif direction == "y":
        FL, FR = euler_flux_y(UL, gamma), euler_flux_y(UR, gamma)
        velocity_index = 2
    elif direction == "z":
        FL, FR = euler_flux_z(UL, gamma), euler_flux_z(UR, gamma)
        velocity_index = 3
    else:
        raise ValueError(direction)
    prim_l = conservative_to_primitive(UL, gamma=gamma)
    prim_r = conservative_to_primitive(UR, gamma=gamma)
    a_l, a_r = sound_speed(UL, gamma=gamma), sound_speed(UR, gamma=gamma)
    smax = torch.maximum(prim_l[velocity_index].abs() + a_l, prim_r[velocity_index].abs() + a_r).unsqueeze(1)
    return 0.5 * (FL + FR) - 0.5 * smax * (UR - UL)


def _central_diff(q: torch.Tensor, spacing: float, axis: int, bc: Boundary) -> torch.Tensor:
    momentum = {-1: 1, -2: 2, -3: 3}[axis]
    qp = _pad_axis(q, axis, bc, min(momentum, q.shape[1] - 1))
    left = qp.narrow(axis, 0, qp.shape[axis] - 2)
    right = qp.narrow(axis, 2, qp.shape[axis] - 2)
    return (right - left) / (2.0 * float(spacing))


class StructuredCompressibleSolver:
    def __init__(self, config: CompressibleConfig | None = None):
        self.cfg = config or CompressibleConfig()

    def inviscid_rhs(self, U: torch.Tensor, dx: float, dy: float | None = None, dz: float | None = None) -> torch.Tensor:
        gamma = self.cfg.gamma
        if U.shape[1] == 3 and U.ndim == 3:
            Up = _pad_x(U, self.cfg.bc_x)
            flux = _rusanov(Up[..., :-1], Up[..., 1:], "x", gamma)
            return -(flux[..., 1:] - flux[..., :-1]) / float(dx)
        if U.shape[1] == 4 and U.ndim == 4:
            if dy is None:
                raise ValueError("dy is required for two-dimensional states")
            Ux = _pad_x(U, self.cfg.bc_x)
            Fx = _rusanov(Ux[..., :-1], Ux[..., 1:], "x", gamma)
            rhs = -(Fx[..., 1:] - Fx[..., :-1]) / float(dx)
            Uy = _pad_y(U, self.cfg.bc_y)
            Gy = _rusanov(Uy[..., :-1, :], Uy[..., 1:, :], "y", gamma)
            return rhs - (Gy[..., 1:, :] - Gy[..., :-1, :]) / float(dy)
        if U.shape[1] == 5 and U.ndim == 5:
            if dy is None or dz is None:
                raise ValueError("dy and dz are required for three-dimensional states")
            Ux = _pad_x(U, self.cfg.bc_x)
            Fx = _rusanov(Ux[..., :-1], Ux[..., 1:], "x", gamma)
            rhs = -(Fx[..., 1:] - Fx[..., :-1]) / float(dx)
            Uy = _pad_y(U, self.cfg.bc_y)
            Gy = _rusanov(Uy[..., :-1, :], Uy[..., 1:, :], "y", gamma)
            rhs = rhs - (Gy[..., 1:, :] - Gy[..., :-1, :]) / float(dy)
            Uz = _pad_z(U, self.cfg.bc_z)
            Hz = _rusanov(Uz[..., :-1, :, :], Uz[..., 1:, :, :], "z", gamma)
            return rhs - (Hz[..., 1:, :, :] - Hz[..., :-1, :, :]) / float(dz)
        raise ValueError("Expected U with shape (B,3,Nx), (B,4,Ny,Nx), or (B,5,Nz,Ny,Nx)")

    def viscous_rhs(self, U: torch.Tensor, dx: float, dy: float | None = None, dz: float | None = None) -> torch.Tensor:
        mu = float(self.cfg.viscosity)
        if self.cfg.equation != "navier_stokes" or mu <= 0.0:
            return torch.zeros_like(U)
        gamma = float(self.cfg.gamma)
        cp = gamma * float(self.cfg.gas_constant) / (gamma - 1.0)
        kappa = mu * cp / float(self.cfg.prandtl)
        prim = conservative_to_primitive(U, gamma=gamma)
        rho, p = prim[0], prim[-1]
        T = p / (rho * float(self.cfg.gas_constant))
        if U.shape[1] == 3:
            u = prim[1]
            ux = _central_diff(u.unsqueeze(1), dx, -1, self.cfg.bc_x).squeeze(1)
            Tx = _central_diff(T.unsqueeze(1), dx, -1, self.cfg.bc_x).squeeze(1)
            tau = (4.0 / 3.0) * mu * ux
            qx = -kappa * Tx
            Fv = torch.stack((torch.zeros_like(tau), tau, u * tau - qx), dim=1)
            return _central_diff(Fv, dx, -1, self.cfg.bc_x)
        if U.shape[1] == 4:
            if dy is None:
                raise ValueError("dy is required for two-dimensional Navier--Stokes")
            u, v = prim[1], prim[2]
            ux = _central_diff(u.unsqueeze(1), dx, -1, self.cfg.bc_x).squeeze(1)
            uy = _central_diff(u.unsqueeze(1), dy, -2, self.cfg.bc_y).squeeze(1)
            vx = _central_diff(v.unsqueeze(1), dx, -1, self.cfg.bc_x).squeeze(1)
            vy = _central_diff(v.unsqueeze(1), dy, -2, self.cfg.bc_y).squeeze(1)
            Tx = _central_diff(T.unsqueeze(1), dx, -1, self.cfg.bc_x).squeeze(1)
            Ty = _central_diff(T.unsqueeze(1), dy, -2, self.cfg.bc_y).squeeze(1)
            div = ux + vy
            tau_xx = 2.0 * mu * (ux - div / 3.0)
            tau_yy = 2.0 * mu * (vy - div / 3.0)
            tau_xy = mu * (uy + vx)
            qx, qy = -kappa * Tx, -kappa * Ty
            zero = torch.zeros_like(rho)
            Fv = torch.stack((zero, tau_xx, tau_xy, u * tau_xx + v * tau_xy - qx), dim=1)
            Gv = torch.stack((zero, tau_xy, tau_yy, u * tau_xy + v * tau_yy - qy), dim=1)
            return _central_diff(Fv, dx, -1, self.cfg.bc_x) + _central_diff(Gv, dy, -2, self.cfg.bc_y)
        if dy is None or dz is None:
            raise ValueError("dy and dz are required for three-dimensional Navier--Stokes")
        u, v, w = prim[1], prim[2], prim[3]
        ux = _central_diff(u.unsqueeze(1), dx, -1, self.cfg.bc_x).squeeze(1)
        uy = _central_diff(u.unsqueeze(1), dy, -2, self.cfg.bc_y).squeeze(1)
        uz = _central_diff(u.unsqueeze(1), dz, -3, self.cfg.bc_z).squeeze(1)
        vx = _central_diff(v.unsqueeze(1), dx, -1, self.cfg.bc_x).squeeze(1)
        vy = _central_diff(v.unsqueeze(1), dy, -2, self.cfg.bc_y).squeeze(1)
        vz = _central_diff(v.unsqueeze(1), dz, -3, self.cfg.bc_z).squeeze(1)
        wx = _central_diff(w.unsqueeze(1), dx, -1, self.cfg.bc_x).squeeze(1)
        wy = _central_diff(w.unsqueeze(1), dy, -2, self.cfg.bc_y).squeeze(1)
        wz = _central_diff(w.unsqueeze(1), dz, -3, self.cfg.bc_z).squeeze(1)
        Tx = _central_diff(T.unsqueeze(1), dx, -1, self.cfg.bc_x).squeeze(1)
        Ty = _central_diff(T.unsqueeze(1), dy, -2, self.cfg.bc_y).squeeze(1)
        Tz = _central_diff(T.unsqueeze(1), dz, -3, self.cfg.bc_z).squeeze(1)
        div = ux + vy + wz
        tau_xx = 2.0 * mu * (ux - div / 3.0)
        tau_yy = 2.0 * mu * (vy - div / 3.0)
        tau_zz = 2.0 * mu * (wz - div / 3.0)
        tau_xy = mu * (uy + vx)
        tau_xz = mu * (uz + wx)
        tau_yz = mu * (vz + wy)
        qx, qy, qz = -kappa * Tx, -kappa * Ty, -kappa * Tz
        zero = torch.zeros_like(rho)
        Fv = torch.stack((zero, tau_xx, tau_xy, tau_xz, u * tau_xx + v * tau_xy + w * tau_xz - qx), dim=1)
        Gv = torch.stack((zero, tau_xy, tau_yy, tau_yz, u * tau_xy + v * tau_yy + w * tau_yz - qy), dim=1)
        Hv = torch.stack((zero, tau_xz, tau_yz, tau_zz, u * tau_xz + v * tau_yz + w * tau_zz - qz), dim=1)
        return _central_diff(Fv, dx, -1, self.cfg.bc_x) + _central_diff(Gv, dy, -2, self.cfg.bc_y) + _central_diff(Hv, dz, -3, self.cfg.bc_z)

    def rhs(self, U: torch.Tensor, dx: float, dy: float | None = None, dz: float | None = None) -> torch.Tensor:
        return self.inviscid_rhs(U, dx, dy, dz) + self.viscous_rhs(U, dx, dy, dz)

    def stable_dt(self, U: torch.Tensor, dx: float, dy: float | None = None, dz: float | None = None) -> float:
        prim = conservative_to_primitive(U, gamma=self.cfg.gamma)
        a = sound_speed(U, gamma=self.cfg.gamma, p_floor=self.cfg.p_floor)
        inv_dt = float((prim[1].abs() + a).amax().item()) / float(dx)
        if U.shape[1] >= 4:
            if dy is None:
                raise ValueError("dy is required")
            inv_dt += float((prim[2].abs() + a).amax().item()) / float(dy)
        if U.shape[1] == 5:
            if dz is None:
                raise ValueError("dz is required")
            inv_dt += float((prim[3].abs() + a).amax().item()) / float(dz)
        dt = float(self.cfg.cfl) / max(inv_dt, 1e-12)
        if self.cfg.equation == "navier_stokes" and self.cfg.viscosity > 0.0:
            rho_min = float(U[:, 0].amin().item())
            nu = float(self.cfg.viscosity) / max(rho_min, self.cfg.rho_floor)
            spacings = [float(dx)] + ([] if dy is None else [float(dy)]) + ([] if dz is None else [float(dz)])
            dt = min(dt, 0.2 * min(h * h for h in spacings) / max(nu, 1e-12))
        return max(dt, 1e-12)

    def _safe_stage(self, reference: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        safe, _ = positivity_preserving_blend(reference, candidate, gamma=self.cfg.gamma, rho_floor=self.cfg.rho_floor, p_floor=self.cfg.p_floor)
        return safe

    def rk3_step(self, U: torch.Tensor, dt: float, dx: float, dy: float | None = None, dz: float | None = None) -> torch.Tensor:
        U1 = self._safe_stage(U, U + float(dt) * self.rhs(U, dx, dy, dz))
        U2 = self._safe_stage(U, 0.75 * U + 0.25 * (U1 + float(dt) * self.rhs(U1, dx, dy, dz)))
        return self._safe_stage(U, (1.0 / 3.0) * U + (2.0 / 3.0) * (U2 + float(dt) * self.rhs(U2, dx, dy, dz)))

    @torch.no_grad()
    def advance(self, U0: torch.Tensor, dt_total: float, dx: float, dy: float | None = None, dz: float | None = None) -> tuple[torch.Tensor, int]:
        U = U0.clone()
        elapsed = 0.0
        steps = 0
        while elapsed < float(dt_total) - 1e-14:
            dt = min(self.stable_dt(U, dx, dy, dz), float(dt_total) - elapsed)
            U = self.rk3_step(U, dt, dx, dy, dz)
            elapsed += dt
            steps += 1
            if steps > int(self.cfg.max_substeps):
                raise RuntimeError("Exceeded max_substeps while advancing compressible state")
        return U, steps
