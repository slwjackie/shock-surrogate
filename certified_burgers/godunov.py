"""First-order monotone Godunov finite-volume solver for inviscid Burgers."""
from __future__ import annotations
import numpy as np

def burgers_flux(u):
    u = np.asarray(u)
    return 0.5 * u * u

def godunov_flux(u_left, u_right):
    ul = np.asarray(u_left)
    ur = np.asarray(u_right)
    ul, ur = np.broadcast_arrays(ul, ur)
    out = np.empty_like(ul, dtype=np.result_type(ul, ur, np.float64))
    rare = ul <= ur
    shock = ~rare
    r_pos = rare & (ul >= 0.0)
    r_neg = rare & (ur <= 0.0)
    r_cross = rare & ~(r_pos | r_neg)
    out[r_pos] = burgers_flux(ul[r_pos])
    out[r_neg] = burgers_flux(ur[r_neg])
    out[r_cross] = 0.0
    if np.any(shock):
        speed = 0.5 * (ul + ur)
        s_right = shock & (speed >= 0.0)
        s_left = shock & (speed < 0.0)
        out[s_right] = burgers_flux(ul[s_right])
        out[s_left] = burgers_flux(ur[s_left])
    return out

def stable_dt(u, dx: float, cfl: float = 0.8, eps: float = 1e-14) -> float:
    if not (0.0 < cfl <= 1.0):
        raise ValueError("For the monotone first-order PoC, require 0 < cfl <= 1.")
    umax = float(np.max(np.abs(np.asarray(u))))
    if umax <= eps:
        return float("inf")
    return cfl * float(dx) / umax

def _check_dt(u, dx: float, dt: float, cfl_limit: float = 1.0) -> None:
    cfl_number = float(dt) * float(np.max(np.abs(u))) / float(dx)
    if cfl_number > cfl_limit + 1e-12:
        raise ValueError(f"CFL violation: dt*max|u|/dx={cfl_number:.6g} > {cfl_limit}.")

def godunov_step(u, dx: float, dt: float | None = None, *, cfl: float = 0.8, boundary: str = "periodic"):
    state = np.asarray(u, dtype=np.float64)
    if state.ndim < 1 or state.shape[-1] < 2:
        raise ValueError("u must contain at least two cells.")
    if dx <= 0:
        raise ValueError("dx must be positive.")
    if dt is None:
        dt = stable_dt(state, dx, cfl)
        if not np.isfinite(dt):
            return state.copy(), float(dt)
    else:
        dt = float(dt)
        if dt <= 0:
            raise ValueError("dt must be positive.")
        _check_dt(state, dx, dt)
    if boundary == "periodic":
        right = np.roll(state, -1, axis=-1)
        flux_right = godunov_flux(state, right)
        flux_left = np.roll(flux_right, 1, axis=-1)
    elif boundary == "outflow":
        left_states = np.concatenate([state[..., :1], state], axis=-1)
        right_states = np.concatenate([state, state[..., -1:]], axis=-1)
        face_flux = godunov_flux(left_states, right_states)
        flux_left = face_flux[..., :-1]
        flux_right = face_flux[..., 1:]
    else:
        raise ValueError("boundary must be 'periodic' or 'outflow'.")
    updated = state - (dt / float(dx)) * (flux_right - flux_left)
    return updated, dt

def advance(u, dx: float, n_steps: int, *, dt: float | None = None, cfl: float = 0.8, boundary: str = "periodic"):
    if n_steps < 0:
        raise ValueError("n_steps must be nonnegative.")
    state = np.asarray(u, dtype=np.float64).copy()
    total = 0.0
    for _ in range(n_steps):
        state, used_dt = godunov_step(state, dx, dt=dt, cfl=cfl, boundary=boundary)
        if not np.isfinite(used_dt):
            break
        total += used_dt
    return state, total
