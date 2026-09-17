"""WENO5 + Rusanov + SSP-RK3 solver for reactive viscous Burgers dynamics."""

from __future__ import annotations

import numpy as np

from numerics.remap import (
    cell_edges_from_centers,
    conservative_remap_1d,
    is_uniform_grid,
    uniform_centers_for_domain,
)


def weno5_left(v, i, eps=1e-6):
    vmm, vm, v0, vp, vpp = v[i-2], v[i-1], v[i], v[i+1], v[i+2]
    p0 = (2*vmm - 7*vm + 11*v0)/6
    p1 = (-vm + 5*v0 + 2*vp)/6
    p2 = (2*v0 + 5*vp - vpp)/6
    b0 = 13/12*(vmm-2*vm+v0)**2 + 1/4*(vmm-4*vm+3*v0)**2
    b1 = 13/12*(vm-2*v0+vp)**2 + 1/4*(vm-vp)**2
    b2 = 13/12*(v0-2*vp+vpp)**2 + 1/4*(3*v0-4*vp+vpp)**2
    d0,d1,d2 = 0.1,0.6,0.3
    a0,a1,a2 = d0/(eps+b0)**2,d1/(eps+b1)**2,d2/(eps+b2)**2
    total = a0+a1+a2
    return (a0*p0+a1*p1+a2*p2)/total


def weno5_right(v, i, eps=1e-6):
    vm, v0, vp, vpp, vppp = v[i-1], v[i], v[i+1], v[i+2], v[i+3]
    p0 = (-vppp + 5*vpp + 2*vp)/6
    p1 = (2*vpp + 5*vp - v0)/6
    p2 = (11*vp - 7*v0 + 2*vm)/6
    b0 = 13/12*(vppp-2*vpp+vp)**2 + 1/4*(vppp-4*vpp+3*vp)**2
    b1 = 13/12*(vpp-2*vp+v0)**2 + 1/4*(vpp-v0)**2
    b2 = 13/12*(vp-2*v0+vm)**2 + 1/4*(3*vp-4*v0+vm)**2
    d0,d1,d2 = 0.1,0.6,0.3
    a0,a1,a2 = d0/(eps+b0)**2,d1/(eps+b1)**2,d2/(eps+b2)**2
    total = a0+a1+a2
    return (a0*p0+a1*p1+a2*p2)/total


def apply_reflective(u, ng):
    for offset in range(ng):
        u[offset] = u[2*ng-offset-1]
        u[-offset-1] = u[-2*ng+offset]


def flux(u):
    return 0.5*u*u


def temperature_field(x_normalized, dTdx=0.0, b_quad=0.0):
    return 1.0 + 0.35*float(dTdx)*x_normalized + 0.40*float(b_quad)*x_normalized**2


def rhs_weno(u, dx, nu, Tfield, k, E, ng=3):
    up = u.copy()
    apply_reflective(up, ng)
    nx = len(u)-2*ng
    face_indices = np.arange(ng-1, ng+nx)
    u_left,u_right = np.zeros(nx+1),np.zeros(nx+1)
    for j,index in enumerate(face_indices):
        u_left[j] = weno5_left(up,index)
        u_right[j] = weno5_right(up,index)
    speed = np.maximum(np.abs(u_left),np.abs(u_right))
    numerical_flux = 0.5*(flux(u_left)+flux(u_right)) - 0.5*speed*(u_right-u_left)
    dudt = np.zeros_like(u)
    dudt[ng:ng+nx] = -(numerical_flux[1:]-numerical_flux[:-1])/dx
    laplacian = (up[ng-1:ng+nx-1]-2*up[ng:ng+nx]+up[ng+1:ng+nx+1])/(dx*dx)
    dudt[ng:ng+nx] += float(nu)*laplacian
    dudt[ng:ng+nx] += float(k)*(1.0-up[ng:ng+nx])*np.exp(-float(E)/np.maximum(Tfield,1e-6))
    return dudt


def _ssprk3_step(u, dt, dx, nu, Tfield, k, E, ng=3):
    def operator(state):
        return rhs_weno(state, dx, nu, Tfield, k, E, ng=ng)
    k1 = operator(u); u1 = u + dt*k1
    k2 = operator(u1); u2 = 0.75*u + 0.25*(u1 + dt*k2)
    k3 = operator(u2)
    return (1.0/3.0)*u + (2.0/3.0)*(u2 + dt*k3)


def _stable_dt(u_inner, dx, nu, k, cfl):
    advective = float(cfl)*dx/(float(np.max(np.abs(u_inner)))+1e-8)
    diffusive = 0.45*dx*dx/max(float(nu),1e-12)
    reactive = 0.25/max(float(k),1e-12)
    return max(min(advective,diffusive,reactive),1e-8)


def _prepare_solver_grid(state, L_mm, x_physical=None):
    """Return uniform internal state/grid and optional source grid for remapping."""
    nx = state.size
    if x_physical is None:
        internal_x = np.linspace(0.0, float(L_mm), nx)
        return state, internal_x, None

    source_x = np.asarray(x_physical, dtype=np.float64).reshape(-1)
    if source_x.size != nx:
        raise ValueError("x_physical and state must have identical length")
    if np.any(np.diff(source_x) <= 0):
        raise ValueError("x_physical must be strictly increasing")
    if is_uniform_grid(source_x):
        return state, source_x, None

    internal_x = uniform_centers_for_domain(source_x, n_cells=nx)
    internal_state = conservative_remap_1d(state, source_x, internal_x)
    return internal_state, internal_x, source_x


def advance_state(
    u0,
    dt_total,
    L_mm=20.0,
    CFL=0.45,
    nu=0.002,
    k=1.5,
    E=6.0,
    dTdx=0.0,
    b_quad=0.0,
    max_steps=200000,
    x_physical=None,
):
    """Advance any monotone 1-D grid state by one saved-step interval.

    WENO5 remains a uniform-grid method. Nonuniform inputs are conservatively
    remapped to a uniform computational grid, advanced there, and conservatively
    remapped back to the original centers.
    """
    state = np.asarray(u0,dtype=np.float64).reshape(-1)
    nx = state.size
    if nx < 7:
        raise ValueError("WENO5 requires at least seven physical cells")
    internal_state, x, source_x = _prepare_solver_grid(state, L_mm, x_physical=x_physical)
    dx_values = np.diff(x)
    if not np.allclose(dx_values, dx_values.mean(), rtol=1e-5, atol=1e-10):
        raise RuntimeError("Internal WENO grid is not uniform")
    dx = float(dx_values.mean())

    if x_physical is None:
        x_normalized = x / float(L_mm)
    else:
        edges = cell_edges_from_centers(x)
        x_normalized = (x - edges[0]) / max(edges[-1] - edges[0], 1e-12)
    temperature = temperature_field(x_normalized,dTdx=dTdx,b_quad=b_quad)
    ng = 3
    u = np.zeros(nx+2*ng,dtype=np.float64)
    u[ng:ng+nx] = internal_state
    elapsed,steps = 0.0,0
    while elapsed < float(dt_total)-1e-14:
        dt = min(_stable_dt(u[ng:ng+nx],dx,nu,k,CFL),float(dt_total)-elapsed)
        u = _ssprk3_step(u,dt,dx,nu,temperature,k,E,ng=ng)
        elapsed += dt; steps += 1
        if steps >= int(max_steps):
            raise RuntimeError("advance_state exceeded max_steps")
    result = u[ng:ng+nx]
    if source_x is not None:
        result = conservative_remap_1d(result, x, source_x)
    return result.astype(np.float32),steps


def simulate_case(L_mm=20.0,Nx=256,t_end=1.0,Nt_save=100,CFL=0.45,nu=0.002,k=1.5,E=5.0,dTdx=0.0,b_quad=0.0,seed=0,target_label=None):
    rng = np.random.default_rng(seed)
    x = np.linspace(0,float(L_mm),int(Nx)); dx = float(x[1]-x[0]); xn = x/float(L_mm)
    temperature = temperature_field(xn,dTdx=dTdx,b_quad=b_quad)
    u0 = 0.5*np.ones(int(Nx))
    if target_label == "detonation_like":
        x0,width,amplitude = rng.uniform(0.2,0.9),rng.uniform(0.12,0.35),rng.uniform(1.4,2.4)
    elif target_label == "no_detonation":
        x0,width,amplitude = rng.uniform(0.8,1.6),rng.uniform(0.45,1.0),rng.uniform(0.4,1.0)
    else:
        x0,width,amplitude = rng.uniform(0.3,1.4),rng.uniform(0.18,0.70),rng.uniform(0.8,1.8)
    u0 += amplitude*np.exp(-((x-x0)/width)**2)
    u0 += 0.6*(temperature-1.0)
    u0 = np.clip(u0,0.0,3.0)
    ng = 3
    u = np.zeros(int(Nx)+2*ng); u[ng:ng+int(Nx)] = u0
    save_times = np.linspace(0.0,float(t_end),int(Nt_save))
    snapshots = np.zeros((int(Nt_save),int(Nx)),dtype=np.float32)
    snapshots[0] = u0.astype(np.float32)
    save_index,time,steps = 1,0.0,0
    while save_index < int(Nt_save):
        target_time = float(save_times[save_index])
        while time < target_time-1e-14:
            dt = min(_stable_dt(u[ng:ng+int(Nx)],dx,nu,k,CFL),target_time-time)
            u = _ssprk3_step(u,dt,dx,nu,temperature,k,E,ng=ng)
            time += dt; steps += 1
            if steps > 200000:
                raise RuntimeError("simulate_case exceeded 200000 steps")
        snapshots[save_index] = u[ng:ng+int(Nx)]
        save_index += 1
    metadata = {"x0":float(x0),"w":float(width),"A":float(amplitude)}
    return xn.astype(np.float32),(save_times/float(t_end)).astype(np.float32),snapshots,metadata
