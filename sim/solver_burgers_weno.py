import numpy as np

def weno5_left(v, i, eps=1e-6):
    vmm, vm, v0, vp, vpp = v[i-2], v[i-1], v[i], v[i+1], v[i+2]
    p0 = (2*vmm - 7*vm + 11*v0)/6
    p1 = (-vm + 5*v0 + 2*vp)/6
    p2 = (2*v0 + 5*vp - vpp)/6
    b0 = 13/12*(vmm-2*vm+v0)**2 + 1/4*(vmm-4*vm+3*v0)**2
    b1 = 13/12*(vm-2*v0+vp)**2 + 1/4*(vm-vp)**2
    b2 = 13/12*(v0-2*vp+vpp)**2 + 1/4*(3*v0-4*vp+vpp)**2
    d0,d1,d2 = 0.1,0.6,0.3
    a0 = d0/(eps+b0)**2
    a1 = d1/(eps+b1)**2
    a2 = d2/(eps+b2)**2
    s = a0+a1+a2
    w0,w1,w2 = a0/s,a1/s,a2/s
    return w0*p0+w1*p1+w2*p2

def weno5_right(v, i, eps=1e-6):
    vm, v0, vp, vpp, vppp = v[i-1], v[i], v[i+1], v[i+2], v[i+3]
    p0 = (-vppp + 5*vpp + 2*vp)/6
    p1 = (2*vpp + 5*vp - v0)/6
    p2 = (11*vp - 7*v0 + 2*vm)/6
    b0 = 13/12*(vppp-2*vpp+vp)**2 + 1/4*(vppp-4*vpp+3*vp)**2
    b1 = 13/12*(vpp-2*vp+v0)**2 + 1/4*(vpp-v0)**2
    b2 = 13/12*(vp-2*v0+vm)**2 + 1/4*(3*vp-4*v0+vm)**2
    d0,d1,d2 = 0.1,0.6,0.3
    a0 = d0/(eps+b0)**2
    a1 = d1/(eps+b1)**2
    a2 = d2/(eps+b2)**2
    s = a0+a1+a2
    w0,w1,w2 = a0/s,a1/s,a2/s
    return w0*p0+w1*p1+w2*p2

def apply_reflective(u, ng):
    # Neumann BC via mirror (approx)
    for k in range(ng):
        u[k] = u[2*ng-k-1]
        u[-k-1] = u[-2*ng+k]

def flux(u):
    return 0.5*u*u

def rhs_weno(u, dx, nu, Tfield, k, E, ng=3):
    '''
    viscous Burgers + reaction:
      u_t + (0.5 u^2)_x = nu u_xx + k*(1-u)*exp(-E/T)
    '''
    up = u.copy()
    apply_reflective(up, ng)

    nx = len(u) - 2*ng
    face_i = np.arange(ng-1, ng+nx)  # nx+1 faces

    uL = np.zeros(nx+1)
    uR = np.zeros(nx+1)
    for j,ii in enumerate(face_i):
        uL[j] = weno5_left(up, ii)
        uR[j] = weno5_right(up, ii)

    a = np.maximum(np.abs(uL), np.abs(uR))
    f = 0.5*(flux(uL)+flux(uR)) - 0.5*a*(uR-uL)  # Rusanov flux

    dudt = np.zeros_like(u)
    div = (f[1:]-f[:-1]) / dx
    dudt[ng:ng+nx] = -div

    ux2 = (up[ng-1:ng+nx-1] - 2*up[ng:ng+nx] + up[ng+1:ng+nx+1]) / (dx*dx)
    dudt[ng:ng+nx] += nu * ux2

    Tin = Tfield
    dudt[ng:ng+nx] += k * (1.0 - up[ng:ng+nx]) * np.exp(-E/np.maximum(Tin, 1e-6))
    return dudt

# def simulate_case(L_mm=20.0, Nx=256, t_end=1.0, Nt_save=100, CFL=0.45,
#                   nu=0.002, k=1.5, E=5.0, dTdx=0.0, b_quad=0.0, seed=0):
#     '''
#     Returns:
#       x (normalized 0..1): (Nx,)
#       t (0..1): (Nt_save,)
#       U: (Nt_save, Nx) solution snapshots
#     '''
#     rng = np.random.default_rng(seed)
#     L = L_mm
#     x = np.linspace(0, L, Nx)
#     dx = x[1]-x[0]
#     xn = x / L
#     # temperature proxy (nondim): 1 + a*x + b*x^2
#     Tfield = 1.0 + 0.35*dTdx*(xn) + 0.40*b_quad*(xn*xn)

#     # # initial condition: kernel near left + stratification imprint
#     # u0 = 0.5*np.ones(Nx)
#     # u0 += 1.2*np.exp(-((x-0.8)/(0.4))**2)
#     # # imprint temperature stratification into initial state (amplifies OOD differences)
#     # u0 += 0.6*(Tfield - 1.0)
#     # u0 = np.clip(u0, 0.0, 2.0)

#     # initial condition: randomized kernel + stratification imprint
#     u0 = 0.5*np.ones(Nx)

#     # 랜덤 커널 파라미터 (seed에 의해 재현됨)
#     x0 = rng.uniform(0.2, 1.6)      # 커널 중심 위치 (원하면 범위 조절)
#     w  = rng.uniform(0.15, 0.8)     # 폭
#     A  = rng.uniform(0.6, 2.0)      # 진폭

#     u0 += A*np.exp(-((x-x0)/(w))**2)

#     # imprint temperature stratification into initial state (amplifies OOD differences)
#     u0 += 0.6*(Tfield - 1.0)

#     # clip 완화(추천): 상한을 조금 올려서 포화 줄이기
#     u0 = np.clip(u0, 0.0, 3.0)

#     ng = 3
#     u = np.zeros(Nx+2*ng)
#     u[ng:ng+Nx] = u0

#     ts = np.linspace(0, t_end, Nt_save)
#     U = np.zeros((Nt_save, Nx), dtype=np.float32)
#     t = 0.0
#     ksave = 0

#     steps = 0
#     while t < t_end - 1e-12:
#         steps += 1
#         umax = np.max(np.abs(u[ng:ng+Nx])) + 1e-6
#         dt = CFL * dx / umax
#         if ksave < Nt_save:
#             dt = min(dt, ts[ksave]-t + 1e-12)
#         dt = max(dt, 1e-6)

#         def F(uvec):
#             return rhs_weno(uvec, dx, nu, Tfield, k, E, ng=ng)

#         # SSP-RK3
#         k1 = F(u); u1 = u + dt*k1
#         k2 = F(u1); u2 = 0.75*u + 0.25*(u1 + dt*k2)
#         k3 = F(u2); u  = (1/3)*u + (2/3)*(u2 + dt*k3)

#         t += dt
#         while ksave < Nt_save and t >= ts[ksave] - 1e-12:
#             U[ksave,:] = u[ng:ng+Nx]
#             ksave += 1
#             if ksave >= Nt_save:
#                 break

#         if steps > 200000:
#             break

#     # return xn.astype(np.float32), (ts/t_end).astype(np.float32), U
#     meta_ic = {"x0": float(x0), "w": float(w), "A": float(A)}
#     return xn.astype(np.float32), (ts/t_end).astype(np.float32), U, meta_ic
def simulate_case(L_mm=20.0, Nx=256, t_end=1.0, Nt_save=100, CFL=0.45,
                  nu=0.002, k=1.5, E=5.0, dTdx=0.0, b_quad=0.0,
                  seed=0, target_label=None):
    """
    Returns:
      x (normalized 0..1): (Nx,)
      t (0..1): (Nt_save,)
      U: (Nt_save, Nx) solution snapshots
      meta_ic: dict with x0, w, A

    target_label:
      - "detonation_like"
      - "deflagration_like"
      - "no_detonation"
      - None
    """
    rng = np.random.default_rng(seed) 
    L = L_mm
    x = np.linspace(0, L, Nx)
    dx = x[1] - x[0]
    xn = x / L

    # temperature proxy (nondim): 1 + a*x + b*x^2
    Tfield = 1.0 + 0.35 * dTdx * xn + 0.40 * b_quad * (xn * xn)

    # initial condition: randomized kernel + stratification imprint
    u0 = 0.5 * np.ones(Nx)

    # target_label-aware kernel sampling
    if target_label == "detonation_like":
        # stronger / sharper / earlier kernel
        x0 = rng.uniform(0.2, 0.9)
        w  = rng.uniform(0.12, 0.35)
        A  = rng.uniform(1.4, 2.4)
    elif target_label == "no_detonation":
        # weaker / broader / later kernel
        x0 = rng.uniform(0.8, 1.6)
        w  = rng.uniform(0.45, 1.0)
        A  = rng.uniform(0.4, 1.0)
    else:
        # default / middle regime
        x0 = rng.uniform(0.3, 1.4)
        w  = rng.uniform(0.18, 0.70)
        A  = rng.uniform(0.8, 1.8)

    u0 += A * np.exp(-((x - x0) / w) ** 2)

    # imprint temperature stratification into initial state
    u0 += 0.6 * (Tfield - 1.0)

    # keep some headroom
    u0 = np.clip(u0, 0.0, 3.0)

    ng = 3
    u = np.zeros(Nx + 2 * ng)
    u[ng:ng + Nx] = u0

    ts = np.linspace(0, t_end, Nt_save)
    U = np.zeros((Nt_save, Nx), dtype=np.float32)
    t = 0.0
    ksave = 0

    steps = 0
    while t < t_end - 1e-12:
        steps += 1
        umax = np.max(np.abs(u[ng:ng + Nx])) + 1e-6
        dt = CFL * dx / umax
        if ksave < Nt_save:
            dt = min(dt, ts[ksave] - t + 1e-12)
        dt = max(dt, 1e-6)

        def F(uvec):
            return rhs_weno(uvec, dx, nu, Tfield, k, E, ng=ng)

        # SSP-RK3
        k1 = F(u);  u1 = u + dt * k1
        k2 = F(u1); u2 = 0.75 * u + 0.25 * (u1 + dt * k2)
        k3 = F(u2); u  = (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt * k3)

        t += dt
        while ksave < Nt_save and t >= ts[ksave] - 1e-12:
            U[ksave, :] = u[ng:ng + Nx]
            ksave += 1
            if ksave >= Nt_save:
                break

        if steps > 200000:
            break

    meta_ic = {"x0": float(x0), "w": float(w), "A": float(A)}
    return xn.astype(np.float32), (ts / t_end).astype(np.float32), U, meta_ic