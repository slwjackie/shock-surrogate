# Stage A/B mathematical contracts

## Claim levels

1. **Stage A:** a deterministic error bound relative to the exact-arithmetic,
   same-grid Burgers Godunov map. Offline interval tables include the declared
   floating neural-flux implementation; online rational accounting includes
   update and fallback roundoff. This is not an exact-PDE error certificate.
2. **Stage B default:** physical admissibility plus empirical residual,
   step-consistency or dropout gates. These are not error certificates.
3. **Stage B interval extension:** a computable, restricted exact-ODE error
   enclosure. Supported thermochemistry and a closed trajectory tube are
   required. Unsupported or inconclusive cases return `available=False`.
   A local certified neural step plus unvalidated CVODES fallback is **not** a
   globally certified trajectory. `audit_complete_trajectory` must validate
   every slab, including fallback slabs, before providing a final exact-ODE bound.

The algebra below is elementary stability/error propagation, not a claim of a
new PDE stability theorem. The research questions concern useful enclosures,
verification cost and selective integration under these contracts.

## A1. A simultaneous verifier counterexample

On the unit periodic grid x_i=(i+1/2)/N, take v_i=c and
p_i=c+a sin(2 pi K x_i), with a != 0, 4 < K and 2K+4 < N.
The trusted Godunov update leaves v constant. Discrete Fourier orthogonality gives
sum_i (p_i-v_i)=0. The current weak verifier tests only sine/cosine modes 1..4.
Its midpoint Burgers flux contains frequencies 0,K,2K; none aliases a tested
mode. Thus its temporal and flux pairings vanish in real arithmetic, while
h sum_i |p_i-c| > 0. Numerical values near 1e-16 are roundoff, not the proof.

Consequently no finite C can establish eta <= C*q universally for these
arbitrary admissible state/candidate pairs. This does **not** prove impossibility
on the range of a particular trained model. In particular, a deterministic
translation-equivariant CNN produces a constant output from a constant input;
this Fourier candidate need not lie in that CNN's range.

## A2. One-step bound and global bound

Write lambda=dt/h, e_(i+1/2)=Fhat_(i+1/2)-G(v_i,v_(i+1)). For a common input,

    P(v)_i - S(v)_i = -lambda*(e_(i+1/2)-e_(i-1/2)).

If |e_(i+1/2)| <= delta_(i+1/2), then

    ||P(v)-S(v)||_(1,h)
      <= lambda*h*sum_i(delta_(i+1/2)+delta_(i-1/2))
       = 2*lambda*h*sum_i delta_(i+1/2).

Only the final equality uses the periodic interface counting. This local
algebra does not require CFL. The global theorem does: on |v_i|<=M with
lambda*M<=1, Burgers Godunov is monotone and conservative and hence L1
nonexpansive. To see the latter, set m=min(a,b), q=max(a,b) componentwise.
Monotonicity places S(a),S(b) between S(m),S(q). Therefore
sum|S(a)-S(b)| <= sum(S(q)-S(m)) = sum(q-m) = sum|a-b|.

For hybrid v_(n+1)=P(v_n) on accept and S(v_n) on fallback, and reference
u_(n+1)=S(u_n), insert S(v_n) in the difference. This yields

    E_(n+1) <= E_n + accepted_n*eta_bar_n,
    E_N <= E_0 + sum_(accepted n) eta_bar_n.

The NN itself need not be monotone. The candidate must remain in the common
state envelope, and the same lambda must be used by both trajectories.

## A3. Floating-point realization

`FrozenFlux.predict` uses separate binary64 multiplication and addition in a
fixed order, not an unbounded BLAS/GPU kernel. Interval propagation outwardly
encloses that implementation on every input box. The Burgers flux has the exact
formula G(a,b)=max(max(a,0)^2,min(b,0)^2)/2 and is monotone in a and antitone in b.
Its box extrema are therefore obtained at the corresponding two corners.
Rational arithmetic computes the final difference bound without under-rounding.

The optional Lipschitz construction uses the real ReLU network's induced
infinity-norm bound and Burgers' M-Lipschitz constant on the box. A forward-error
bound (gamma_(2d), propagated layerwise, with subnormal allowances) adds the
frozen implementation's floating error. `hybrid` takes the smaller of two valid
bounds; subdivision exhaustively covers a box, rather than sampling it.

The floating state update adds its exact rational arithmetic defect to the
local bound. Floating fallback also has a small defect rho_n relative to ideal
Godunov, so the implemented budget is

    B_N = E_0 + sum_(accepted n) eta_bar_n + sum_(fallback n) rho_n.

Audits comparing with a floating all-Godunov trajectory add that reference's
roundoff budget as well. The accepted-advice budget is exact rational; a nominal
zero global tolerance cannot eliminate unavoidable fallback roundoff.

Assumptions: IEEE-754 correctly rounded elementary operations, gradual
underflow, no fast-math/FMA substitution, and the documented frozen operation
order. This is an explicit arithmetic argument plus tested implementation,
not a proof-assistant-checked program or a hardware-independent claim.

A certificate is bound to a model fingerprint. Loading a table normally
recomputes its proof obligations, including subdivision/method; a model hash
alone does not authenticate a hand-edited bound table. `verify=False` is only
for already verified in-memory artifacts, not for untrusted tables.

## A4. Limitations and architecture

The old `ConservativeFluxSurrogate` uses a GELU spatial CNN and is unchanged.
The new certified model is a two-input ReLU numerical flux. Its direct Godunov
flux training targets remove the undetermined constant flux gauge present in
state-only training. A per-face absolute-error bound can still be loose because
only spatial flux differences affect the state. H>1 and certification of the
old 5-input CNN are not covered by this numerical certificate.

## B1. Constant-volume kinetics

At fixed density rho, the state is z=(T,Y_1,...,Y_s). With molar production
rates omega_k [kmol/(m^3 s)] and molecular weights W_k [kg/kmol],

    dY_k/dt = W_k*omega_k/rho,
    dT/dt = -sum_k ubar_k(T)*omega_k / (rho*c_v).

Here ubar_k includes formation energy. No additional heat source is added to
conserved total internal energy. The usual enthalpy heat-release diagnostic
and the internal-energy source are reported separately. Element-constrained
composition projection and constant-U,V temperature recovery are numerical
operations; guard tolerances and projection costs remain visible.

## B2. Restricted ODE certificate

Let zhat(t) be the linear reconstruction between a current state and a proposed
endpoint, with residual r=zhat'-R(zhat). In fixed component-scaled infinity norm,
assume the logarithmic norm of the Jacobian on a convex tube obeys mu<=mu_bar.
Interval forward-mode AD supplies an enclosure J; the bound used is

    mu_inf(D^-1 J D) <= max_i[J_ii.upper + sum_(j!=i) sup|J_ij|*D_j/D_i].

This is not the eigenvalue spectral radius. A large negative eigenvalue causing
stiffness does not itself imply forward perturbation growth. Finite-difference
spectra in `stiffness_diagnostic` are explicitly only diagnostics.

Dini differentiation and Gronwall give, for residual norm <=r_bar on a slab,

    e(t) <= exp(mu_bar*t)*e(0)
            + r_bar*(exp(mu_bar*t)-1)/mu_bar,

with the continuous limit e(0)+t*r_bar at mu_bar=0. Interval exp/ln and outward
arithmetic evaluate the expression. The current tube checker conservatively
replaces negative mu_bar by zero. A first-exit argument validates the assumed
tube only when the bound stays strictly inside its radius; otherwise it fails
closed. Each time piece uses an interval enclosure over its *whole* path,
not a few sampled residuals.

Implemented kinetics: fixed-density ideal gas, NASA7 within a single polynomial
branch, integer elementary mass action, reversible equilibrium constants,
third bodies, Lindemann and Troe falloff. The shipped h2o2.yaml mechanism's 10
species and 29 reactions are supported, including its Troe reaction. Crossings
of NASA breakpoints, nonpositive log/heat-capacity enclosures, unsupported rate
laws, modified reaction multipliers, and failed tube closure reject.

The target is the exact ODE defined by the declared binary64 mechanism and
thermodynamic constants, **not** experimental combustion truth or CVODES' exact
trajectory. CVODES tolerance refinement is an independent numerical comparison,
not a rigorous error bound. The expensive all-slab audit withdraws the global
claim at its first unvalidated slab. Full ignition-trajectory certification and
practical speedup are experimental goals, not assumed results.

## B3. Shock conditioning and what is not solved

Frozen incident/reflected shocks solve mass, momentum and energy jumps with
composition-dependent, temperature-dependent Cantera thermodynamics, not fixed
gamma. A reflected wave is chosen so the gas behind it is stationary at the
wall. The resulting thermochemical state starts an independent adiabatic,
constant-volume reactor. This is an idealized post-reflected-shock ignition
model, with no boundary-layer correction, shock propagation, or reaction-wave
feedback. It is not reactive Euler, detonation structure, flashback,
thermoacoustic stability or a NOx model.

## References / model definitions

- Cantera ideal-gas reactor equations: https://cantera.org/stable/reference/reactors/ideal-gas-reactor.html
- Cantera h2o2 mechanism: https://cantera.org/stable/examples/input/h2o2.html
- Cantera shock-tube reactor example: https://cantera.org/stable/examples/python/reactors/NonIdealShockTube.html
- Python Decimal correctly rounded exp/ln: https://docs.python.org/3/library/decimal.html
- Crandall & Majda (1980), Monotone difference approximations for scalar
  conservation laws, Mathematics of Computation 34, 1–21.

`h2o2.yaml` is a development mechanism derived from the GRI-Mech H/O subset.
Agreement with the same mechanism in Cantera is numerical verification, not
independent experimental validation. Supply an independently chosen mechanism
with `--mechanism` for a separate mechanism-dependence study.
