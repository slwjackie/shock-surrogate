# Numerical data generation: scientific basis and implementation

## Scope

The repository's Phase-1 dataset is not a chemically complete detonation CFD
dataset. It is a controlled one-dimensional analogue intended to isolate three
features that matter for the learning-augmented controller:

1. nonlinear wave steepening and shock formation,
2. viscous regularization, and
3. temperature-dependent reaction amplification.

The generated trajectories therefore support algorithm development and ablation,
not quantitative prediction of a real explosive or hypersonic vehicle.

## Governing equation

The solver advances

\[
 u_t + \left(\frac{u^2}{2}\right)_x
 = \nu u_{xx} + k(1-u)\exp[-E/T(x)],
\]

with

\[
 T(x)=1+0.35\,dTdx\,x_n+0.40\,b_{quad}x_n^2,
 \qquad x_n=x/L.
\]

The terms have deliberately separated roles:

- `u_t + (u^2/2)_x`: Burgers transport; larger values travel faster and steepen
  a compression into a shock-like front.
- `nu*u_xx`: viscosity; sets the regularized front thickness and suppresses
  unresolved grid-scale oscillations.
- `k*(1-u)*exp(-E/T)`: bounded Arrhenius-like source; `k` controls reaction time,
  `E` controls activation sensitivity, and the temperature proxy controls spatial
  forcing.

This is a model problem. `u` is a scalar progress/state variable rather than the
Euler state vector `(rho, rho*u, E)`.

## Initial-condition ensemble

Each case starts from a background state plus a Gaussian pulse and a temperature
bias:

\[
 u(x,0)=0.5+A\exp[-((x-x_0)/w)^2]+0.6(T(x)-1),
\]

followed by clipping to `[0,3]`.

The generator samples `(x0,w,A)` from broad label-guided proposal ranges. The
proposal label is not the final label; it only improves sampling efficiency for
rare strong/weak trajectories. The final regime is assigned from solver-derived
gradient diagnostics after the trajectory has been computed.

## Spatial discretization

The physical grid is uniform with `Nx` points over `[0,L_mm]` for dataset
creation. Three ghost cells are used on each side.

For the nonlinear flux:

1. WENO5 constructs left and right states at every interface from three candidate
   third-order polynomials.
2. Smoothness indicators downweight stencils that cross a discontinuity.
3. A local Lax-Friedrichs/Rusanov flux combines the two reconstructed states:

\[
 \hat f(u_L,u_R)=\frac{f(u_L)+f(u_R)}2
 -\frac{\max(|u_L|,|u_R|)}2(u_R-u_L),
 \qquad f(u)=u^2/2.
\]

4. Flux differences form the conservative advective update.

The viscous term uses a centered second difference. The reaction term is evaluated
pointwise. Reflective/zero-normal-gradient ghost filling is used by the current
one-dimensional testbed.

## Time integration and stability

SSP-RK3 advances the method of lines. Every internal step is limited by the most
restrictive of

\[
 \Delta t_{adv}=CFL\,\Delta x/(\max|u|+\epsilon),
\]

\[
 \Delta t_{diff}=0.45\,\Delta x^2/\nu,
 \qquad
 \Delta t_{react}=0.25/k.
\]

The solver takes as many internal stable steps as necessary to land exactly on
each requested snapshot time. `Nt_save` is therefore the number of saved states,
not the number of RK steps.

## Regime diagnostics and coefficient-aware labels

A trajectory is summarized by:

- maximum state value,
- maximum spatial gradient over time,
- shock/front location from `argmax |du/dx|`,
- front speed from a least-squares fit over active-gradient times,
- run-up times based on state and gradient thresholds.

A single global gradient threshold would make coefficient-mismatch cases collapse
into one label. The builder therefore performs a preliminary calibration:

1. sample `(nu,k,E)` over nominal and mismatch ranges,
2. divide this space into coefficient buckets,
3. simulate many calibration cases,
4. store the 20th and 90th percentiles of peak gradient in each bucket,
5. label later cases relative to the closest populated bucket.

The resulting labels are proxy classes:

- `no_detonation`,
- `deflagration_like`,
- `detonation_like`.

They describe relative wave-strength regimes of the analogue equation, not a
chemical detonation classification validated against experiment.

## Dataset splits

- `train`: nominal profile (`b_quad=0`) and nominal coefficient family.
- `val`: same generating family, independent random cases; used for model
  selection and risk calibration.
- `test_profile_ood`: nonlinear temperature curvature `b_quad` is outside the
  training profile family.
- `test_mismatch_ood`: viscosity/reaction coefficients differ from the nominal
  training family.

The final arrays have shape `(Ncases,Nt,Nx)`. `meta.csv` stores the parameters,
regime, diagnostics, actual end time, and saved-step `dt` for each case.

## Conversion to supervised windows

For history length `H`, each trajectory produces

\[
 (u_{t-H+1},\ldots,u_t;\,x,\theta)\mapsto u_{t+1},
\]

where `theta=(nu,k,E,dTdx,b_quad,dt,L_mm)`. The training loader transposes the
history to `(Nx,H)` so each grid point carries its temporal feature vector.

## Verification expected before reporting results

A publication-quality experiment should report:

- grid-convergence checks against a finer reference grid,
- CFL/time-step sensitivity,
- conservation and residual histories,
- split-wise parameter and diagnostic distributions,
- label counts and bucket occupancy,
- multiple random seeds,
- confirmation that OOD trajectories differ dynamically, not only in metadata.

`verify_ood.py` provides the basic parameter and trajectory-diagnostic checks; it
is a sanity check rather than a formal convergence study.
