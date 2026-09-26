# Stage A/B: run guide

Base branch: `certified-burgers-poc` at `93262e7`.
Implementation branch: `certified-stage-ab-h2`.
The old Burgers/legacy Transformer files are preserved.

## Installation and regression

Use Python 3.11–3.13 and a CPU environment for reproducible comparisons.

```bash
python -m pip install -r requirements-stage-ab.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q
```

The CLI fixes PyTorch's thread count at one. All timing excludes model training
and offline certificate construction but includes the online inference,
projection, EOS, verification and fallback costs. Offline costs are reported
separately. Oracle policies remain deliberately non-deployable.

## Stage A

```bash
python -m stage_ab.experiments a --smoke --out outputs/stage_a_smoke
python -m stage_ab.experiments a --epochs 200 --bins 16 --out outputs/stage_a
python -m stage_ab.plots outputs/stage_a
```

Outputs: trained frozen flux JSON, verified table JSON, result JSON, CSV,
accuracy/runtime and acceptance plots. The runner compares solver, neural,
oracle, conservation, residual, MC-dropout, calibrated empirical envelope, and
certified policies on separate ID/OOD initial states. Six certified error
budgets are swept. Original `certified_burgers.experiment` remains available
for the original state/flux CNN controls and H-sweeps.

Programmatic table construction supports `method='interval'`, `'lipschitz'`,
or `'hybrid'`, and exhaustive `subdivisions`. Loading normally revalidates the
table. Do not reuse a certificate with different weights or a different
inference implementation. `stage_ab.adapters` connects the new certificate to
the original `Proposal` / `VerifierResult` / `DecisionPolicy` contracts.

## Stage B

```bash
python -m stage_ab.experiments b --smoke --out outputs/stage_b_smoke
python -m stage_ab.experiments b --config configs/stage_ab/h2_integration.json --out outputs/stage_b_integration
python -m stage_ab.experiments b --config configs/stage_ab/h2_pilot.json --out outputs/stage_b_pilot
python -m stage_ab.experiments b --config configs/stage_ab/h2_full.json --out outputs/stage_b_full
python -m stage_ab.plots outputs/stage_b_integration
```

`h2_full.json` is a larger evaluation, not the default smoke test. It can be
expensive: each policy is diagnosed with numerical oracles and then benchmarked
separately. Completed case results are checkpointed under `cases/`; an
interrupted run must not be presented as a completed aggregate evaluation.

Train/calibration/test splitting is by complete initial-condition trajectory.
Temperature, pressure, equivalence-ratio and reflected-shock OOD groups are
separate. All feature normalization uses training trajectories only. Physical
units are K, Pa, kg/m^3, seconds, kg/kg and J/kg. Density is fixed within each
reactor trajectory, not across all experiments.

Each trained flow map predicts composition change; an element-constrained
nonnegative projection and constant-internal-energy temperature recovery form
part of the proposal. Their runtime is counted. The default gates are empirical
residual, step consistency and MC-dropout, with physical guards. Thresholds are
chosen only on calibration trajectories against a declared local-error
threshold; their risk figures are empirical, not conformal guarantees.

The scaled infinity norm uses 1000 K for temperature, 1 for major species, and
1e-4 for H/O/OH/HO2/H2O2. A 0.05 tolerance is thus a *pilot* error criterion,
not a certified combustion-accuracy specification. Inspect and tighten it for
the target study. Radical-species and ignition metrics must be reported in
addition to this norm.

The strongest classical baseline is the faster of persistent CVODES and the
restarted reference-map implementation. Do not report speedup against only an
artificially slow restarted baseline. `always_neural` still rejects invalid
states; its output is not an unrestricted neural-only trajectory.

Ignition is a first crossing of T0+400 K with interpolation; max-dT/dt time is
reported separately when the peak is internal. Unignited trajectories are
censored, not assigned a fake ignition delay at the final time. The coarse smoke
time grid is a functional test, not an ignition-delay accuracy study.

Outputs include reference trajectories NPZ with case IDs, mechanism hash and
species order, model JSON, calibration/held-out risk, trajectories, radical
species, pressure, heat release, fallback reasons, timings, CSV and plots.
A second mechanism can be passed via `--mechanism path/to/mechanism.yaml`;
models and mechanism fingerprints are intentionally not interchangeable.

## Restricted formal ODE extension

```bash
python -m stage_ab.experiments b --smoke --certified-ode --out outputs/stage_b_interval
```

This adds the expensive interval-ODE gate. It is allowed to reject every
practically sized slab. A separate 1e-10 s probe demonstrates a successfully
closed tube; it is not evidence of useful full-ignition speedup.

For an all-slab exact-ODE audit use `audit_complete_trajectory` from
`stage_ab.kinetics_bounds`. It includes incoming uncertainty and refuses a final
bound as soon as any slab fails. This prevents unvalidated Cantera fallback from
being silently treated as exact. `stiffness_diagnostic` gives finite-difference
spectral indicators only; those are never fed into a rigorous certificate.

## Before a thesis/performance claim

Freeze the mechanism and target thermochemical ranges; perform time-grid,
CVODES-tolerance and mechanism-dependence studies. Train beyond the smoke
budget, use multiple independent seeds, inspect censored ignition cases, and
report both useful acceptance and end-to-end cost. A valid but loose certificate
or a slower hybrid is a measured limitation, not a successful acceleration.
See `stage_ab_theory.md` for all mathematical and physical claim boundaries.
