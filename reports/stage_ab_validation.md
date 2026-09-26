# Stage A/B validation report

## Executed checks

- Full local regression: **47 passed**, 0 failures, 0 errors, 0 skips (8.572 s).
- Original baseline: 19 tests passed in GitHub Actions before extending the source.
- Stage A: 200 training epochs, 16-bin certificate table, 128 cells, 24 steps, 8 ID + 8 OOD trajectories.
- All checked Stage A local/global certificate inequalities held; six certified tolerances were swept.
- Stage B integration: 60 epochs, 5 training cases, 40 reference samples/case, 12 rollout steps, 5 independent ID/OOD cases and 8 policies.
- Stage B interval smoke: all 5 ID/OOD cases completed with the additional restricted interval-ODE gate.
- Exact numerical RHS, conservation, frozen/reflected shock jumps, NASA branch rejection, interval-AD derivatives, all-slab fail-closed audit, checkpoint validation and plot/CSV generation were exercised.

## Measured results (not performance claims)

| Stage A cohort | Certified acceptance | Speedup vs Godunov | Mean final L1 error |
|---|---:|---:|---:|
| id | 0.250 | 0.0193x | 0.000683619 |
| ood | 0.000 | 0.0130x | 0 |

The Stage A certificate is sound under its documented arithmetic contract but is **not faster** than vectorized scalar Godunov. This is a certification testbed, not an acceleration result.

At the tested H2 local-error tolerance (0.05 in the declared component-scaled norm), residual, consistency and uncertainty gates rejected every proposal in the 60-epoch integration rollout. They matched restarted Cantera but were slower than the fastest classical baseline. The guard-limited neural-only model missed ignition in the equivalence-ratio OOD case. These are recorded failure/limitation results, not suppressed observations.

The restricted interval probe at 1100 K, 1 bar and dt=1e-10 s closed its tube with a weighted local exact-ODE upper bound 1.6794129e-16. This very short step does not demonstrate useful certification through a full ignition event.

## Not completed / not claimed

The larger `h2_full.json` performance run exceeded the execution limit. Completed integration and smoke evaluations must not be confused with that full sweep. No multi-seed final study, second-mechanism validation, experimental validation, reactive Euler implementation, or universal full-trajectory H2 certificate is claimed.

The workflow `.github/workflows/stage-ab.yml` repeats regression, Stage A smoke, Stage B interval smoke and plotting on GitHub. Its job/artifact status is the remote execution record. Local measured numbers above are preserved in `stage_ab_validation.json`, with source fingerprints.

## Environment

Python 3.13.5, NumPy 2.3.5, PyTorch 2.10.0+cpu, Cantera 3.2.0; CPU, one compute thread.

See [the run guide](../docs/stage_ab_workflow.md) and [the mathematical contracts](../docs/stage_ab_theory.md).
