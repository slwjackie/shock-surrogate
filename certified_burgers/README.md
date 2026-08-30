# Certified Burgers PoC

Minimal theory testbed for the research question:

> Can a cheap observable score predict or certify the local advice error
> `eta_t = ||P_theta(u_t) - S^H(u_t)||_1` without executing trusted solver `S^H`?

This package is intentionally separate from the reactive Burgers/WENO pipeline.

## Reference problem

- PDE: `u_t + (u^2/2)_x = 0` (1-D inviscid Burgers)
- trusted step `S`: first-order exact Godunov finite volume
- default boundary: periodic
- fixed CFL-safe time step from a declared global state envelope
- numerical L1 non-expansiveness sweep over grids and CFL values

## Advice model

`TinyConvSurrogate` is deliberately small; architecture novelty is not the goal.
Use `--horizon H` to train one neural call against `H` Godunov steps:

`P_H(u_t) ~= S^H(u_t)`.

## Oracle and cheap verifier scores

For research diagnostics, the experiment computes

`eta_t = ||P_H(u_t) - S^H(u_t)||_1`.

This is an oracle quantity and is not deployable because it requires `S^H`.
Three cheap observable scores are audited:

1. conservation defect,
2. smooth-test-function weak residual proxy,
3. MC-dropout uncertainty.

They are **proxies, not certificates** until a deterministic or probabilistic theorem is proved.
The experiment also reports an empirical calibration scale `C` for `eta <= C q`; this is only a
coverage diagnostic and must not be presented as a rigorous certificate.

## Autoregressive trust-or-fallback evaluation

The current branch now performs true trajectories

`u_0 -> decision -> u_1 -> decision -> ... -> u_T`

and compares five baselines on the same initial conditions:

1. Always Solver,
2. Always Neural,
3. Oracle Gate,
4. Residual Gate,
5. Uncertainty Gate.

For every diagnostic rollout it evaluates the research-note inequality

`||u_T^ALG - u_T^S||_1 <= sum_{t in A} eta_t`

and records the full observed-deviation and cumulative-accepted-error curves.
Residual and uncertainty gates use thresholds calibrated on a separate calibration split.

## Threshold sweeps and OOD

Oracle/residual/uncertainty thresholds are swept to produce error-vs-fallback curves.
Tests are separated into ID and OOD sets. OOD includes stronger jumps, higher-frequency smooth
profiles, and three-state shock/rarefaction interaction patterns not used for training.
Shock-heavy samples are marked using a compressive Burgers jump sensor.

## Runtime measurements

The experiment measures actual wall-clock time for each complete rollout policy and separately benchmarks:

- one `S^H` Godunov macro-step,
- one neural prediction,
- one weak-residual verification,
- one MC-dropout uncertainty verification.

Oracle Gate's measured runtime includes computing `S^H` to reveal `eta`, so it is deliberately
non-deployable and need not be faster than Always Solver.

## Run

```bash
python -m certified_burgers.experiment \
  --epochs 60 \
  --horizon 1 \
  --out outputs/certified_burgers_poc.json
```

Multi-step leap:

```bash
python -m certified_burgers.experiment \
  --epochs 60 \
  --horizon 4 \
  --out outputs/certified_burgers_h4.json
```

The run writes JSON summaries plus PNG figures for:

- `q_t` versus oracle `eta_t` (ID/OOD, shock-heavy points marked),
- final L1 error versus fallback rate,
- final L1 error versus actual wall-clock speedup,
- observed trajectory deviation versus the Theorem-1 oracle upper bound.

If Oracle Gate itself gives no meaningful error/fallback tradeoff, do not over-invest in verifier theory.
If a cheap score tracks `eta` and retains coverage under OOD, the next research step is a restricted
deterministic or high-probability certificate theorem.
