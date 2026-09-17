# Backbone-Agnostic Robust Learning-Augmented Shock Surrogate

This repository studies a **learning-augmented controller for shock-like PDE
surrogates**.  A neural model supplies inexpensive advice for the next field,
while a distributionally calibrated online policy decides whether to:

1. accept the surrogate prediction,
2. apply a local conservative residual projection, or
3. fall back to the WENO5 solver.

The first testbed is a one-dimensional viscous Burgers equation with a reaction
source term.  It is a controlled detonation analogue, not yet a full hypersonic
Euler/Navier–Stokes solver.

## Research question

Can a surrogate remain inexpensive when its advice is accurate while avoiding
catastrophic long-rollout failures under profile and coefficient distribution
shifts?

The project separates two concerns:

- **Advice quality:** Transformer or FNO one-step prediction.
- **Advice control:** the Residual-Calibrated Clamp Policy (RCCP), which is
  independent of the chosen backbone.

This separation supports consistency–robustness–cost experiments without
confounding every algorithmic change with a new neural architecture.

## Implemented pipeline

```text
state history + PDE coefficients
              │
              ▼
 parameter-conditioned advice backbone
    ├── temporal/spatial Transformer
    └── 1-D Fourier Neural Operator
              │
              ▼
 discrete conservative residual + risk diagnostics
              │
              ▼
 empirical distributional calibration
              │
              ▼
 Residual-Calibrated Clamp Policy (RCCP)
    ├── accept surrogate
    ├── local residual projection
    └── WENO5 fallback
```

## Governing test equation

The data generator solves

\[
  u_t + \left(\frac{u^2}{2}\right)_x
  = \nu u_{xx} + k(1-u)\exp(-E/T(x)),
\]

with

\[
  T(x)=1+0.35\,dTdx\,x+0.40\,b_{quad}x^2.
\]

Numerical trajectories are generated using:

- WENO5 reconstruction,
- Rusanov flux,
- SSP-RK3 time integration, and
- an explicit stability limit for advection, diffusion, and reaction.

## What changed from the original version

### 1. Parameter-conditioned backbones

The dataset already stored `nu`, `k`, `E`, `dTdx`, `b_quad`, and `dt`, but the
original predictor received only `x` and `u_hist`.  All backbones now use a shared
parameter-conditioning interface.  `L_mm` is also supplied so spatial residuals
use the physical rather than normalized grid spacing.

### 2. Discrete conservative physics loss

The default physics loss no longer differentiates a grid surrogate twice with
respect to the coordinate input.  It computes a differentiable finite-volume-like
residual using a Rusanov flux divergence and a Neumann finite-difference
Laplacian.  High-gradient cells receive a detached shock-aware weight.

The legacy continuous-coordinate autograd residual remains available as an
ablation:

```bash
--physics_residual autograd
```

The default is:

```bash
--physics_residual discrete
```

### 3. Distributional residual calibration

Validation predictions are converted into empirical distributions for:

- discrete physics residual,
- total-variation change,
- shock-position shift,
- coefficient OOD distance, and
- optional predictive uncertainty.

The calibrator maps each heterogeneous diagnostic to an empirical percentile and
combines them into one risk score.  It stores global and, when sufficiently
populated, regime-conditional thresholds.

### 4. Residual-Calibrated Clamp Policy

RCCP applies two calibrated thresholds:

```text
risk <= tau_low              accept surrogate

tau_low < risk <= tau_high  local residual correction

risk > tau_high              WENO fallback
```

A raw-tail safety guard also triggers fallback when the residual or coefficient
shift lies far outside validation support.

### 5. Local conservative correction

The middle action performs a small trust-region residual projection only around
cells selected by a shock sensor.  It is intentionally cheaper than a full WENO
fallback and does not require a GNN on the present uniform one-dimensional grid.

### 6. Interchangeable advice backbones

Both implemented backbones expose the same call:

```python
u_next, logits = model(x, state_history, parameters)
```

Supported names:

- `transformer_hybrid`
- `fno1d`

A later local–global neural operator or mesh GNN can be registered without
changing calibration, policy, solver fallback, or rollout evaluation.

## Repository layout

```text
backbones/
├── base.py                  common interface
├── conditioning.py          shared physical-parameter FiLM conditioning
├── transformer.py           Transformer factory
├── fno.py                   parameter-conditioned FNO1d
└── registry.py              backbone registry

physics/
├── discrete_residual.py     conservative residual and risk components
└── residual_projection.py   local shock-region correction

calibration/
└── residual_calibrator.py   empirical percentile calibration and OOD scoring

policies/
└── clamp_policy.py          RCCP accept/correct/fallback decisions

solvers/
└── weno_adapter.py          Torch-facing fallback adapter

evaluation/
└── rollout_policy_eval.py   long-rollout and cost metrics
```

The original dataset builder, solver, one-step trainer/evaluator, OOD verifier,
and experiment runner remain available. Existing datasets remain compatible.
Because parameter conditioning changes the model state dictionary, checkpoints
trained by the pre-RCCP code must be retrained; they are not silently loaded as
if they represented the new architecture.

## Installation

```bash
pip install -r requirements.txt
```

Core dependencies are NumPy, pandas, and PyTorch.  Pytest is needed only for the
test suite.

## Dataset generation

Coefficient-aware regime thresholds must be calibrated first:

```bash
python sim/build_dataset.py --calibrate_by_coeff
```

Then generate the data:

```bash
python sim/build_dataset.py
```

Expected outputs:

```text
data/
├── meta.csv
├── grid.npz
├── thresholds_by_coeff.json
├── u_train.npz
├── u_val.npz
├── u_test_profile_ood.npz
└── u_test_mismatch_ood.npz
```

## Train an advice backbone

### Parameter-conditioned Transformer

```bash
python train_transformer_hybrid.py \
  --arch transformer_hybrid \
  --mode full \
  --physics_residual discrete \
  --epochs 1200
```

### Parameter-conditioned FNO

```bash
python train_transformer_hybrid.py \
  --arch fno1d \
  --mode full \
  --physics_residual discrete \
  --width 64 \
  --modes 24 \
  --depth 4 \
  --epochs 1200
```

Available modes retain the original ablation semantics:

| mode | losses/settings |
|---|---|
| `full` | data + discrete physics + TV matching + classification |
| `no_causal` | Transformer without a causal temporal mask |
| `no_phys` | data + TV matching + classification |
| `data_only` | data + classification |

Unlike the old TV magnitude penalty, the new objective matches the target total
variation and therefore does not systematically erase shocks.

## One-step evaluation

```bash
python eval_transformer_hybrid.py \
  --arch transformer_hybrid \
  --mode full \
  --save_metrics
```

Metrics include:

- MSE, RMSE, and MAE,
- regime accuracy,
- discrete physics-residual MAE, and
- peak-gradient error.

## Calibrate the advice-risk distribution

```bash
python calibrate_residual_policy.py \
  --checkpoint ckpt/best_transformer_hybrid_full_seed0_H5.pt \
  --out outputs/calibration_transformer_full_seed0.json
```

Calibration uses only the validation split.  The generated JSON contains the
component empirical quantiles, low/high clamp thresholds, regime thresholds, and
training-parameter OOD statistics.

## Compare pure-surrogate and RCCP rollouts

```bash
python eval_policy.py \
  --checkpoint ckpt/best_transformer_hybrid_full_seed0_H5.pt \
  --calibration outputs/calibration_transformer_full_seed0.json \
  --threshold_scale 1.0 \
  --out outputs/policy_rollout.json
```

The evaluator reports:

- rollout and final-time errors,
- shock-position and peak-gradient errors,
- selected-state physics residual,
- accept/correct/fallback counts and rates,
- WENO internal substeps, and
- normalized computational cost.

`--threshold_scale` exposes the consistency–robustness trade-off:

- values below `1.0` are more conservative,
- values above `1.0` trust neural advice more.

The cost model is configurable with `--surrogate_cost`, `--correction_cost`, and
`--fallback_cost`.  These are normalized proxies; wall-clock measurements should
also be reported in final experiments.

## Multi-seed experiments and policy sweeps

Standard backbone ablations:

```bash
python run_experiments_hybrid.py \
  --arch transformer_hybrid \
  --modes full no_causal no_phys data_only \
  --seeds 0 1 2 3 4
```

End-to-end learning-augmented experiments:

```bash
python run_experiments_hybrid.py \
  --arch transformer_hybrid \
  --modes full no_phys data_only \
  --seeds 0 1 2 3 4 \
  --with_policy \
  --threshold_scales 0.75 1.0 1.25
```

Run the same command with `--arch fno1d` to test whether RCCP improvements are
backbone-independent.

## Tests

```bash
pytest
```

The tests cover:

- shared Transformer/FNO interfaces,
- the constant-state discrete residual,
- calibrated clamp decisions, and
- arbitrary-state WENO fallback.

## Recommended paper ablations

At minimum, compare:

| advice backbone | controller |
|---|---|
| Transformer | none |
| Transformer | RCCP |
| FNO | none |
| FNO | RCCP |

Additional controller ablations:

- no parameter conditioning,
- autograd residual versus discrete residual,
- no coefficient OOD component,
- no raw-tail guard,
- accept/fallback only,
- no local correction, and
- multiple threshold scales and cost ratios.

The central claim should concern the **controller**, not merely a stronger neural
operator: accurate advice should be used cheaply, while unreliable/OOD advice
should be clamped toward conservative correction or solver fallback.

## Scope and roadmap

The present implementation is Phase 1–2 infrastructure:

- a 1-D reactive Burgers proof-of-concept,
- parameter-conditioned Transformer and FNO advice,
- backbone-agnostic calibration and RCCP,
- local residual correction, and
- WENO fallback evaluation.

Not yet implemented:

- compressible Euler/Navier–Stokes state vectors,
- density/pressure positivity projection,
- a published LGNO reproduction,
- 2-D/3-D geometry,
- adaptive or unstructured meshes, and
- a shock-front GNN.

A GNN becomes technically justified when the project moves to unstructured or
adaptive meshes.  Until then, the local residual projection provides a controlled
and interpretable middle action without unnecessary architectural complexity.
