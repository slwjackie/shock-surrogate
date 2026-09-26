# `certified-stage-ab-h2` branch

This branch extends `certified-burgers-poc` with certified Burgers fluxes and
0-D hydrogen autoignition. See the Stage A/B section below.

## Preserved Burgers PoC

The original package is the active **1-D inviscid Burgers / first-order Godunov trust-or-fallback study**.
The surrogate used by `certified_burgers/` is a deliberately small **local 1-D CNN**, not a
physics-informed Transformer. Architecture novelty is intentionally not the objective of this
proof-of-concept; the main research questions are oracle opportunity, cheap verification,
shock/OOD failure modes, and the error-runtime frontier.

## Active surrogate architectures

- `StateConvSurrogate`: direct state-to-state CNN. This is the non-conservative control needed
  to test whether conservation defect has any predictive value.
- `ConservativeFluxSurrogate`: local periodic CNN that predicts an average interface flux and
  applies a telescoping finite-volume update. It is the preferred model for conservative
  one-step and multi-step horizon studies.
- `TinyConvSurrogate` remains as a backward-compatible alias for `StateConvSurrogate`.

Both models use circular padding on the periodic domain. The conservative model uses a
dilated local receptive field for multi-step leaps and supports `[B, C, N]` tensors so the
interface can later be reused for multi-variable conservative systems.

See [`certified_burgers/README.md`](certified_burgers/README.md) for the mathematical
definitions, experiment protocol, commands, and claim boundaries.

## Recommended runs

Verifier calibration with the state-prediction control:

```bash
python -m certified_burgers.experiment --surrogate state --horizon 1 --reference_steps 32
```

Conservative one-/multi-step horizon study:

```bash
python -m certified_burgers.horizon_study \
  --surrogate flux \
  --horizons 1 2 4 8 16 \
  --reference_steps 32
```

## Legacy code

Files such as `train_transformer_hybrid.py`, `eval_transformer_hybrid.py`,
`models/model_hybrid_temporal_spatial.py`, and the reactive-Burgers/WENO scripts are retained
from an earlier project direction. They are **not used by the active `certified_burgers`
experiment** and should not be read as the architecture description for this branch.

## Stage A/B extension (new branch)

`certified-stage-ab-h2` preserves the original PoC and adds:

- **Stage A:** H=1 two-input ReLU flux, offline interval/Lipschitz certificates,
  roundoff-aware cumulative error budget and certified fallback.
- **Stage B:** Cantera H2/O2/N2 autoignition, frozen reflected-shock conditioning,
  physically guarded neural chemistry and independent ID/OOD evaluation.
- A restricted interval-AD ODE certificate is included; empirical chemistry gates
  and unvalidated CVODES fallback are not mislabelled as a global certificate.

Start with [the run guide](docs/stage_ab_workflow.md),
[mathematical contracts](docs/stage_ab_theory.md), and
[validation report](reports/stage_ab_validation.md).

```bash
python -m pip install -r requirements-stage-ab.txt
python -m pytest -q
python -m stage_ab.experiments a --smoke --out outputs/stage_a
python -m stage_ab.experiments b --smoke --out outputs/stage_b
```
