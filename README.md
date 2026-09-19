# `certified-burgers-poc` branch

This branch is the active **1-D inviscid Burgers / first-order Godunov trust-or-fallback study**.
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
