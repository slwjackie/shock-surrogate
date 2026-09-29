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

### Stage A2: difference-aware certificates and certified routing

`stage_ab/stage_a2.py`, `stage_ab/vinterval.py` and `stage_ab/budget.py` strengthen
Stage A (all claims remain relative to same-grid exact-arithmetic Godunov):

- **Difference-aware certificate** (table and local-box variants) bounding
  `e_i - e_(i-1)` instead of `|e_i| + |e_(i-1)|`: O(h) instead of Θ(1), and
  asymptotically exact on smooth data under the local variant.
- **Face-selective routing**: per-face NN/Godunov choice, exact trust frontier by
  an O(N²) dynamic program on the cycle, certified for every trust pattern.
- **Online budget allocation** of the global certified error (online
  multiple-choice knapsack): threshold policy with a proved competitive bound,
  robustification of any heuristic, lower bounds for greedy and pacing.

Proofs: [docs/stage_a2_theory.md](docs/stage_a2_theory.md).
Measurements: [reports/stage_a2_validation.md](reports/stage_a2_validation.md).

```bash
python -m stage_ab.experiments_a2 --out outputs/stage_a2      # ~5 min, CPU
python -m stage_ab.plots_a2 outputs/stage_a2
```

```bash
python -m pip install -r requirements-stage-ab.txt
python -m pytest -q
python -m stage_ab.experiments a --smoke --out outputs/stage_a
python -m stage_ab.experiments b --smoke --out outputs/stage_b
```
