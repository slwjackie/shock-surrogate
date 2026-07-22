# Robust Learning-Augmented Shock Surrogates

This repository studies **learning-augmented simulation for shock-dominated PDEs**. A neural surrogate provides inexpensive one-step advice, while a risk-aware controller may accept that advice, apply a conservative correction, or invoke a numerical solver fallback.

The repository now contains two deliberately separated research tracks:

1. **Controlled algorithmic testbed** — one-dimensional reactive Burgers dynamics with parameter-conditioned Transformer/FNO advice and the Residual-Calibrated Clamp Policy (RCCP).
2. **Compressible-flow extension** — one-, two-, and small-scale three-dimensional Euler/Navier–Stokes states, positivity preservation, a two-dimensional Local–Global Neural Operator, and an unstructured mesh GNN.

The original Burgers pipeline is preserved. The compressible extension is additive and does not silently replace the established baselines.

## Research question

Can a learned surrogate remain inexpensive when its advice is accurate while avoiding catastrophic long-rollout failure under strong shocks, coefficient shifts, and geometric or mesh distribution shifts?

The project separates:

- **advice quality** — Transformer, FNO, Local–Global Neural Operator, or mesh GNN;
- **advice control** — calibrated accept/correct/fallback decisions;
- **reference physics** — conservative numerical solvers and physical admissibility checks.

This separation supports consistency–robustness–cost experiments without attributing every gain to a larger neural architecture.

## Pipeline A — reactive Burgers + RCCP

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
Residual-Calibrated Clamp Policy
   ├── accept surrogate
   ├── local residual projection
   └── WENO5 fallback
```

The controlled equation is

\[
 u_t + \left(\frac{u^2}{2}\right)_x
 = \nu u_{xx} + k(1-u)\exp(-E/T(x)).
\]

### Generate data

```bash
python sim/build_dataset.py --calibrate_by_coeff
python sim/build_dataset.py
```

### Train Transformer or FNO advice

```bash
python train_transformer_hybrid.py \
  --arch transformer_hybrid \
  --mode full \
  --physics_residual discrete \
  --epochs 1200
```

```bash
python train_transformer_hybrid.py \
  --arch fno1d \
  --mode full \
  --physics_residual discrete \
  --width 64 --modes 24 --depth 4 \
  --epochs 1200
```

### Calibrate and evaluate RCCP

```bash
python calibrate_residual_policy.py \
  --checkpoint ckpt/best_transformer_hybrid_full_seed0_H5.pt \
  --out outputs/calibration_transformer_full_seed0.json
```

```bash
python eval_policy.py \
  --checkpoint ckpt/best_transformer_hybrid_full_seed0_H5.pt \
  --calibration outputs/calibration_transformer_full_seed0.json \
  --threshold_scale 1.0 \
  --out outputs/policy_rollout.json
```

## Pipeline B — compressible hypersonic extension

The `hypersonic/` package implements conservative perfect-gas states with:

- 1-D, 2-D, and small-scale 3-D Euler fluxes;
- optional laminar compressible Navier–Stokes viscous stress and heat conduction;
- Rusanov finite-volume fluxes and SSP-RK3 time advancement;
- advective and diffusive time-step constraints;
- density and thermodynamic-pressure positivity preservation;
- Mach 5–15 planar oblique-shock and interacting Riemann benchmarks;
- a 2-D Local–Global Neural Operator;
- a conservative MeshGraphNet-style model for unstructured 2-D/3-D coordinates.

### Positivity and conservation

A candidate state is blended toward a known admissible reference state:

```text
U_safe = U_reference + theta * (U_candidate - U_reference),  0 <= theta <= 1
```

One scalar `theta` is selected per batch item or graph so every cell has positive density and pressure. When a periodic neural increment has zero spatial mean, the global blend also preserves the mean of every conserved component.

### Local–Global Neural Operator

Each LGNO layer contains:

```text
hidden state
  ├── global low-mode Fourier branch
  ├── local coarse-to-fine convolution branch
  ├── multiplicative global/local coupling
  └── residual pointwise fusion
```

Training combines componentwise relative L1 prediction error with a high-frequency spectral error penalty. Periodic updates are projected to zero spatial mean before positivity enforcement.

### Generate Mach-number-varied data

```bash
python build_hypersonic_dataset.py \
  --cases 32 --nx 128 --ny 96 --steps 20 \
  --mach_min 5 --mach_max 15 \
  --out data/hypersonic_2d.npz
```

Laminar Navier–Stokes option:

```bash
python build_hypersonic_dataset.py \
  --equation navier_stokes --viscosity 1e-4 \
  --out data/hypersonic_ns_2d.npz
```

### Train and evaluate LGNO

```bash
python train_lgno2d.py \
  --data data/hypersonic_2d.npz \
  --width 64 --modes_x 16 --modes_y 16 --depth 4 \
  --boundary outflow_learned \
  --epochs 100
```

```bash
python eval_lgno2d.py \
  --data data/hypersonic_2d.npz \
  --checkpoint ckpt/lgno2d_hypersonic.pt \
  --case 0
```

See [`hypersonic/README.md`](hypersonic/README.md) for implementation scope and commands.

## Repository layout

```text
backbones/       1-D Transformer/FNO advice registry
physics/         Burgers discrete residual and local projection
calibration/     empirical risk calibration and OOD scoring
policies/        RCCP accept/correct/fallback policy
solvers/         Burgers WENO fallback adapter
evaluation/      rollout error and cost metrics

hypersonic/
├── state.py             conservative/primitive variables and Euler fluxes
├── positivity.py        density/pressure admissibility and convex limiting
├── solver.py            1-D/2-D/3-D Euler/Navier–Stokes fallback solver
├── benchmarks.py        strong-shock and Mach-number-varied initial states
├── losses.py            physical and high-frequency LGNO losses
└── models/
    ├── lgno2d.py         local–global structured-grid operator
    └── mesh_gnn.py       conservative unstructured-mesh GNN
```

## Tests

```bash
pip install -r requirements.txt
pytest
```

Tests cover the original learning-augmented pipeline plus:

- primitive/conservative state round trips;
- density/pressure positivity repair;
- uniform Euler/Navier–Stokes preservation in 1-D and 3-D;
- finite and admissible Mach-10 2-D and Mach-6 3-D shock evolution;
- LGNO output shape, positivity, and periodic conservation;
- mesh-GNN graph conservation;
- high-frequency spectral training loss.

## Scientific scope

The compressible solver is an **executable research baseline and fallback**, not a production hypersonic CFD package. Its present interface reconstruction is first-order and intentionally robust. Publication-quality claims require:

- WENO-Z/TENO or another validated high-order reconstruction;
- high-order positivity-preserving flux limiting;
- grid-convergence and benchmark studies;
- realistic wall boundary conditions and transport models;
- high-fidelity 2-D/3-D datasets;
- measured runtime and memory comparisons;
- controller calibration for compressible residual, entropy, positivity margin, shock position, uncertainty, and Mach/geometry OOD.

Accordingly, current claims should distinguish **implemented software capability** from **validated high-fidelity hypersonic performance**.
