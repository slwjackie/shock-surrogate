# Hypersonic Compressible-Flow Extension

This package extends the original one-dimensional Burgers testbed without replacing it. It provides an executable path toward **Robust Learning-Augmented Local–Global Neural Operator for Hypersonic Shock Dynamics**.

## Implemented

- 1-D, 2-D, and small-scale 3-D compressible Euler finite-volume fallback solver
- optional laminar compressible Navier–Stokes viscous stress and heat conduction
- Rusanov flux, SSP-RK3, CFL and diffusion time-step control
- density/pressure positivity-preserving global convex blending
- Mach 5–15 planar oblique-shock data generation and four-quadrant Riemann cases
- LGNO-style 2-D operator:
  - global low-mode Fourier branch
  - local coarse-to-fine multiresolution convolution branch
  - multiplicative local/global coupling
  - pointwise residual fusion
  - mean-zero conservative increment on periodic domains
  - componentwise relative L1 plus high-frequency spectral loss
- dependency-light conservative MeshGraphNet-style backbone for unstructured 2-D or 3-D coordinates

## Scope statement

This is an executable research baseline, not a production hypersonic CFD code. The structured solver is first-order in space and intended as a robust fallback, smoke-test reference, and dataset bootstrapper. Publication-quality shock resolution should replace its piecewise-constant interface states with WENO-Z or another validated high-order positivity-preserving reconstruction. Likewise, 3-D tests validate software and physical admissibility on small grids; they are not DNS or high-fidelity shock/boundary-layer validation.

## Generate data

```bash
python build_hypersonic_dataset.py \
  --cases 32 --nx 128 --ny 96 --steps 20 \
  --mach_min 5 --mach_max 15 \
  --out data/hypersonic_2d.npz
```

For the laminar Navier–Stokes path:

```bash
python build_hypersonic_dataset.py \
  --equation navier_stokes --viscosity 1e-4 \
  --out data/hypersonic_ns_2d.npz
```

## Train LGNO

```bash
python train_lgno2d.py \
  --data data/hypersonic_2d.npz \
  --width 64 --modes_x 16 --modes_y 16 --depth 4 \
  --boundary outflow_learned \
  --epochs 100
```

## Evaluate rollout

```bash
python eval_lgno2d.py \
  --data data/hypersonic_2d.npz \
  --checkpoint ckpt/lgno2d_hypersonic.pt \
  --case 0
```

## Relation to RCCP

The LGNO and Mesh-GNN are advice backbones. The existing residual-calibrated clamp controller should next receive a compressible-state risk adapter using Euler/Navier–Stokes discrete residual, density/pressure positivity margin, entropy and shock-position diagnostics, surrogate ensemble uncertainty, and coefficient/Mach-number OOD score. The expensive fallback action becomes `StructuredCompressibleSolver.advance`.
