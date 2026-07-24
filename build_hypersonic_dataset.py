#!/usr/bin/env python3
"""Generate strong-shock 2-D Euler/Navier--Stokes trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hypersonic.benchmarks import four_quadrant_riemann_2d, planar_hypersonic_shock_2d
from hypersonic.solver import CompressibleConfig, StructuredCompressibleSolver


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/hypersonic_2d.npz")
    p.add_argument("--cases", type=int, default=32)
    p.add_argument("--nx", type=int, default=128)
    p.add_argument("--ny", type=int, default=96)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--sample_dt", type=float, default=2.5e-4)
    p.add_argument("--mach_min", type=float, default=5.0)
    p.add_argument("--mach_max", type=float, default=15.0)
    p.add_argument("--angle_min", type=float, default=-25.0)
    p.add_argument("--angle_max", type=float, default=25.0)
    p.add_argument("--equation", choices=["euler", "navier_stokes"], default="euler")
    p.add_argument("--viscosity", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    cfg = CompressibleConfig(equation=args.equation, viscosity=args.viscosity, bc_x="outflow", bc_y="outflow")
    solver = StructuredCompressibleSolver(cfg)
    dx, dy = 1.0 / args.nx, 1.0 / args.ny
    trajectories, metadata = [], []
    for case in range(args.cases):
        if case % 5 == 4:
            U = four_quadrant_riemann_2d(args.nx, args.ny)
            mach, angle, case_type = np.nan, np.nan, "four_quadrant"
        else:
            mach = float(rng.uniform(args.mach_min, args.mach_max))
            angle = float(rng.uniform(args.angle_min, args.angle_max))
            offset = float(rng.uniform(0.35, 0.65))
            U = planar_hypersonic_shock_2d(args.nx, args.ny, mach=mach, shock_normal_angle_deg=angle, offset=offset)
            case_type = "planar_hypersonic_shock"
        snapshots = [U.squeeze(0).cpu().numpy().astype(np.float32)]
        substeps = 0
        for _ in range(args.steps):
            U, count = solver.advance(U, args.sample_dt, dx, dy)
            substeps += count
            snapshots.append(U.squeeze(0).cpu().numpy().astype(np.float32))
        trajectories.append(np.stack(snapshots, axis=0))
        metadata.append((case_type, mach, angle, substeps))
        print(f"case {case + 1}/{args.cases}: {case_type} mach={mach} angle={angle} substeps={substeps}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        u=np.stack(trajectories, axis=0),
        case_type=np.asarray([m[0] for m in metadata]),
        mach=np.asarray([m[1] for m in metadata], dtype=np.float32),
        angle=np.asarray([m[2] for m in metadata], dtype=np.float32),
        solver_substeps=np.asarray([m[3] for m in metadata], dtype=np.int64),
        sample_dt=np.asarray(args.sample_dt, dtype=np.float32),
        dx=np.asarray(dx, dtype=np.float32),
        dy=np.asarray(dy, dtype=np.float32),
        equation=np.asarray(args.equation),
        viscosity=np.asarray(args.viscosity, dtype=np.float32),
    )
    print(f"Wrote {out} with shape {(args.cases, args.steps + 1, 4, args.ny, args.nx)}")


if __name__ == "__main__":
    main()
