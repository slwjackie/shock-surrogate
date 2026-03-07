#!/usr/bin/env python3
"""
Dataset for Hybrid Temporal+Spatial Transformer.

Produces samples:
  - x: (Nx,)
  - u_hist: (Nx, H)
  - u_next: (Nx,)
  - u_last: (Nx,) (last snapshot in history window)
  - regime_id: int (case-level)
  - params: dTdx, b_quad, nu, k, E, dt (per-case when available)

Assumes NPZ arrays store per-case full trajectories:
  u_* shape: (Ncases, Nt, Nx)

Notes:
  - This dataset assumes that the per-split NPZ (e.g., data/u_train.npz) is ordered
    consistently with the filtered meta.csv rows for that split (after sorting by case_id).
  - x defaults to [0, 1] which matches solver_burgers_weno.py returning normalized x.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _load_npz(path: str) -> np.ndarray:
    d = np.load(path)
    if "u" in d:
        return d["u"]
    return d[list(d.keys())[0]]


class HybridTemporalDataset(Dataset):
    def __init__(self, meta_csv: str, u_npz: str, split: str, H: int = 5,
                 dt: float | None = None, stride: int = 1):
        self.meta = pd.read_csv(meta_csv)
        self.meta = self.meta[self.meta["split"] == split].copy()
        self.meta.sort_values("case_id", inplace=True)
        self.meta.reset_index(drop=True, inplace=True)

        self.u = _load_npz(u_npz)  # (Ncases, Nt, Nx)
        if self.u.ndim != 3:
            raise ValueError(f"Expected u to be (Ncases,Nt,Nx), got {self.u.shape}")
        self.Ncases, self.Nt, self.Nx = self.u.shape

        if len(self.meta) != self.Ncases:
            raise ValueError(
                f"Meta rows for split='{split}' ({len(self.meta)}) != Ncases in '{u_npz}' ({self.Ncases}). "
                "Check that you are passing the matching per-split NPZ and meta.csv."
            )

        self.H = int(H)
        self.stride = int(stride)

        # dt: prefer per-row dt in __getitem__; keep a fallback here
        if dt is None:
            if "dt" in self.meta.columns:
                dt = float(self.meta["dt"].iloc[0])
            else:
                dt = 1.0 / max(self.Nt - 1, 1)
        self.dt_fallback = float(dt)

        # x grid defaults to [0,1] (solver normalized x)
        if "x_min" in self.meta.columns and "x_max" in self.meta.columns:
            x_min = float(self.meta["x_min"].iloc[0])
            x_max = float(self.meta["x_max"].iloc[0])
        else:
            x_min, x_max = 0.0, 1.0
        self.x = np.linspace(x_min, x_max, self.Nx, dtype=np.float32)

        # build indices (case_idx, t0, meta_row_idx)
        self.indices: list[tuple[int, int, int]] = []
        for i in range(self.Ncases):
            ci = i  # split-local alignment
            for t0 in range(0, self.Nt - self.H, self.stride):
                self.indices.append((ci, t0, i))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        ci, t0, mi = self.indices[idx]
        row = self.meta.iloc[mi]

        u_hist = self.u[ci, t0:t0 + self.H, :]  # (H, Nx)
        u_next = self.u[ci, t0 + self.H, :]     # (Nx,)
        u_last = u_hist[-1, :]                  # (Nx,)

        x = torch.from_numpy(self.x).float()
        u_hist = torch.from_numpy(u_hist.T).float()
        u_next = torch.from_numpy(u_next).float()
        u_last = torch.from_numpy(u_last).float()

        regime_id = int(row["regime_id"]) if "regime_id" in row else 0

        params = {
            "dTdx": float(row["dTdx"]) if "dTdx" in row else 0.0,
            "b_quad": float(row["b_quad"]) if "b_quad" in row else 0.0,
            "nu": float(row["nu"]) if "nu" in row else 0.0,
            "k": float(row["k"]) if "k" in row else 0.0,
            "E": float(row["E"]) if "E" in row else 0.0,
            "dt": float(row["dt"]) if "dt" in row else self.dt_fallback,
        }

        return x, u_hist, u_next, u_last, regime_id, params, t0
