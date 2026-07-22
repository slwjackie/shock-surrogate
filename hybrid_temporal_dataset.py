#!/usr/bin/env python3
"""Windowed and case-level access to generated shock trajectories."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from backbones.conditioning import PARAMETER_KEYS


def _load_npz(path: str) -> np.ndarray:
    data = np.load(path)
    if "u" in data:
        return data["u"]
    return data[list(data.keys())[0]]


class HybridTemporalDataset(Dataset):
    def __init__(
        self,
        meta_csv: str,
        u_npz: str,
        split: str,
        H: int = 5,
        dt: float | None = None,
        stride: int = 1,
    ):
        self.meta = pd.read_csv(meta_csv)
        self.meta = self.meta[self.meta["split"] == split].copy()
        self.meta.sort_values("case_id", inplace=True)
        self.meta.reset_index(drop=True, inplace=True)
        self.split = split

        self.u = _load_npz(u_npz)
        if self.u.ndim != 3:
            raise ValueError(f"Expected u with shape (Ncases,Nt,Nx), got {self.u.shape}")
        self.Ncases, self.Nt, self.Nx = self.u.shape
        if len(self.meta) != self.Ncases:
            raise ValueError(
                f"Meta rows for split='{split}' ({len(self.meta)}) != cases in {u_npz} ({self.Ncases})"
            )

        self.H = int(H)
        self.stride = int(stride)
        if self.H <= 0 or self.H >= self.Nt:
            raise ValueError(f"H must satisfy 0 < H < Nt={self.Nt}, got H={self.H}")
        if self.stride <= 0:
            raise ValueError("stride must be positive")

        if dt is None:
            dt = float(self.meta["dt"].iloc[0]) if "dt" in self.meta.columns else 1.0 / max(self.Nt - 1, 1)
        self.dt_fallback = float(dt)

        if "x_min" in self.meta.columns and "x_max" in self.meta.columns:
            x_min = float(self.meta["x_min"].iloc[0])
            x_max = float(self.meta["x_max"].iloc[0])
        else:
            x_min, x_max = 0.0, 1.0
        self.x = np.linspace(x_min, x_max, self.Nx, dtype=np.float32)

        self.indices: list[tuple[int, int, int]] = []
        for case_index in range(self.Ncases):
            for t0 in range(0, self.Nt - self.H, self.stride):
                self.indices.append((case_index, t0, case_index))

    def __len__(self) -> int:
        return len(self.indices)

    def _row_params(self, row: pd.Series) -> dict[str, float]:
        defaults = {
            "nu": 0.0,
            "k": 0.0,
            "E": 0.0,
            "dTdx": 0.0,
            "b_quad": 0.0,
            "dt": self.dt_fallback,
            "L_mm": 20.0,
        }
        return {
            key: float(row[key]) if key in row.index and pd.notna(row[key]) else default
            for key, default in defaults.items()
        }

    def __getitem__(self, idx: int):
        case_index, t0, meta_index = self.indices[idx]
        row = self.meta.iloc[meta_index]
        history = self.u[case_index, t0 : t0 + self.H, :]
        next_state = self.u[case_index, t0 + self.H, :]
        last_state = history[-1]

        x = torch.from_numpy(self.x).float()
        history_tensor = torch.from_numpy(history.T.copy()).float()
        next_tensor = torch.from_numpy(next_state.copy()).float()
        last_tensor = torch.from_numpy(last_state.copy()).float()
        regime_id = int(row["regime_id"]) if "regime_id" in row.index else 0
        return (
            x,
            history_tensor,
            next_tensor,
            last_tensor,
            regime_id,
            self._row_params(row),
            t0,
        )

    def case_params(self, case_index: int) -> dict[str, float]:
        return self._row_params(self.meta.iloc[int(case_index)])

    def case_trajectory(self, case_index: int) -> np.ndarray:
        return self.u[int(case_index)]

    def case_regime_id(self, case_index: int) -> int:
        row = self.meta.iloc[int(case_index)]
        return int(row["regime_id"]) if "regime_id" in row.index else 0

    def parameter_matrix(self, keys: Sequence[str] = PARAMETER_KEYS) -> np.ndarray:
        rows = []
        for i in range(self.Ncases):
            params = self.case_params(i)
            rows.append([params[key] for key in keys])
        return np.asarray(rows, dtype=np.float64)
