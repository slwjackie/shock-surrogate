"""Distributional calibration for residual-based learning-augmented advice."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from backbones.conditioning import PARAMETER_KEYS, transform_parameter_array

COMPONENT_NAMES = ("residual", "tv_growth", "shock_shift", "coeff_ood", "uncertainty")
DEFAULT_WEIGHTS = {
    "residual": 0.40,
    "tv_growth": 0.20,
    "shock_shift": 0.15,
    "coeff_ood": 0.20,
    "uncertainty": 0.05,
}


@dataclass
class ParameterOODScorer:
    mean: np.ndarray
    std: np.ndarray
    keys: tuple[str, ...] = PARAMETER_KEYS

    @classmethod
    def fit(cls, matrix: np.ndarray, keys: Sequence[str] = PARAMETER_KEYS):
        values = transform_parameter_array(np.asarray(matrix, dtype=np.float64))
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        floor = np.maximum(0.05, 0.05 * np.maximum(np.abs(mean), 1.0))
        return cls(mean=mean, std=np.maximum(std, floor), keys=tuple(keys))

    def score_matrix(self, matrix):
        values = transform_parameter_array(np.asarray(matrix, dtype=np.float64))
        z = (values - self.mean[None, :]) / self.std[None, :]
        return np.sqrt(np.mean(z * z, axis=1))

    def vector_from_mapping(self, params: Mapping[str, Any]):
        raw = []
        for key in self.keys:
            value = params.get(key, 20.0 if key == "L_mm" else 0.0)
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            arr = np.asarray(value, dtype=np.float64)
            arr = arr.reshape(1) if arr.ndim == 0 else arr.reshape(arr.shape[0], -1)[:, 0]
            raw.append(arr)
        batch = max(len(arr) for arr in raw)
        columns = []
        for arr in raw:
            if len(arr) == 1 and batch > 1:
                arr = np.repeat(arr, batch)
            if len(arr) != batch:
                raise ValueError("Parameter mapping contains incompatible batch dimensions")
            columns.append(arr)
        return np.stack(columns, axis=1)

    def score_mapping(self, params):
        return self.score_matrix(self.vector_from_mapping(params))

    def to_dict(self):
        return {"keys": list(self.keys), "mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, payload):
        return cls(
            np.asarray(payload["mean"], dtype=np.float64),
            np.asarray(payload["std"], dtype=np.float64),
            tuple(payload.get("keys", PARAMETER_KEYS)),
        )


class EmpiricalRiskCalibrator:
    def __init__(self, weights=None, q_low=0.80, q_high=0.95, grid_size=101):
        selected = dict(DEFAULT_WEIGHTS if weights is None else weights)
        selected = {name: float(selected.get(name, 0.0)) for name in COMPONENT_NAMES}
        total = sum(max(v, 0.0) for v in selected.values())
        if total <= 0:
            raise ValueError("At least one risk-component weight must be positive")
        self.weights = {k: max(v, 0.0) / total for k, v in selected.items()}
        self.q_low, self.q_high = float(q_low), float(q_high)
        if not 0 <= self.q_low <= self.q_high <= 1:
            raise ValueError("Expected 0 <= q_low <= q_high <= 1")
        self.quantile_grid = np.linspace(0.0, 1.0, int(grid_size), dtype=np.float64)
        self.component_quantiles: dict[str, np.ndarray] = {}
        self.global_thresholds: tuple[float, float] | None = None
        self.group_thresholds: dict[str, tuple[float, float]] = {}
        self.ood_scorer: ParameterOODScorer | None = None

    def _component_percentile(self, name, values):
        if name not in self.component_quantiles:
            return np.zeros_like(np.asarray(values, dtype=np.float64))
        knots = self.component_quantiles[name]
        unique_values, unique_idx = np.unique(knots, return_index=True)
        unique_q = self.quantile_grid[unique_idx]
        if unique_values.size == 1:
            return (np.asarray(values) > unique_values[0]).astype(np.float64)
        return np.interp(
            np.asarray(values, dtype=np.float64), unique_values, unique_q,
            left=0.0, right=1.0,
        )

    def component_percentile(self, name, value):
        return float(self._component_percentile(name, np.asarray([value]))[0])

    def raw_quantile(self, name, quantile):
        if name not in self.component_quantiles:
            return 0.0
        return float(np.interp(float(quantile), self.quantile_grid, self.component_quantiles[name]))

    def score_arrays(self, components):
        first = next(iter(components.values()))
        score = np.zeros_like(np.asarray(first, dtype=np.float64))
        for name, weight in self.weights.items():
            if weight > 0:
                values = np.asarray(components.get(name, np.zeros_like(score)), dtype=np.float64)
                score += weight * self._component_percentile(name, values)
        return score

    def score_one(self, components):
        return float(
            self.score_arrays(
                {name: np.asarray([components.get(name, 0.0)]) for name in COMPONENT_NAMES}
            )[0]
        )

    def fit(self, components, groups=None, min_group_size=20):
        lengths = {len(np.asarray(v).reshape(-1)) for v in components.values()}
        if len(lengths) != 1 or not lengths:
            raise ValueError("All calibration components must have the same nonzero length")
        n = next(iter(lengths))
        if n == 0:
            raise ValueError("Cannot calibrate on an empty set")
        normalized = {}
        for name in COMPONENT_NAMES:
            values = np.asarray(components.get(name, np.zeros(n)), dtype=np.float64).reshape(-1)
            values = np.nan_to_num(values, nan=np.inf, posinf=np.inf, neginf=0.0)
            finite = values[np.isfinite(values)]
            fill = float(np.max(finite)) if finite.size else 0.0
            values = np.where(np.isfinite(values), values, fill)
            self.component_quantiles[name] = np.quantile(values, self.quantile_grid)
            normalized[name] = values
        scores = self.score_arrays(normalized)
        self.global_thresholds = (
            float(np.quantile(scores, self.q_low)),
            float(np.quantile(scores, self.q_high)),
        )
        self.group_thresholds = {}
        if groups is not None:
            group_arr = np.asarray([str(g) for g in groups])
            if len(group_arr) != n:
                raise ValueError("groups length does not match components")
            for group in np.unique(group_arr):
                mask = group_arr == group
                if int(mask.sum()) >= int(min_group_size):
                    self.group_thresholds[str(group)] = (
                        float(np.quantile(scores[mask], self.q_low)),
                        float(np.quantile(scores[mask], self.q_high)),
                    )
        return self

    def thresholds(self, group=None):
        if self.global_thresholds is None:
            raise RuntimeError("Calibrator has not been fitted")
        if group is not None and str(group) in self.group_thresholds:
            return self.group_thresholds[str(group)]
        return self.global_thresholds

    def all_thresholds(self, include_global: bool = True) -> dict[str, tuple[float, float]]:
        if self.global_thresholds is None:
            raise RuntimeError("Calibrator has not been fitted")
        result = dict(self.group_thresholds)
        if include_global:
            result["__global__"] = self.global_thresholds
        return result

    def to_dict(self):
        if self.global_thresholds is None:
            raise RuntimeError("Calibrator has not been fitted")
        return {
            "version": 2,
            "components": list(COMPONENT_NAMES),
            "weights": self.weights,
            "q_low": self.q_low,
            "q_high": self.q_high,
            "quantile_grid": self.quantile_grid.tolist(),
            "component_quantiles": {
                name: values.tolist() for name, values in self.component_quantiles.items()
            },
            "global_thresholds": list(self.global_thresholds),
            "group_thresholds": {
                name: list(values) for name, values in self.group_thresholds.items()
            },
            "parameter_ood": self.ood_scorer.to_dict() if self.ood_scorer is not None else None,
        }

    @classmethod
    def from_dict(cls, payload):
        obj = cls(
            payload["weights"],
            float(payload["q_low"]),
            float(payload["q_high"]),
            len(payload["quantile_grid"]),
        )
        obj.quantile_grid = np.asarray(payload["quantile_grid"], dtype=np.float64)
        obj.component_quantiles = {
            name: np.asarray(values, dtype=np.float64)
            for name, values in payload["component_quantiles"].items()
        }
        obj.global_thresholds = tuple(float(v) for v in payload["global_thresholds"])
        obj.group_thresholds = {
            str(name): tuple(float(v) for v in values)
            for name, values in payload.get("group_thresholds", {}).items()
        }
        if payload.get("parameter_ood") is not None:
            obj.ood_scorer = ParameterOODScorer.from_dict(payload["parameter_ood"])
        return obj

    def save(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))
