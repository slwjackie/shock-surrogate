"""Shared parameter conditioning for surrogate backbones.

All backbones consume the same ordered physical parameter vector.  Positive
parameters are transformed with log1p before standardization; signed profile
parameters remain in their original coordinates.  The fitted mean/std are
stored as module buffers, so checkpoints are self-contained.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn

PARAMETER_KEYS: tuple[str, ...] = (
    "nu",
    "k",
    "E",
    "dTdx",
    "b_quad",
    "dt",
    "L_mm",
)
POSITIVE_PARAMETER_INDICES: tuple[int, ...] = (0, 1, 2, 5, 6)


def _as_batch_tensor(value: Any, like: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        out = value.to(device=like.device, dtype=like.dtype)
    else:
        out = torch.as_tensor(value, device=like.device, dtype=like.dtype)
    if out.ndim == 0:
        out = out.view(1).expand(like.shape[0])
    elif out.ndim > 1:
        out = out.reshape(out.shape[0], -1)[:, 0]
    if out.shape[0] == 1 and like.shape[0] > 1:
        out = out.expand(like.shape[0])
    if out.shape[0] != like.shape[0]:
        raise ValueError(f"Parameter batch {out.shape[0]} does not match state batch {like.shape[0]}")
    return out


def stack_parameter_mapping(
    params: Mapping[str, Any] | None,
    like: torch.Tensor,
    keys: Sequence[str] = PARAMETER_KEYS,
) -> torch.Tensor | None:
    """Stack a collated parameter mapping into a tensor with shape ``(B, P)``."""
    if params is None:
        return None
    columns = []
    for key in keys:
        default = 20.0 if key == "L_mm" else 0.0
        columns.append(_as_batch_tensor(params.get(key, default), like))
    return torch.stack(columns, dim=1)


def transform_parameter_tensor(values: torch.Tensor) -> torch.Tensor:
    out = values.clone()
    if out.shape[-1] != len(PARAMETER_KEYS):
        raise ValueError(f"Expected {len(PARAMETER_KEYS)} parameters, got {out.shape[-1]}")
    idx = torch.as_tensor(POSITIVE_PARAMETER_INDICES, device=out.device)
    positive = torch.clamp(out.index_select(-1, idx), min=0.0)
    out[..., idx] = torch.log1p(positive)
    return out


def transform_parameter_array(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    if out.shape[-1] != len(PARAMETER_KEYS):
        raise ValueError(f"Expected {len(PARAMETER_KEYS)} parameters, got {out.shape[-1]}")
    out[..., list(POSITIVE_PARAMETER_INDICES)] = np.log1p(
        np.clip(out[..., list(POSITIVE_PARAMETER_INDICES)], 0.0, None)
    )
    return out


def fit_parameter_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transformed = transform_parameter_array(values)
    mean = transformed.mean(axis=0)
    std = transformed.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


class ParameterConditioner(nn.Module):
    """Encode physical coefficients and produce FiLM parameters."""

    def __init__(self, feature_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden = int(hidden_dim or max(32, feature_dim // 2))
        self.register_buffer("param_mean", torch.zeros(len(PARAMETER_KEYS)))
        self.register_buffer("param_std", torch.ones(len(PARAMETER_KEYS)))
        self.encoder = nn.Sequential(
            nn.Linear(len(PARAMETER_KEYS), hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * feature_dim),
        )
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)

    def set_stats(self, mean: np.ndarray | torch.Tensor, std: np.ndarray | torch.Tensor) -> None:
        mean_t = torch.as_tensor(mean, dtype=self.param_mean.dtype, device=self.param_mean.device)
        std_t = torch.as_tensor(std, dtype=self.param_std.dtype, device=self.param_std.device)
        if mean_t.numel() != len(PARAMETER_KEYS) or std_t.numel() != len(PARAMETER_KEYS):
            raise ValueError("Parameter statistics have the wrong dimension")
        self.param_mean.copy_(mean_t.reshape_as(self.param_mean))
        self.param_std.copy_(std_t.clamp_min(1e-6).reshape_as(self.param_std))

    def normalized_parameters(
        self,
        params: Mapping[str, Any] | None,
        like: torch.Tensor,
    ) -> torch.Tensor | None:
        stacked = stack_parameter_mapping(params, like)
        if stacked is None:
            return None
        transformed = transform_parameter_tensor(stacked)
        return (transformed - self.param_mean[None, :]) / self.param_std[None, :]

    def forward(
        self,
        params: Mapping[str, Any] | None,
        like: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = like.shape[0]
        normalized = self.normalized_parameters(params, like)
        if normalized is None:
            zeros = torch.zeros(
                batch,
                self.encoder[-1].out_features // 2,
                device=like.device,
                dtype=like.dtype,
            )
            return zeros, zeros
        film = self.encoder(normalized)
        gamma, beta = film.chunk(2, dim=-1)
        return gamma, beta
