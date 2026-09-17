#!/usr/bin/env python3
"""Parameter-conditioned temporal Transformer with local spatial mixing.

The model remains the Phase-1 advice backbone. It consumes physical
coefficients explicitly through FiLM conditioning, so coefficient-mismatch OOD
is no longer hidden from the predictor.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from backbones.base import SurrogateBackbone
from backbones.conditioning import ParameterConditioner


@dataclass
class HybridTransformerCfg:
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.0
    causal: bool = True
    mlp_hidden: int = 128
    n_classes: int = 3


def _causal_mask(length, device):
    return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)


class TemporalEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.in_proj = nn.Linear(1, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward, dropout=cfg.dropout,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.ln = nn.LayerNorm(cfg.d_model)

    def forward(self, u_seq, causal):
        h = self.in_proj(u_seq)
        mask = _causal_mask(h.shape[1], h.device) if causal else None
        return self.ln(self.enc(h, mask=mask))


class HybridTemporalSpatialTransformer(SurrogateBackbone):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.temporal = TemporalEncoder(cfg)
        self.spatial = nn.Sequential(
            nn.Conv1d(cfg.d_model, cfg.d_model, 3, padding=1), nn.GELU(),
            nn.Conv1d(cfg.d_model, cfg.d_model, 3, padding=1), nn.GELU(),
        )
        self.conditioner = ParameterConditioner(cfg.d_model)
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model + 1, cfg.mlp_hidden), nn.GELU(), nn.Linear(cfg.mlp_hidden, 1)
        )
        cls_in_dim = 2 * cfg.d_model + 6
        self.cls_head = (
            nn.Sequential(
                nn.LayerNorm(cls_in_dim), nn.Linear(cls_in_dim, cfg.mlp_hidden),
                nn.GELU(), nn.Dropout(cfg.dropout),
                nn.Linear(cfg.mlp_hidden, cfg.mlp_hidden), nn.GELU(),
                nn.Dropout(cfg.dropout), nn.Linear(cfg.mlp_hidden, cfg.n_classes),
            ) if cfg.n_classes > 0 else None
        )

    def set_parameter_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.conditioner.set_stats(mean, std)

    def forward(self, x, state_history, parameters: Mapping[str, Any] | None = None):
        batch, nx, history = state_history.shape
        u_seq = state_history.reshape(batch * nx, history, 1)
        h_last = self.temporal(u_seq, causal=self.cfg.causal)[:, -1, :]
        h_grid = self.spatial(h_last.reshape(batch, nx, -1).transpose(1, 2)).transpose(1, 2)
        gamma, beta = self.conditioner(parameters, x)
        h_grid = h_grid * (1.0 + gamma[:, None, :]) + beta[:, None, :]
        u_next = self.head(torch.cat([h_grid, x.unsqueeze(-1)], dim=-1)).squeeze(-1)
        logits = None
        if self.cls_head is not None:
            pooled = torch.cat([h_grid.mean(dim=1), h_grid.max(dim=1).values], dim=1)
            du = (u_next[:, 1:] - u_next[:, :-1]).abs()
            physics_features = torch.stack([
                du.max(dim=1).values, du.mean(dim=1), du.std(dim=1, unbiased=False),
                u_next.max(dim=1).values, u_next.mean(dim=1),
                du.argmax(dim=1).float() / max(nx - 1, 1),
            ], dim=1)
            logits = self.cls_head(torch.cat([pooled, physics_features], dim=1))
        return u_next, logits


def build_model(arch: str, n_classes: int, causal: bool = True, **kwargs):
    if arch not in {"transformer_hybrid", "hybrid_transformer", "transformer_temporal_hybrid", "transformer"}:
        raise ValueError(f"Unsupported Transformer alias: {arch}")
    cfg = HybridTransformerCfg(
        d_model=int(kwargs.get("d_model", 128)), nhead=int(kwargs.get("nhead", 4)),
        num_layers=int(kwargs.get("num_layers", 4)),
        dim_feedforward=int(kwargs.get("dim_feedforward", 256)),
        dropout=float(kwargs.get("dropout", 0.0)), causal=causal,
        mlp_hidden=int(kwargs.get("mlp_hidden", 128)), n_classes=n_classes,
    )
    return HybridTemporalSpatialTransformer(cfg)
