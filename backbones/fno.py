"""Parameter-conditioned 1-D Fourier Neural Operator with a true local branch."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones.base import SurrogateBackbone
from backbones.conditioning import ParameterConditioner


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes = int(modes)
        scale = 1.0 / max(1, in_channels * out_channels)
        weight = scale * torch.randn(in_channels, out_channels, self.modes, dtype=torch.cfloat)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nx = x.shape[-1]
        x_ft = torch.fft.rfft(x, dim=-1)
        n_modes = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros(
            x.shape[0], self.out_channels, x_ft.shape[-1],
            dtype=x_ft.dtype, device=x.device,
        )
        out_ft[:, :, :n_modes] = torch.einsum(
            "bix,iox->box", x_ft[:, :, :n_modes], self.weight[:, :, :n_modes]
        )
        return torch.fft.irfft(out_ft, n=nx, dim=-1)


class LocalMultiScaleConv1d(nn.Module):
    """Local shock-sensitive branch with several finite receptive fields.

    The previous 1x1 layer was only a pointwise channel mixer. These depthwise
    convolutions explicitly exchange information across neighboring cells at
    three scales before a pointwise fusion.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.branch3 = nn.Conv1d(
            channels, channels, kernel_size=3, padding=1,
            groups=channels, padding_mode="replicate",
        )
        self.branch5 = nn.Conv1d(
            channels, channels, kernel_size=5, padding=2,
            groups=channels, padding_mode="replicate",
        )
        self.branch_dilated = nn.Conv1d(
            channels, channels, kernel_size=3, padding=2, dilation=2,
            groups=channels, padding_mode="replicate",
        )
        self.mix = nn.Conv1d(3 * channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = torch.cat(
            [self.branch3(x), self.branch5(x), self.branch_dilated(x)], dim=1
        )
        return self.mix(features)


class ConditionedFNO1d(SurrogateBackbone):
    def __init__(
        self,
        history: int = 5,
        width: int = 64,
        modes: int = 24,
        depth: int = 4,
        n_classes: int = 3,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.history = int(history)
        self.width = int(width)
        self.depth = int(depth)
        self.lift = nn.Linear(self.history + 1, self.width)
        self.spectral = nn.ModuleList(
            [SpectralConv1d(self.width, self.width, modes) for _ in range(self.depth)]
        )
        self.local = nn.ModuleList(
            [LocalMultiScaleConv1d(self.width) for _ in range(self.depth)]
        )
        self.norms = nn.ModuleList([nn.GroupNorm(1, self.width) for _ in range(self.depth)])
        self.conditioner = ParameterConditioner(self.width)
        self.dropout = nn.Dropout(dropout)
        self.project = nn.Sequential(
            nn.Linear(self.width, 2 * self.width), nn.GELU(), nn.Linear(2 * self.width, 1)
        )
        cls_in = 2 * self.width + 6
        self.cls_head = (
            nn.Sequential(
                nn.LayerNorm(cls_in), nn.Linear(cls_in, self.width), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(self.width, n_classes),
            ) if n_classes > 0 else None
        )

    def set_parameter_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.conditioner.set_stats(mean, std)

    def forward(
        self,
        x: torch.Tensor,
        state_history: torch.Tensor,
        parameters: Mapping[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if state_history.shape[-1] != self.history:
            raise ValueError(
                f"FNO was initialized with history={self.history}, got {state_history.shape[-1]}"
            )
        features = torch.cat([state_history, x.unsqueeze(-1)], dim=-1)
        h = self.lift(features).transpose(1, 2)
        gamma, beta = self.conditioner(parameters, x)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        for i, (spectral, local, norm) in enumerate(zip(self.spectral, self.local, self.norms)):
            y = norm(spectral(h) + local(h))
            y = y * (1.0 + gamma) + beta
            # Apply dropout on every operator block so MC-dropout yields a field
            # distribution even for shallow (depth=1) FNO configurations.
            h = h + self.dropout(F.gelu(y))
        h_grid = h.transpose(1, 2)
        u_next = self.project(h_grid).squeeze(-1)
        logits = None
        if self.cls_head is not None:
            pooled = torch.cat([h_grid.mean(dim=1), h_grid.max(dim=1).values], dim=1)
            du = (u_next[:, 1:] - u_next[:, :-1]).abs()
            physics_features = torch.stack([
                du.max(dim=1).values,
                du.mean(dim=1),
                du.std(dim=1, unbiased=False),
                u_next.max(dim=1).values,
                u_next.mean(dim=1),
                du.argmax(dim=1).float() / max(u_next.shape[1] - 1, 1),
            ], dim=1)
            logits = self.cls_head(torch.cat([pooled, physics_features], dim=1))
        return u_next, logits
