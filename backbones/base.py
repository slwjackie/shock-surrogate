"""Common interface for interchangeable shock-surrogate backbones."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class SurrogateBackbone(nn.Module, ABC):
    """Backbones predict one next field and optional regime logits."""

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        state_history: torch.Tensor,
        parameters: Mapping[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        raise NotImplementedError

    def set_parameter_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        conditioner = getattr(self, "conditioner", None)
        if conditioner is not None:
            conditioner.set_stats(mean, std)
