"""Predictive distributions for interchangeable surrogate backbones.

The repository keeps ``forward`` backward-compatible: a model still returns
``(field_mean, regime_logits)``.  This module obtains an actual distributional
advice signal by sampling dropout masks at inference time.  The same helper can
later be extended to multiple checkpoint ensembles without changing the policy.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn


_DROPOUT_TYPES = (
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.AlphaDropout,
    nn.FeatureAlphaDropout,
)


@dataclass(frozen=True)
class AdviceDistribution:
    mean: torch.Tensor
    variance: torch.Tensor
    std: torch.Tensor
    uncertainty: torch.Tensor
    logits: torch.Tensor | None
    samples: int


@contextmanager
def _sampling_mode(model: nn.Module, enable_dropout: bool):
    """Use evaluation behavior, optionally re-enabling only dropout layers."""
    states = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        if enable_dropout:
            for module, _ in states:
                if isinstance(module, _DROPOUT_TYPES):
                    module.train(True)
        yield
    finally:
        for module, training in states:
            module.train(training)


def has_active_dropout(model: nn.Module) -> bool:
    for module in model.modules():
        if isinstance(module, _DROPOUT_TYPES) and float(module.p) > 0:
            return True
    return False


def predict_distribution(
    model: nn.Module,
    x: torch.Tensor,
    state_history: torch.Tensor,
    parameters: Mapping[str, Any] | None = None,
    mc_samples: int = 8,
    relative_scale_floor: float = 1e-6,
) -> AdviceDistribution:
    """Return MC-dropout field moments and a scalar risk feature per case.

    ``uncertainty`` is the root-mean predictive variance divided by the mean
    absolute field magnitude, so it is dimensionless and comparable across
    coefficient regimes.  With ``mc_samples=1`` the function is deterministic
    and returns exactly zero variance.
    """
    samples = max(1, int(mc_samples))
    predictions: list[torch.Tensor] = []
    logits_samples: list[torch.Tensor] = []

    # One sample means ordinary deterministic evaluation. More than one sample
    # enables dropout masks while BatchNorm/LayerNorm remain in evaluation mode.
    with _sampling_mode(model, enable_dropout=samples > 1), torch.no_grad():
        for _ in range(samples):
            prediction, logits = model(x, state_history, parameters)
            predictions.append(prediction)
            if logits is not None:
                logits_samples.append(logits)

    stacked = torch.stack(predictions, dim=0)
    mean = stacked.mean(dim=0)
    variance = stacked.var(dim=0, unbiased=False) if samples > 1 else torch.zeros_like(mean)
    std = torch.sqrt(variance.clamp_min(0.0))
    field_scale = mean.abs().mean(dim=1).clamp_min(float(relative_scale_floor))
    uncertainty = torch.sqrt(variance.mean(dim=1).clamp_min(0.0)) / field_scale
    mean_logits = torch.stack(logits_samples, dim=0).mean(dim=0) if logits_samples else None
    return AdviceDistribution(
        mean=mean,
        variance=variance,
        std=std,
        uncertainty=uncertainty,
        logits=mean_logits,
        samples=samples,
    )
