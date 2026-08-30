"""MC-dropout statistical uncertainty proxy."""
from __future__ import annotations
import torch
from ..surrogate import mc_dropout_predictions
@torch.no_grad()
def mc_dropout_uncertainty(model,u,samples=8):
    draws=mc_dropout_predictions(model,u,samples=samples)
    return draws.std(dim=0,unbiased=True).mean(dim=-1)
