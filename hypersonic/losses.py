"""LGNO-style physical-space and high-frequency spectral losses."""

from __future__ import annotations

import torch


def componentwise_relative_l1(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape")
    spatial = tuple(range(2, pred.ndim))
    numerator = (pred - target).abs().sum(dim=spatial)
    denominator = target.abs().sum(dim=spatial).clamp_min(float(eps))
    return (numerator / denominator).mean()


def high_frequency_spectral_loss(pred: torch.Tensor, target: torch.Tensor, high_fraction: float = 0.5) -> torch.Tensor:
    error = pred - target
    if error.ndim == 3:
        spectrum = torch.fft.rfft(error, dim=-1, norm="ortho")
        k0 = int(float(high_fraction) * spectrum.shape[-1])
        return spectrum[..., k0:].abs().square().mean()
    if error.ndim != 4:
        raise ValueError("Expected 1-D or 2-D batched fields")
    spectrum = torch.fft.rfft2(error, norm="ortho")
    ny, nxh = spectrum.shape[-2:]
    ky = torch.fft.fftfreq(ny, device=error.device).abs().view(ny, 1)
    kx = torch.fft.rfftfreq((nxh - 1) * 2, device=error.device).abs().view(1, nxh)
    radius = torch.sqrt(ky.square() + kx.square())
    threshold = float(high_fraction) * float(radius.max().item())
    mask = radius >= threshold
    return spectrum[..., mask].abs().square().mean()


def lgno_loss(pred: torch.Tensor, target: torch.Tensor, spectral_weight: float = 1e-3, high_fraction: float = 0.5) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    physical = componentwise_relative_l1(pred, target)
    spectral = high_frequency_spectral_loss(pred, target, high_fraction=high_fraction)
    total = physical + float(spectral_weight) * spectral
    return total, {"physical": physical, "high_frequency": spectral}
