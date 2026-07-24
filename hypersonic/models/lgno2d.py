"""Local--Global Neural Operator for two-dimensional conservation laws.

This implementation follows the published LGNO layer structure:
  * global low-mode Fourier branch,
  * local coarse-to-fine multiresolution convolution branch,
  * multiplicative global/local coupling,
  * pointwise residual fusion,
  * residual one-step flow-map parameterization,
  * mean-zero increment projection for periodic conservation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from hypersonic.positivity import positivity_preserving_blend


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes_y: int, modes_x: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes_y = int(modes_y)
        self.modes_x = int(modes_x)
        scale = 1.0 / math.sqrt(max(1, in_channels * out_channels))
        shape = (in_channels, out_channels, self.modes_y, self.modes_x)
        self.weight_pos = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weight_neg = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))

    @staticmethod
    def _mul(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, ny, nx = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(b, self.out_channels, ny, nx // 2 + 1, device=x.device, dtype=torch.cfloat)
        my = min(self.modes_y, ny // 2)
        mx = min(self.modes_x, nx // 2 + 1)
        out_ft[:, :, :my, :mx] = self._mul(x_ft[:, :, :my, :mx], self.weight_pos[:, :, :my, :mx])
        out_ft[:, :, -my:, :mx] = self._mul(x_ft[:, :, -my:, :mx], self.weight_neg[:, :, :my, :mx])
        return torch.fft.irfft2(out_ft, s=(ny, nx), norm="ortho")


class LearnedOutflowPad2d(nn.Module):
    """Lightweight learned ghost padding for local outflow convolutions."""

    def __init__(self, channels: int, ghost_width: int = 1, context: int = 3):
        super().__init__()
        self.ghost_width = int(ghost_width)
        self.context = int(context)
        self.left = nn.Conv2d(channels, channels * ghost_width, kernel_size=1)
        self.right = nn.Conv2d(channels, channels * ghost_width, kernel_size=1)
        self.bottom = nn.Conv2d(channels, channels * ghost_width, kernel_size=1)
        self.top = nn.Conv2d(channels, channels * ghost_width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.ghost_width
        c = min(self.context, x.shape[-1], x.shape[-2])
        left_ctx = x[..., :c].mean(dim=-1, keepdim=True)
        right_ctx = x[..., -c:].mean(dim=-1, keepdim=True)
        left = self.left(left_ctx).view(x.shape[0], x.shape[1], g, x.shape[-2]).permute(0, 1, 3, 2)
        right = self.right(right_ctx).view(x.shape[0], x.shape[1], g, x.shape[-2]).permute(0, 1, 3, 2)
        x_lr = torch.cat((left, x, right), dim=-1)
        bottom_ctx = x_lr[..., :c, :].mean(dim=-2, keepdim=True)
        top_ctx = x_lr[..., -c:, :].mean(dim=-2, keepdim=True)
        bottom = self.bottom(bottom_ctx).view(x.shape[0], x.shape[1], g, x_lr.shape[-1])
        top = self.top(top_ctx).view(x.shape[0], x.shape[1], g, x_lr.shape[-1])
        return torch.cat((bottom, x_lr, top), dim=-2)


class LocalMultiresolution2d(nn.Module):
    def __init__(self, channels: int, boundary: str = "periodic", kernel_size: int = 3):
        super().__init__()
        self.boundary = boundary
        pad = kernel_size // 2
        self.learned_pad = LearnedOutflowPad2d(channels, pad) if boundary == "outflow_learned" else None
        conv_pad = 0 if self.learned_pad is not None else pad
        padding_mode = "circular" if boundary == "periodic" else "replicate"
        self.coarse_conv1 = nn.Conv2d(channels, channels, kernel_size, padding=conv_pad, padding_mode=padding_mode)
        self.coarse_conv2 = nn.Conv2d(channels, channels, kernel_size, padding=conv_pad, padding_mode=padding_mode)
        self.fuse = nn.Conv2d(2 * channels, channels, kernel_size=1)

    def _conv(self, conv: nn.Conv2d, z: torch.Tensor) -> torch.Tensor:
        return conv(self.learned_pad(z) if self.learned_pad is not None else z)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        coarse = F.avg_pool2d(h, kernel_size=2, stride=2, ceil_mode=True)
        coarse = F.gelu(self._conv(self.coarse_conv1, coarse))
        coarse = F.gelu(self._conv(self.coarse_conv2, coarse))
        coarse = F.interpolate(coarse, size=h.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat((h, coarse), dim=1))


class LGNOLayer2d(nn.Module):
    def __init__(self, width: int, modes_y: int, modes_x: int, boundary: str):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes_y, modes_x)
        self.global_pointwise = nn.Conv2d(width, width, 1)
        self.local = LocalMultiresolution2d(width, boundary=boundary)
        self.global_gate = nn.Conv2d(width, width, 1)
        self.local_gate = nn.Conv2d(width, width, 1)
        self.mix = nn.Conv2d(4 * width, width, 1)
        self.norm = nn.GroupNorm(1, width)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        g = F.gelu(self.global_pointwise(h) + self.spectral(h))
        local = F.gelu(self.local(h))
        coupling = self.global_gate(g) * self.local_gate(local)
        update = self.mix(torch.cat((h, g, local, coupling), dim=1))
        return self.norm(h + update)


class LocalGlobalNeuralOperator2d(nn.Module):
    """One-step structured-grid operator for 2-D Euler/Navier--Stokes states."""

    def __init__(self, state_channels: int = 4, width: int = 64, modes_y: int = 16, modes_x: int = 16, depth: int = 4, boundary: str = "periodic", gamma: float = 1.4, rho_floor: float = 1e-6, p_floor: float = 1e-6, enforce_positivity: bool = True, add_coordinates: bool = True):
        super().__init__()
        self.state_channels = int(state_channels)
        self.boundary = boundary
        self.gamma = float(gamma)
        self.rho_floor = float(rho_floor)
        self.p_floor = float(p_floor)
        self.enforce_positivity = bool(enforce_positivity)
        self.add_coordinates = bool(add_coordinates)
        in_channels = state_channels + (2 if add_coordinates else 0)
        self.lift = nn.Conv2d(in_channels, width, 1)
        self.layers = nn.ModuleList([LGNOLayer2d(width, modes_y, modes_x, boundary) for _ in range(int(depth))])
        self.project = nn.Sequential(nn.Conv2d(width, 2 * width, 1), nn.GELU(), nn.Conv2d(2 * width, state_channels, 1))

    @staticmethod
    def coordinate_grid(x: torch.Tensor) -> torch.Tensor:
        b, _, ny, nx = x.shape
        yy = torch.linspace(0.0, 1.0, ny, device=x.device, dtype=x.dtype)
        xx = torch.linspace(0.0, 1.0, nx, device=x.device, dtype=x.dtype)
        y, xcoord = torch.meshgrid(yy, xx, indexing="ij")
        return torch.stack((xcoord, y), dim=0).unsqueeze(0).expand(b, -1, -1, -1)

    def increment(self, state: torch.Tensor) -> torch.Tensor:
        features = state
        if self.add_coordinates:
            features = torch.cat((state, self.coordinate_grid(state)), dim=1)
        h = F.gelu(self.lift(features))
        for layer in self.layers:
            h = layer(h)
        inc = self.project(h)
        if self.boundary == "periodic":
            inc = inc - inc.mean(dim=(-2, -1), keepdim=True)
        return inc

    def forward(self, state: torch.Tensor, dt: float | torch.Tensor = 1.0):
        inc = self.increment(state)
        if isinstance(dt, torch.Tensor):
            dt_t = dt.to(state).reshape(state.shape[0], *([1] * (state.ndim - 1)))
        else:
            dt_t = torch.as_tensor(float(dt), device=state.device, dtype=state.dtype)
        candidate = state + dt_t * inc
        theta = torch.ones(state.shape[0], device=state.device, dtype=state.dtype)
        if self.enforce_positivity:
            candidate, theta = positivity_preserving_blend(state, candidate, gamma=self.gamma, rho_floor=self.rho_floor, p_floor=self.p_floor)
        return candidate, {"increment": inc, "positivity_theta": theta}
