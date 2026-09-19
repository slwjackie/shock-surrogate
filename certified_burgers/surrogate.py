"""Small local CNN advice models for the Burgers trust-or-fallback study.

Architecture novelty is intentionally not the goal. The state model is kept as a
non-conservative control, while the flux model enforces global conservation by construction.
Both use circular padding because the active experiment uses a periodic 1-D domain.
"""
from __future__ import annotations

import torch
from torch import nn


def _channel_first(u: torch.Tensor, channels: int):
    if u.ndim == 2:
        if int(channels) != 1:
            raise ValueError("Rank-2 input is only valid for a single state channel.")
        return u[:, None, :], True
    if u.ndim == 3 and u.shape[1] == int(channels):
        return u, False
    raise ValueError(f"Expected (B,N) for C=1 or (B,C,N) with C={channels}.")


def _restore_shape(y: torch.Tensor, squeeze_channel: bool):
    return y[:, 0, :] if squeeze_channel else y


def _periodic_conv(in_channels, out_channels, kernel_size, dilation=1):
    kernel_size = int(kernel_size)
    dilation = int(dilation)
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer.")
    if dilation < 1:
        raise ValueError("dilation must be positive.")
    return nn.Conv1d(
        int(in_channels), int(out_channels), kernel_size,
        padding=(kernel_size // 2) * dilation,
        dilation=dilation, padding_mode="circular",
    )


def _spatial_dilations(horizon: int, depth: int, kernel_size: int):
    """Use only as many spatial layers as the H-step domain of dependence needs."""
    horizon=int(horizon); depth=int(depth); kernel_size=int(kernel_size)
    if horizon<1 or depth<1:
        raise ValueError("horizon and depth must be positive.")
    radius_per_dilation=kernel_size//2
    values=[]; radius=0; dilation=1
    while radius<horizon and len(values)<depth:
        values.append(dilation)
        radius += radius_per_dilation*dilation
        dilation *= 2
    if radius<horizon:
        raise ValueError("Increase depth: receptive field is too small for this horizon.")
    return tuple(values)


class StateConvSurrogate(nn.Module):
    """Direct state-prediction CNN used as the non-conservative control."""
    architecture_kind="state"

    def __init__(self,channels=1,width=24,depth=3,kernel_size=5,dropout=0.05):
        super().__init__()
        if int(depth)<1: raise ValueError("depth must be >= 1")
        self.channels=int(channels); self.width=int(width); self.depth=int(depth)
        self.kernel_size=int(kernel_size); self.dropout=float(dropout)
        # Exactly one 5-point spatial stencil for H=1; later layers only mix channels.
        layers=[_periodic_conv(self.channels,self.width,self.kernel_size),
                nn.GELU(),nn.Dropout(self.dropout)]
        for _ in range(self.depth-1):
            layers += [nn.Conv1d(self.width,self.width,kernel_size=1),
                       nn.GELU(),nn.Dropout(self.dropout)]
        layers.append(nn.Conv1d(self.width,self.channels,kernel_size=1))
        self.net=nn.Sequential(*layers)
        self.receptive_radius_cells=self.kernel_size//2

    def forward(self,u):
        x,squeeze_channel=_channel_first(u,self.channels)
        return _restore_shape(x+self.net(x),squeeze_channel)

    def architecture_summary(self):
        return {"kind":self.architecture_kind,"channels":self.channels,"width":self.width,
                "depth":self.depth,"kernel_size":self.kernel_size,"periodic_padding":True,
                "receptive_radius_cells":self.receptive_radius_cells,
                "parameters":int(sum(p.numel() for p in self.parameters()))}


class ConservativeFluxSurrogate(nn.Module):
    """Local flux CNN with a conservative finite-volume state update."""
    architecture_kind="flux"

    def __init__(self,*,horizon=1,dt_over_dx=0.2,channels=1,width=24,depth=4,
                 kernel_size=5,dropout=0.05):
        super().__init__()
        if int(horizon)<1: raise ValueError("horizon must be positive.")
        if float(dt_over_dx)<=0: raise ValueError("dt_over_dx must be positive.")
        self.horizon=int(horizon); self.dt_over_dx=float(dt_over_dx)
        self.channels=int(channels); self.width=int(width); self.depth=int(depth)
        self.kernel_size=int(kernel_size); self.dropout=float(dropout)
        self.dilations=_spatial_dilations(self.horizon,self.depth,self.kernel_size)
        self.receptive_radius_cells=(self.kernel_size//2)*sum(self.dilations)
        layers=[]; in_channels=self.channels
        for dilation in self.dilations:
            layers += [_periodic_conv(in_channels,self.width,self.kernel_size,dilation),
                       nn.GELU(),nn.Dropout(self.dropout)]
            in_channels=self.width
        # Keep total depth small without enlarging the spatial stencil unnecessarily.
        for _ in range(self.depth-len(self.dilations)):
            layers += [nn.Conv1d(self.width,self.width,kernel_size=1),
                       nn.GELU(),nn.Dropout(self.dropout)]
        layers.append(nn.Conv1d(self.width,self.channels,kernel_size=1))
        self.flux_net=nn.Sequential(*layers)

    @property
    def update_scale(self):
        return float(self.horizon)*self.dt_over_dx

    def forward(self,u):
        x,squeeze_channel=_channel_first(u,self.channels)
        right_face_flux=self.flux_net(x)
        left_face_flux=torch.roll(right_face_flux,shifts=1,dims=-1)
        y=x-self.update_scale*(right_face_flux-left_face_flux)
        return _restore_shape(y,squeeze_channel)

    def architecture_summary(self):
        return {"kind":self.architecture_kind,"channels":self.channels,"width":self.width,
                "depth":self.depth,"kernel_size":self.kernel_size,"horizon":self.horizon,
                "dt_over_dx":self.dt_over_dx,"dilations":list(self.dilations),
                "periodic_padding":True,"globally_conservative_by_construction":True,
                "receptive_radius_cells":self.receptive_radius_cells,
                "parameters":int(sum(p.numel() for p in self.parameters()))}


def build_surrogate(kind="state",*,horizon=1,dt_over_dx=0.2,channels=1):
    if kind=="state":
        return StateConvSurrogate(channels=channels,width=24,depth=3,kernel_size=5)
    if kind=="flux":
        return ConservativeFluxSurrogate(horizon=horizon,dt_over_dx=dt_over_dx,
                                         channels=channels,width=24,depth=4,kernel_size=5)
    raise ValueError("kind must be 'state' or 'flux'.")


TinyConvSurrogate=StateConvSurrogate


@torch.no_grad()
def mc_dropout_predictions(model,u,samples=8):
    if samples<2: raise ValueError("samples must be >= 2")
    was_training=model.training
    model.train()
    preds=torch.stack([model(u) for _ in range(samples)],dim=0)
    model.train(was_training)
    return preds
