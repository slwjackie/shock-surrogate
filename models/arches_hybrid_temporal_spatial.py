#!/usr/bin/env python3
"""
Hybrid Temporal (causal) Transformer + Spatial autograd PINN-ready head.

Idea:
  - Input: history snapshots u_hist(x, t_{k..k+H-1}) for each spatial grid point x
           shape (B, Nx, H)
  - Temporal encoder: causal self-attention over the H time tokens, applied PER spatial point.
    We reshape to (B*Nx, H, 1) and run a small TransformerEncoder.
  - Head: uses continuous coordinate x (so we can autograd du/dx) + temporal representation
    to predict u_next(x, t_{k+H}) for each x. Output shape (B, Nx)
  - Optional regime classifier: pool temporal representations over space and classify.

This gives:
  - temporal derivative via FD: (u_pred - u_last)/dt
  - spatial derivatives via autograd: u_x, u_xx from dependence on x
"""

from dataclasses import dataclass
import torch
import torch.nn as nn

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

def _causal_mask(L: int, device):
    # True = masked in PyTorch boolean attn_mask for TransformerEncoderLayer in newer versions
    return torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)

class TemporalEncoder(nn.Module):
    def __init__(self, cfg: HybridTransformerCfg):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(1, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.ln = nn.LayerNorm(cfg.d_model)

    def forward(self, u_seq: torch.Tensor, causal: bool):
        """
        u_seq: (B*Nx, H, 1)
        returns: (B*Nx, H, d_model)
        """
        h = self.in_proj(u_seq)
        mask = _causal_mask(h.shape[1], h.device) if causal else None
        h = self.enc(h, mask=mask)
        return self.ln(h)

class HybridTemporalSpatialTransformer(nn.Module):
    def __init__(self, cfg: HybridTransformerCfg):
        super().__init__()
        self.cfg = cfg
        self.temporal = TemporalEncoder(cfg)
        # head predicts scalar u_next per x
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model + 1, cfg.mlp_hidden),
            nn.GELU(),
            nn.Linear(cfg.mlp_hidden, cfg.mlp_hidden),
            nn.GELU(),
            nn.Linear(cfg.mlp_hidden, 1),
        )
        # optional classifier
        # self.cls_head = nn.Sequential(
        #     nn.Linear(cfg.d_model, cfg.mlp_hidden),
        #     nn.GELU(),
        #     nn.Linear(cfg.mlp_hidden, cfg.n_classes),
        # )
        # optional classifier (uses pooled hidden + simple physics features from u_next)
        self.cls_head = nn.Sequential(
            nn.Linear(cfg.d_model + 3, cfg.mlp_hidden),  # +3: g_peak, u_max, tv
            nn.GELU(),
            nn.Linear(cfg.mlp_hidden, cfg.n_classes),
        ) if cfg.n_classes and cfg.n_classes > 0 else None
        
        

    def forward(self, x: torch.Tensor, u_hist: torch.Tensor):
        """
        x: (B, Nx) continuous coordinates (requires_grad can be True)
        u_hist: (B, Nx, H) history snapshots for each x

        returns:
          u_next: (B, Nx)
          logits: (B, n_classes) or None
        """
        B, Nx, H = u_hist.shape
        # temporal encoding per spatial point
        u_seq = u_hist.reshape(B*Nx, H, 1)
        h = self.temporal(u_seq, causal=self.cfg.causal)         # (B*Nx, H, d_model)
        h_last = h[:, -1, :]                                     # (B*Nx, d_model)

        # head uses x coordinate
        x_flat = x.reshape(B*Nx, 1)
        feat = torch.cat([h_last, x_flat], dim=-1)
        u_next = self.head(feat).reshape(B, Nx)

        # logits = None
        # if self.cls_head is not None:
        #     # pool h_last over space to get case-level representation
        #     pooled = h_last.reshape(B, Nx, -1).mean(dim=1)       # (B, d_model)
        #     logits = self.cls_head(pooled)
        logits = None
        if self.cls_head is not None:
            # 1) pooled hidden (case-level)
            pooled = h_last.reshape(B, Nx, -1).mean(dim=1)  # (B, d_model)

            # 2) physics-inspired features from predicted field u_next (aka u_pred)
            # u_next: (B, Nx)
            du = (u_next[:, 1:] - u_next[:, :-1]).abs()     # (B, Nx-1)
            g_peak = du.max(dim=1).values                   # (B,)  ~ max|du/dx| proxy (dx constant)
            tv = du.mean(dim=1)                             # (B,)  ~ TV proxy
            u_max = u_next.max(dim=1).values                # (B,)

            feat = torch.stack([g_peak, u_max, tv], dim=1)  # (B, 3)

            # 3) concat and classify
            pooled_plus = torch.cat([pooled, feat], dim=1)  # (B, d_model+3)
            logits = self.cls_head(pooled_plus)
        return u_next, logits

def build_model(arch: str, n_classes: int, causal: bool = True, **kwargs):
    if arch not in ["transformer_hybrid", "hybrid_transformer", "transformer_temporal_hybrid"]:
        raise ValueError(f"Unsupported arch for hybrid: {arch}")
    cfg = HybridTransformerCfg(
        d_model=kwargs.get("d_model", 128),
        nhead=kwargs.get("nhead", 4),
        num_layers=kwargs.get("num_layers", 4),
        dim_feedforward=kwargs.get("dim_feedforward", 256),
        dropout=kwargs.get("dropout", 0.0),
        causal=causal,
        mlp_hidden=kwargs.get("mlp_hidden", 128),
        n_classes=n_classes,
    )
    return HybridTemporalSpatialTransformer(cfg)
