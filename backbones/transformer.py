"""Factory for the parameter-conditioned temporal/spatial Transformer."""

from __future__ import annotations

from models.arches_hybrid_temporal_spatial import (
    HybridTemporalSpatialTransformer,
    HybridTransformerCfg,
)


def build_transformer_backbone(
    n_classes: int = 3,
    causal: bool = True,
    **kwargs,
) -> HybridTemporalSpatialTransformer:
    cfg = HybridTransformerCfg(
        d_model=int(kwargs.get("d_model", 128)),
        nhead=int(kwargs.get("nhead", 4)),
        num_layers=int(kwargs.get("num_layers", 4)),
        dim_feedforward=int(kwargs.get("dim_feedforward", 256)),
        dropout=float(kwargs.get("dropout", 0.0)),
        causal=bool(causal),
        mlp_hidden=int(kwargs.get("mlp_hidden", 128)),
        n_classes=int(n_classes),
    )
    return HybridTemporalSpatialTransformer(cfg)
