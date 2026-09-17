"""Backbone registry used by training, calibration, and rollout evaluation."""

from __future__ import annotations

from backbones.fno import ConditionedFNO1d
from backbones.transformer import build_transformer_backbone

_TRANSFORMER_ALIASES = {
    "transformer_hybrid", "hybrid_transformer", "transformer_temporal_hybrid", "transformer"
}
_FNO_ALIASES = {"fno", "fno1d", "conditioned_fno"}


def available_backbones() -> tuple[str, ...]:
    return ("transformer_hybrid", "fno1d")


def build_backbone(
    arch: str,
    n_classes: int = 3,
    causal: bool = True,
    history: int = 5,
    **kwargs,
):
    name = arch.lower()
    if name in _TRANSFORMER_ALIASES:
        return build_transformer_backbone(n_classes=n_classes, causal=causal, **kwargs)
    if name in _FNO_ALIASES:
        return ConditionedFNO1d(
            history=history,
            width=int(kwargs.get("width", kwargs.get("d_model", 64))),
            modes=int(kwargs.get("modes", 24)),
            depth=int(kwargs.get("depth", kwargs.get("num_layers", 4))),
            n_classes=n_classes,
            dropout=float(kwargs.get("dropout", 0.0)),
        )
    raise ValueError(f"Unsupported backbone '{arch}'. Available: {available_backbones()}")
