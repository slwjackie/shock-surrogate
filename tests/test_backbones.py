import numpy as np
import torch

from backbones.registry import build_backbone


def _params(batch):
    return {
        "nu": torch.full((batch,), 0.002),
        "k": torch.full((batch,), 1.5),
        "E": torch.full((batch,), 6.0),
        "dTdx": torch.zeros(batch),
        "b_quad": torch.zeros(batch),
        "dt": torch.full((batch,), 0.01),
        "L_mm": torch.full((batch,), 20.0),
    }


def test_transformer_and_fno_share_interface():
    batch, nx, history = 2, 32, 5
    x = torch.linspace(0, 1, nx).repeat(batch, 1)
    state = torch.randn(batch, nx, history)
    mean = np.zeros(7, dtype=np.float32)
    std = np.ones(7, dtype=np.float32)
    transformer = build_backbone(
        "transformer_hybrid", history=history, n_classes=3,
        d_model=32, nhead=4, num_layers=1, dim_feedforward=64,
    )
    fno = build_backbone(
        "fno1d", history=history, n_classes=3, width=16, modes=8, depth=2,
    )
    for model in (transformer, fno):
        model.set_parameter_stats(mean, std)
        prediction, logits = model(x, state, _params(batch))
        assert prediction.shape == (batch, nx)
        assert logits.shape == (batch, 3)
        assert torch.isfinite(prediction).all()
