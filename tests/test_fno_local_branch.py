import torch

from backbones.fno import ConditionedFNO1d, LocalMultiScaleConv1d


def test_local_branch_has_nontrivial_neighbor_receptive_field():
    layer = LocalMultiScaleConv1d(1)
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.zero_()
        layer.branch3.weight.fill_(1.0)
        layer.mix.weight[0, 0, 0] = 1.0
    impulse = torch.zeros(1, 1, 17)
    impulse[0, 0, 8] = 1.0
    output = layer(impulse)
    assert output[0, 0, 7].item() != 0.0
    assert output[0, 0, 8].item() != 0.0
    assert output[0, 0, 9].item() != 0.0


def test_conditioned_fno_forward_and_backward_are_finite():
    model = ConditionedFNO1d(history=3, width=8, modes=4, depth=2, dropout=0.1)
    x = torch.linspace(0, 1, 32).repeat(2, 1)
    history = torch.randn(2, 32, 3, requires_grad=True)
    params = {
        "nu": torch.full((2,), 0.002), "k": torch.full((2,), 1.5),
        "E": torch.full((2,), 6.0), "dTdx": torch.zeros(2),
        "b_quad": torch.zeros(2), "dt": torch.full((2,), 0.01),
        "L_mm": torch.full((2,), 20.0),
    }
    prediction, logits = model(x, history, params)
    loss = prediction.square().mean() + logits.square().mean()
    loss.backward()
    assert torch.isfinite(prediction).all()
    assert torch.isfinite(history.grad).all()
