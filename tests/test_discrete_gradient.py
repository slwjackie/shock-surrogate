import torch

from physics.discrete_residual import shock_aware_residual_loss


def test_discrete_physics_loss_has_finite_gradients():
    batch, nx = 2, 48
    x = torch.linspace(0, 1, nx).repeat(batch, 1)
    last = torch.rand(batch, nx)
    pred = (last + 0.02 * torch.randn(batch, nx)).requires_grad_(True)
    params = {
        "dt": torch.full((batch,), 0.01), "nu": torch.full((batch,), 0.002),
        "k": torch.full((batch,), 1.5), "E": torch.full((batch,), 6.0),
        "dTdx": torch.zeros(batch), "b_quad": torch.zeros(batch),
        "L_mm": torch.full((batch,), 20.0),
    }
    loss, residual = shock_aware_residual_loss(pred, last, x, params)
    loss.backward()
    assert torch.isfinite(residual).all()
    assert torch.isfinite(pred.grad).all()
