import torch

from physics.discrete_residual import burgers_reaction_residual_fd, risk_components_torch


def test_constant_equilibrium_has_zero_residual():
    batch, nx = 2, 64
    u = torch.ones(batch, nx)
    x = torch.linspace(0, 1, nx).repeat(batch, 1)
    params = {
        "dt": torch.full((batch,), 0.01),
        "nu": torch.full((batch,), 0.002),
        "k": torch.full((batch,), 1.5),
        "E": torch.full((batch,), 6.0),
        "dTdx": torch.zeros(batch),
        "b_quad": torch.zeros(batch),
        "L_mm": torch.full((batch,), 20.0),
    }
    residual = burgers_reaction_residual_fd(u, u, x, params)
    assert residual.abs().max().item() < 1e-6
    components = risk_components_torch(u, u, x, params)
    assert components["residual"].max().item() < 1e-6
    assert components["tv_growth"].max().item() == 0.0
    assert components["shock_shift"].max().item() == 0.0
