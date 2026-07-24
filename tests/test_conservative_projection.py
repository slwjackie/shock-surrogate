import torch

from physics.discrete_residual import cell_widths, physical_grid
from physics.residual_projection import ResidualProjectionConfig, project_prediction


def _params(batch):
    return {
        "dt": torch.full((batch,), 0.02),
        "nu": torch.zeros(batch),
        "k": torch.zeros(batch),
        "E": torch.full((batch,), 6.0),
        "dTdx": torch.zeros(batch),
        "b_quad": torch.zeros(batch),
        "L_mm": torch.full((batch,), 20.0),
    }


def test_projection_is_positive_and_mass_conservative():
    batch, nx = 2, 64
    x_base = torch.linspace(0, 1, nx).pow(1.25)
    x = x_base.repeat(batch, 1)
    last = torch.ones(batch, nx)
    pred = last.clone()
    pred[:, 25:32] += 0.8
    pred[0, 0] = -0.05  # feasible negative advice; total mass remains positive.

    corrected, info = project_prediction(
        pred,
        last,
        x,
        _params(batch),
        ResidualProjectionConfig(steps=2, max_update=0.1),
    )
    assert corrected.min().item() >= -1e-7
    weights = cell_widths(physical_grid(x, _params(batch)["L_mm"]))
    mass_before = (pred * weights).sum(dim=1)
    mass_after = (corrected * weights).sum(dim=1)
    assert torch.allclose(mass_after, mass_before, atol=2e-5, rtol=0)
    assert info["max_mass_drift"] < 2e-5
    assert torch.isfinite(corrected).all()
