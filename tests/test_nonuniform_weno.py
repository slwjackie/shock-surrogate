import numpy as np
import torch

from numerics.remap import (
    cell_integral,
    conservative_remap_1d,
    uniform_centers_for_domain,
)
from solvers.weno_adapter import WENOSolverAdapter


def test_conservative_remap_preserves_cell_integral():
    source = np.array([0.0, 0.08, 0.21, 0.43, 0.71, 1.0])
    values = 1.2 + np.sin(2 * np.pi * source)
    target = uniform_centers_for_domain(source)
    remapped = conservative_remap_1d(values, source, target)
    assert abs(cell_integral(values, source) - cell_integral(remapped, target)) < 1e-12
    back = conservative_remap_1d(remapped, target, source)
    assert abs(cell_integral(values, source) - cell_integral(back, source)) < 1e-12


def test_weno_adapter_accepts_nonuniform_grid():
    nx = 32
    x = np.linspace(0, 1, nx) ** 1.4
    state = np.ones(nx, dtype=np.float32) * 0.7
    tensor = torch.from_numpy(state).unsqueeze(0)
    params = {
        "dt": torch.tensor([0.002]), "L_mm": torch.tensor([20.0]),
        "nu": torch.tensor([0.0]), "k": torch.tensor([0.0]),
        "E": torch.tensor([6.0]), "dTdx": torch.tensor([0.0]),
        "b_quad": torch.tensor([0.0]),
    }
    result, steps = WENOSolverAdapter(cfl=0.3).advance(
        tensor, torch.from_numpy(x.astype(np.float32)).unsqueeze(0), params
    )
    assert result.shape == tensor.shape
    assert torch.isfinite(result).all()
    assert torch.allclose(result, tensor, atol=2e-5, rtol=0)
    assert steps[0] > 0
