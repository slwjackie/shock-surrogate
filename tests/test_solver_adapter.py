import numpy as np
import torch

from sim.solver_burgers_weno import advance_state
from solvers.weno_adapter import WENOSolverAdapter


def test_weno_arbitrary_state_step_is_finite():
    nx = 64
    x = np.linspace(0.0, 20.0, nx)
    state = 0.5 + np.exp(-((x - 1.0) / 0.3) ** 2)
    advanced, steps = advance_state(state, dt_total=0.01, L_mm=20.0)
    assert advanced.shape == state.shape
    assert np.isfinite(advanced).all()
    assert steps > 0
    tensor = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
    x_norm = torch.linspace(0, 1, nx).unsqueeze(0)
    params = {
        "dt": torch.tensor([0.01]), "L_mm": torch.tensor([20.0]),
        "nu": torch.tensor([0.002]), "k": torch.tensor([1.5]),
        "E": torch.tensor([6.0]), "dTdx": torch.tensor([0.0]),
        "b_quad": torch.tensor([0.0]),
    }
    out, counts = WENOSolverAdapter().advance(tensor, x_norm, params)
    assert out.shape == tensor.shape
    assert torch.isfinite(out).all()
    assert counts[0] > 0
