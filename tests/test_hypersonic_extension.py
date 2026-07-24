import torch

from hypersonic.benchmarks import (
    planar_hypersonic_shock_2d,
    planar_hypersonic_shock_3d,
    uniform_flow_1d,
    uniform_flow_3d,
)
from hypersonic.losses import high_frequency_spectral_loss, lgno_loss
from hypersonic.models.lgno2d import LocalGlobalNeuralOperator2d
from hypersonic.models.mesh_gnn import ConservativeMeshGNN
from hypersonic.positivity import is_admissible, positivity_preserving_blend
from hypersonic.solver import CompressibleConfig, StructuredCompressibleSolver
from hypersonic.state import conservative_to_primitive, primitive_to_conservative


def test_state_roundtrip_and_positivity_blend():
    rho = torch.tensor([[1.0, 0.8, 1.2]])
    u = torch.tensor([[2.0, -0.3, 0.5]])
    v = torch.tensor([[0.2, 0.1, -0.4]])
    p = torch.tensor([[1.0, 0.7, 2.0]])
    U = primitive_to_conservative(rho, u, v, p)
    rho2, u2, v2, p2 = conservative_to_primitive(U)
    assert torch.allclose(rho, rho2)
    assert torch.allclose(u, u2)
    assert torch.allclose(v, v2)
    assert torch.allclose(p, p2, atol=1e-6)
    bad = U.clone()
    bad[:, -1] = 1e-8
    safe, theta = positivity_preserving_blend(U, bad, rho_floor=1e-5, p_floor=1e-5)
    assert bool(is_admissible(safe, rho_floor=1e-5, p_floor=1e-5).all())
    assert torch.all(theta < 1.0)


def test_uniform_1d_euler_and_navier_stokes_are_preserved():
    U = uniform_flow_1d(nx=32, u=3.0)
    for equation, viscosity in (("euler", 0.0), ("navier_stokes", 1e-3)):
        solver = StructuredCompressibleSolver(CompressibleConfig(equation=equation, viscosity=viscosity, bc_x="periodic"))
        out, _ = solver.advance(U, dt_total=1e-3, dx=1.0 / 32)
        assert torch.allclose(out, U, atol=1e-6, rtol=1e-6)


def test_mach10_planar_shock_stays_finite_and_positive():
    U = planar_hypersonic_shock_2d(nx=24, ny=16, mach=10.0, shock_normal_angle_deg=10.0)
    solver = StructuredCompressibleSolver(CompressibleConfig(cfl=0.25, bc_x="outflow", bc_y="outflow"))
    out, steps = solver.advance(U, dt_total=2e-4, dx=1.0 / 24, dy=1.0 / 16)
    assert steps > 0
    assert bool(torch.isfinite(out).all())
    assert bool(is_admissible(out).all())


def test_uniform_3d_euler_and_navier_stokes_preserved():
    U = uniform_flow_3d(nx=8, ny=6, nz=4, u=2.0, v=0.2, w=-0.1)
    for equation, viscosity in (("euler", 0.0), ("navier_stokes", 1e-3)):
        solver = StructuredCompressibleSolver(
            CompressibleConfig(equation=equation, viscosity=viscosity, bc_x="periodic", bc_y="periodic", bc_z="periodic")
        )
        out, _ = solver.advance(U, 1e-4, 1.0 / 8, 1.0 / 6, 1.0 / 4)
        assert torch.allclose(out, U, atol=2e-6, rtol=2e-6)


def test_3d_hypersonic_shock_is_positive():
    U = planar_hypersonic_shock_3d(nx=8, ny=6, nz=4, mach=6.0, normal=(1.0, 0.2, 0.1))
    solver = StructuredCompressibleSolver(CompressibleConfig(cfl=0.2))
    out, steps = solver.advance(U, 2e-5, 1.0 / 8, 1.0 / 6, 1.0 / 4)
    assert steps > 0
    assert bool(torch.isfinite(out).all())
    assert bool(is_admissible(out).all())


def test_lgno_shape_positivity_and_periodic_conservation():
    b, ny, nx = 2, 12, 16
    rho = 1.0 + 0.05 * torch.rand(b, ny, nx)
    u = 0.2 * torch.randn(b, ny, nx)
    v = 0.2 * torch.randn(b, ny, nx)
    p = 1.0 + 0.05 * torch.rand(b, ny, nx)
    state = primitive_to_conservative(rho, u, v, p)
    model = LocalGlobalNeuralOperator2d(width=16, modes_x=6, modes_y=4, depth=2, boundary="periodic")
    pred, aux = model(state, dt=0.01)
    assert pred.shape == state.shape
    assert aux["increment"].shape == state.shape
    assert torch.all(aux["positivity_theta"] >= 0.0)
    assert torch.all(aux["positivity_theta"] <= 1.0)
    assert torch.allclose(pred.mean(dim=(-2, -1)), state.mean(dim=(-2, -1)), atol=2e-5, rtol=2e-5)


def test_mesh_gnn_works_on_unstructured_graph():
    positions = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    edge_index = torch.tensor([[0, 1, 0, 2, 1, 3, 2, 3, 1, 0, 2, 0, 3, 1, 3, 2], [1, 0, 2, 0, 3, 1, 3, 2, 0, 1, 0, 2, 1, 3, 2, 3]])
    rho = torch.ones(1, 4)
    u = torch.zeros(1, 4)
    v = torch.zeros(1, 4)
    p = torch.ones(1, 4)
    state = primitive_to_conservative(rho, u, v, p).squeeze(0).T
    model = ConservativeMeshGNN(state_channels=4, coordinate_dim=2, hidden=16, message_passing_steps=2)
    pred, aux = model(state, positions, edge_index, dt=0.01)
    assert pred.shape == state.shape
    assert aux["increment"].shape == state.shape
    assert torch.allclose(pred.mean(dim=0), state.mean(dim=0), atol=2e-5, rtol=2e-5)


def test_lgno_loss_is_finite_and_zero_for_exact_prediction():
    target = torch.randn(2, 4, 8, 10)
    loss, parts = lgno_loss(target, target)
    assert float(loss) == 0.0
    assert float(parts["physical"]) == 0.0
    assert float(high_frequency_spectral_loss(target, target)) == 0.0
