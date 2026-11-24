import numpy as np
import pytest

from stage_ab.budget import PacingPolicy
from stage_ab.burgers import FrozenFlux
from stage_ab.stage_a2 import (
    CertificateTables, CertifiedFlux, FPConstants, cell_costs, certified_rollout_a2,
)


@pytest.fixture(scope="module")
def certificate():
    rng = np.random.default_rng(41)
    model = CertifiedFlux(FrozenFlux([
        (rng.normal(size=(4, 2)), rng.normal(size=4)),
        (rng.normal(size=(1, 4)), rng.normal(size=1)),
    ]))
    tables = CertificateTables.build(model, envelope=2., bins=4)
    return model, tables, FPConstants.build(.2, 2.)


@pytest.mark.parametrize("chosen", [[], [7, 0, 3, 1], np.arange(8)])
def test_refinement_matches_full_grid_on_exactly_the_selected_rows(certificate, chosen):
    state = np.array([-2., -.14, 0., 0., 1.49, -.3, -.72, 2.])
    table = cell_costs(*certificate, state, mode="table")
    both = cell_costs(*certificate, state, mode="both")
    expected = table.copy()
    expected[chosen] = both[chosen]

    def refiner(v, costs):
        np.testing.assert_array_equal(v, state)
        np.testing.assert_array_equal(costs, table)
        # Even a selector that makes its private input copies writable cannot
        # alter the state or pass uncertified costs to the router.
        v.setflags(write=True)
        costs.setflags(write=True)
        v[:] = 0
        costs[:] = 0
        return chosen

    original = state.copy()
    result = cell_costs(*certificate, state, mode="selective", refiner=refiner)
    np.testing.assert_array_equal(result.view(np.uint64), expected.view(np.uint64))
    np.testing.assert_array_equal(state, original)


@pytest.mark.parametrize("chosen", [[-1], [2], [1., 0.], [True], [1, 1], [[1]]])
def test_refinement_rejects_invalid_indices(certificate, chosen):
    with pytest.raises(ValueError, match="Refiner"):
        cell_costs(*certificate, np.array([.2, .1]), mode="selective", refiner=lambda *_: chosen)


def test_selective_rollout_is_audited_at_every_step(certificate):
    model, tables, consts = certificate
    state = np.linspace(-1., 1., 8)
    result = certified_rollout_a2(
        model, tables, state, steps=4, lam=consts.lam, total_budget=.5,
        policy=PacingPolicy(), mode="selective", refiner=lambda v, costs: np.argsort(-costs[:, 1, 1])[:3],
        audit=True, record=False,
    )
    assert len(result["rows"]) == 4
    assert not result["menus"]
    assert result["spent"] <= result["total_budget"]
    for row in result["rows"]:
        assert row["step_certificate_holds"] and row["global_certificate_holds"]
