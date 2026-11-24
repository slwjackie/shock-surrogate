from fractions import Fraction

import numpy as np
import pytest

from certified_burgers.godunov import godunov_flux
from stage_ab.anchor_refinement import AnchorTable
from stage_ab.budget import PacingPolicy
from stage_ab.burgers import FrozenFlux, exact_godunov
from stage_ab.stage_a2 import (
    CertificateTables, CertifiedFlux, FPConstants, cell_costs, certified_rollout_a2, pairs,
)


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(83)
    model = CertifiedFlux(FrozenFlux([
        (rng.normal(size=(4, 2)), np.zeros(4)),
        (rng.normal(size=(1, 4)), np.array([.01])),
    ]))
    return model, CertificateTables.build(model, bins=4), FPConstants.build(.2, 2.)


def test_signed_endpoint_rows_enclose_rational_residuals(setup):
    model, tables, consts = setup
    tiny = np.nextafter(0., 1.)
    v = np.array([-2., 0., -0., tiny, -tiny, .6, -.6, -.6000000000000001, 1., .06, 2.])
    x = pairs(v)
    cases = [
        godunov_flux(x[:, 0], x[:, 1]),
        model.predict(x),
    ]
    reference = [exact_godunov(Fraction(a), Fraction(b)) for a, b in x]
    errors = [[Fraction(value) - g for value, g in zip(flux, reference)] for flux in cases]
    chosen = np.array([10, 0, 3, 4, 5, 6, 7])
    baseline = cell_costs(model, tables, consts, v, mode="table")
    point = cell_costs(model, tables, consts, v, mode="selective_point", refiner=lambda *_: chosen)
    replay = cell_costs(model, tables, consts, v, mode="selective_replay", refiner=lambda *_: chosen)
    assert np.all(replay <= point)
    for cost in (point, replay):
        assert np.all(cost <= baseline)
        unselected = np.setdiff1d(np.arange(len(v)), chosen)
        np.testing.assert_array_equal(cost[unselected], baseline[unselected])
        for i in chosen:
            for left in range(2):
                for right in range(2):
                    exact = abs(errors[right][i] - errors[left][(i - 1) % len(v)])
                    assert Fraction(cost[i, left, right]) >= exact


def test_replayed_predictions_are_batch_invariant(setup):
    model, _, _ = setup
    rng = np.random.default_rng(719)
    x = np.vstack((rng.uniform(-2., 2., (97, 2)), [[-0., 0.], [0., -0.]]))
    full = model.predict(x)
    for indices in ([98, 97], [97], [25], [13, 11, 22, 45, 8], np.arange(len(x))[::-1]):
        actual = model.predict(x[indices])
        np.testing.assert_array_equal(actual.view(np.uint64), full[indices].view(np.uint64))


@pytest.mark.parametrize("mode", ["selective_point", "selective_replay"])
def test_endpoint_rollout_passes_original_audit(setup, mode):
    model, tables, consts = setup
    result = certified_rollout_a2(
        model, tables, np.array([-.7, -.5, 0., .9, 1., 1.2]), steps=5,
        lam=consts.lam, total_budget=.4, policy=PacingPolicy(), mode=mode,
        refiner=lambda state, costs: np.argsort(-costs[:, 1, 1])[:3], audit=True)
    assert result["spent"] <= result["total_budget"]
    assert all(r["step_certificate_holds"] and r["global_certificate_holds"] for r in result["rows"])


def test_two_tier_selection_tightens_only_requested_anchor_rows(setup):
    model, tables, consts = setup
    anchor = AnchorTable.build(model, tables, bins=9)
    v = np.array([-2., .06, -.2, -.01, 0., .21, 1.2, 1., -.7])
    all_rows = np.arange(len(v))
    anchor_rows = cell_costs(*setup, v, mode="selective_anchor", anchor=anchor,
                             refiner=lambda *_: all_rows)
    point_rows = cell_costs(*setup, v, mode="selective_point", refiner=lambda *_: all_rows)
    for chosen in (all_rows[:0], np.array([8, 0, 3]), all_rows):
        def choose(state, starting_costs):
            np.testing.assert_array_equal(starting_costs, anchor_rows)
            assert not state.flags.writeable and not starting_costs.flags.writeable
            return chosen
        result = cell_costs(*setup, v, mode="anchor_then_point", refiner=choose, anchor=anchor)
        expected = anchor_rows.copy()
        expected[chosen] = np.minimum(anchor_rows[chosen], point_rows[chosen])
        np.testing.assert_array_equal(result, expected)

    rollout = certified_rollout_a2(
        model, tables, v, steps=7, lam=consts.lam, total_budget=.32,
        policy=PacingPolicy(), mode="anchor_then_point", anchor=anchor,
        refiner=lambda state, costs: np.argsort(-costs[:, 1, 1])[:3], audit=True)
    assert rollout["spent"] <= rollout["total_budget"]
    assert any(0 < r["trusted"] < len(v) for r in rollout["rows"])
    assert all(r["step_certificate_holds"] and r["global_certificate_holds"] for r in rollout["rows"])
