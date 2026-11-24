from fractions import Fraction as Q

import numpy as np
import pytest

from stage_ab import anchor_refinement, stage_a2
from stage_ab.anchor_refinement import AnchorTable
from stage_ab.budget import PacingPolicy
from stage_ab.burgers import FrozenFlux, exact_godunov
from stage_ab.stage_a2 import CertificateTables, CertifiedFlux, FPConstants, cell_costs, certified_rollout_a2


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(393)
    layers = [(rng.normal(size=(4, 2)), np.zeros(4)),
              (rng.normal(size=(1, 4)), np.zeros(1))]
    model = CertifiedFlux(FrozenFlux(layers))
    base = CertificateTables.build(model, envelope=2., bins=4)
    return model, base, AnchorTable.build(model, base, bins=7, chunk_size=8)


def test_signed_anchor_contains_exact_fp_errors_across_bins_and_ties(setup):
    model, base, anchors = setup
    rng = np.random.default_rng(718)
    xs = np.r_[anchors.edges, anchors.centers, 0., np.nextafter(0., 1.)]
    pairs = np.concatenate((np.array(np.meshgrid(xs, xs)).reshape(2, -1).T,
                            rng.uniform(-2, 2, (300, 2)), np.stack((xs, -xs), axis=1)))
    bounds = anchors.signed_errors(*pairs.T)
    actual = model.predict(pairs)
    for (a, b), f, lo, hi in zip(pairs, actual, bounds.lo, bounds.hi):
        e = Q(float(f)) - exact_godunov(Q(float(a)), Q(float(b)))
        assert Q(float(lo)) <= e <= Q(float(hi))
    anchors.check_contract(model, base)
    with pytest.raises(ValueError):
        anchors.coefficients.setflags(write=True)


def test_contract_and_envelope(setup):
    model, base, anchors = setup
    other = CertifiedFlux(FrozenFlux([(np.array([[1., 0.]]), np.zeros(1))]))
    with pytest.raises(ValueError, match="fingerprint"):
        anchors.check_contract(other, base)
    with pytest.raises(ValueError, match="envelope"):
        anchors.signed_errors(np.array([np.nextafter(2., np.inf)]), np.zeros(1))


def test_online_certificate_is_reference_free_and_monotone(setup, monkeypatch):
    model, base, anchors = setup
    state = np.array([-.1, .15, .1, -.3, 0.3, .27])
    consts = FPConstants.build(.2, base.envelope)
    coarse = cell_costs(model, base, consts, state, mode="table")
    checks = np.array([1, 2, 4])

    def forbidden(*args, **kwargs):
        raise AssertionError("Online anchor query attempted a network or reference evaluation")

    for module in (anchor_refinement, stage_a2):
        for name in ("godunov_gradient_cases", "godunov_value_enclosure", "network_value_and_gradient",
                     "network_value_enclosure"):
            monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(model, "predict", forbidden)
    actual = cell_costs(model, base, consts, state, mode="selective_anchor",
                        refiner=lambda *_: checks, anchor=anchors)
    np.testing.assert_array_equal(actual[[0, 3, 5]], coarse[[0, 3, 5]])
    assert np.all(actual[checks] <= coarse[checks])


def test_anchor_rollout_preserves_certified_budget(setup):
    model, base, anchors = setup
    initial = np.linspace(-.7, .7, 12)
    outcome = certified_rollout_a2(model, base, initial, steps=5, lam=.2, total_budget=.08,
                                   policy=PacingPolicy(), mode="selective_anchor", anchor=anchors,
                                   refiner=lambda v, k: np.argsort(-k[:, 1, 1])[:4], audit=True)
    assert outcome["spent"] <= outcome["total_budget"]
    assert all(r["step_certificate_holds"] and r["global_certificate_holds"] for r in outcome["rows"])
