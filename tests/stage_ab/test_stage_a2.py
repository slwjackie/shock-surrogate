"""Stage A2 regression and soundness tests.

Every certificate is checked against an exact-rational oracle; algorithms are
checked against brute force or an exact LP.  Hypothesis is used for property
tests when installed, otherwise a seeded random sweep runs.
"""
from fractions import Fraction as Q
import itertools
import math

import numpy as np
import pytest

from stage_ab.burgers import FrozenFlux, exact_godunov
from stage_ab.vinterval import (VI, rigorous_nonneg_sum, network_value_enclosure,
                                network_value_and_gradient, godunov_value_enclosure,
                                godunov_gradient_cases, difference_enclosure)
from stage_ab.stage_a2 import (CertifiedFlux, CertificateTables, FPConstants, cell_costs,
                               face_bounds, difference_bounds, trust_frontier, pattern_cost,
                               rigorous_step_cost, frontier_upper, mixed_step, exact_step_error,
                               certified_rollout_a2, sub_edges, pairs)
from stage_ab.budget import (GreedyPolicy, AllOrNothingPolicy, PacingPolicy, ThresholdPolicy,
                             RobustifiedPolicy, DensityThresholdPolicy, run_policy_on_menus,
                             lp_exact, lp_dual_bound)
from certified_burgers.godunov import godunov_flux
from certified_burgers.initial_conditions import smooth_state, riemann_state


def random_flux(seed, width=6):
    rng = np.random.default_rng(seed)
    return CertifiedFlux(FrozenFlux([
        (rng.normal(size=(width, 2)), rng.normal(size=width)*.3),
        (rng.normal(size=(width, width))/np.sqrt(width), rng.normal(size=width)*.3),
        (rng.normal(size=(1, width))/np.sqrt(width), rng.normal(size=1)*.1)]))


def godunov_like_flux():
    """A hand-built ReLU flux close to Godunov (so certificates are non-trivial)."""
    w1 = np.array([[1., 0.], [0., -1.], [1., -1.], [.5, .5]])
    b1 = np.array([0., 0., .1, -.2])
    w2 = np.array([[.5, .45, .05, .02]])
    return CertifiedFlux(FrozenFlux([(w1, b1), (w2, np.array([.01]))]))


@pytest.fixture(scope="module")
def setup():
    model = godunov_like_flux()
    tables = CertificateTables.build(model, envelope=2., bins=6)
    return model, tables, FPConstants.build(.2, 2.)


# ------------------------------------------------------------------ intervals
def test_vector_intervals_enclose_exact_arithmetic():
    rng = np.random.default_rng(0)
    special = np.array([5e-324, -5e-324, 2.0**-1022, 1e300, -1e-300, 0.0, 1.0, -3.0])
    a = np.r_[rng.uniform(-10, 10, 200), special]
    b = np.r_[rng.uniform(-10, 10, 200), special[::-1]]
    x, y = VI.point(a), VI.point(b)
    for op, ex in [(x+y, lambda p, q: p+q), (x-y, lambda p, q: p-q), (x*y, lambda p, q: p*q),
                   (x.mul_point(b), lambda p, q: p*q)]:
        for i in range(len(a)):
            e = ex(Q(float(a[i])), Q(float(b[i])))
            assert Q(float(op.lo[i])) <= e <= Q(float(op.hi[i]))


def test_rigorous_nonneg_sum_dominates_exact_sum():
    rng = np.random.default_rng(1)
    for _ in range(50):
        t = rng.uniform(0, 1, rng.integers(1, 300))*10.0**rng.integers(-300, 5)
        assert rigorous_nonneg_sum(t) >= sum((Q(float(x)) for x in t), Q(0))
    with pytest.raises(ValueError):
        rigorous_nonneg_sum(np.array([1., -1.]))


def test_sub_edges_cover_bins_exactly():
    edges = np.linspace(-2., 2., 8)
    S = sub_edges(edges, 7)
    assert np.array_equal(S[:, 0], edges[:-1]) and np.array_equal(S[:, -1], edges[1:])
    assert np.all(np.diff(S, axis=1) >= 0)


def test_network_value_gradient_and_difference_enclosures():
    rng = np.random.default_rng(2)
    for seed in range(6):
        f = random_flux(seed)
        c = rng.uniform(-2, 2, (60, 2))
        r = rng.uniform(0, .4, (60, 2))*rng.choice([0., 1e-7, 1.], (60, 2))
        A, B = VI(c[:, 0]-r[:, 0], c[:, 0]+r[:, 0]), VI(c[:, 1]-r[:, 1], c[:, 1]+r[:, 1])
        val = network_value_enclosure(f.layers, A, B)
        v2, _, _ = network_value_and_gradient(f.layers, A, B)
        gv = godunov_value_enclosure(A, B)
        for i in range(60):
            x = np.array([[rng.uniform(A.lo[i], A.hi[i]), rng.uniform(B.lo[i], B.hi[i])]])
            y = np.array([[rng.uniform(A.lo[i], A.hi[i]), rng.uniform(B.lo[i], B.hi[i])]])
            fp, real = f.predict(x)[0], f.real_predict(x)[0]
            assert val.lo[i] <= fp <= val.hi[i]
            assert Q(float(val.lo[i])) <= real <= Q(float(val.hi[i]))
            assert Q(float(v2.lo[i])) <= real <= Q(float(v2.hi[i]))
            gx = exact_godunov(Q(float(x[0, 0])), Q(float(x[0, 1])))
            assert Q(float(gv.lo[i])) <= gx <= Q(float(gv.hi[i]))
            gy = exact_godunov(Q(float(y[0, 0])), Q(float(y[0, 1])))
            d = (f.real_predict(y)[0]-gy) - (real-gx)
            da = VI.point(y[:, 0]) - VI.point(x[:, 0])
            db = VI.point(y[:, 1]) - VI.point(x[:, 1])
            J = difference_enclosure(f.layers, A[i:i+1], B[i:i+1], da, db)
            assert Q(float(J.lo[0])) <= d <= Q(float(J.hi[0]))


def test_godunov_gradient_branch_cases():
    rng = np.random.default_rng(3)
    lo = rng.uniform(-2, 2, (500, 2)); w = rng.uniform(0, .5, (500, 2))
    A, B = VI(lo[:, 0], lo[:, 0]+w[:, 0]), VI(lo[:, 1], lo[:, 1]+w[:, 1])
    has_p, gpa, has_q, gqb = godunov_gradient_cases(A, B)
    assert np.all(has_p | has_q)
    for i in range(500):
        a, b = rng.uniform(A.lo[i], A.hi[i]), rng.uniform(B.lo[i], B.hi[i])
        p, q = max(a, 0)**2/2, min(b, 0)**2/2
        if p > q:
            assert has_p[i] and gpa.lo[i] <= max(a, 0) <= gpa.hi[i]
        elif q > p:
            assert has_q[i] and gqb.lo[i] <= min(b, 0) <= gqb.hi[i]


# ------------------------------------------------------------------ FP lemmas
def test_godunov_fp_error_bound_including_adversarial_inputs():
    c = FPConstants.build(.2, 2.)
    rng = np.random.default_rng(4)
    a = np.r_[rng.uniform(-2, 2, 2000), [5e-324, -1e-320, 1.5, -1.5, 2.0, 5e-324, 1e-310]]
    b = np.r_[rng.uniform(-2, 2, 2000), [-1e-323, 1e-320, -1.5-2**-52, 1.5, -2.0, -1e-323, -1e-310]]
    # ul + ur = -2^-1074 exercises the signed-zero branch of the shock speed test.
    a = np.r_[a, 3*2.0**-1074]; b = np.r_[b, -4*2.0**-1074]
    fl = godunov_flux(a, b)
    for x, y, z in zip(a, b, fl):
        assert abs(Q(float(z))-exact_godunov(Q(float(x)), Q(float(y)))) <= Q(c.eps_godunov)


def test_update_defect_and_apriori_bound(setup):
    model, tables, consts = setup
    rng = np.random.default_rng(5)
    for _ in range(20):
        n = int(rng.integers(4, 40))
        v = rng.uniform(-1.9, 1.9, n)
        trust = rng.random(n) < .5
        new, rho, p, flux = mixed_step(model, consts, v, trust)
        lq = Q(consts.lam)
        exact = sum((abs(Q(float(p[i])) - (Q(float(v[i])) - lq*(Q(float(flux[i]))-Q(float(flux[i-1])))))
                     for i in range(n)), Q(0))/n
        assert exact <= rho
        fmax = max(tables.flux_abs_max, 2.0*1.01)
        assert rho <= consts.update_defect_apriori(fmax, n)


# -------------------------------------------------------------- certificates
def _random_states(rng, n, count):
    out = [smooth_state(n, amplitude=.9, offset=.1, phase=1.), riemann_state(n, 1.2, -.6),
           riemann_state(n, -.6, 1.2), np.full(n, .3)]
    out += [rng.uniform(-1.9, 1.9, n) for _ in range(count)]
    return out


@pytest.mark.parametrize("mode", ["separable", "table", "local", "both"])
def test_one_step_mixed_certificate_is_sound(setup, mode):
    model, tables, consts = setup
    rng = np.random.default_rng(6)
    for v in _random_states(rng, 12, 6):
        costs = cell_costs(model, tables, consts, v, mode=mode)
        for trust in [np.ones(12, bool), np.zeros(12, bool), rng.random(12) < .5, rng.random(12) < .8]:
            new, rho, _, _ = mixed_step(model, consts, v, trust)
            bound = rigorous_step_cost(costs, trust, consts.lam, 12) + rho
            assert exact_step_error(v, new, consts.lam) <= bound
            assert np.all(np.abs(new) <= 2.)


def test_difference_bound_dominates_actual_face_differences(setup):
    model, tables, consts = setup
    rng = np.random.default_rng(7)
    for v in _random_states(rng, 16, 4):
        n = len(v)
        pred = model.predict(pairs(v))
        e = [Q(float(pred[i])) - exact_godunov(Q(float(v[i])), Q(float(v[(i+1) % n]))) for i in range(n)]
        for mode in ("table", "local"):
            D = difference_bounds(model, tables, v, mode=mode)
            dl = face_bounds(model, tables, v, mode=mode)
            for i in range(n):
                assert abs(e[i]-e[i-1]) <= Q(float(D[i]))
                assert abs(e[i]) <= Q(float(dl[i]))


def test_separable_certificate_does_not_vanish_but_difference_does(setup):
    model, tables, consts = setup
    out = {}
    for n in (32, 128):
        v = smooth_state(n, amplitude=.8, offset=.1, phase=.3)
        out[n] = {m: float(rigorous_step_cost(cell_costs(model, tables, consts, v, mode=m),
                                              np.ones(n, bool), .2, n)) for m in ("separable", "table", "local")}
    assert out[128]["separable"] > .9*out[32]["separable"]
    assert out[128]["table"] < .5*out[32]["table"]
    assert out[128]["local"] < .5*out[32]["local"]


# ------------------------------------------------------------------- frontier
def test_trust_frontier_matches_brute_force():
    rng = np.random.default_rng(8)
    for n in range(2, 9):
        for _ in range(8):
            costs = rng.uniform(0, 1, (n, 2, 2))
            fr = trust_frontier(costs)
            best = np.full(n+1, np.inf)
            for s in itertools.product([0, 1], repeat=n):
                c = pattern_cost(costs, s).sum()
                best[sum(s)] = min(best[sum(s)], c)
            assert np.allclose(fr.cost, best, rtol=1e-12, atol=1e-14)
            for k in range(n+1):
                s = fr.select(k)
                assert s.sum() == k
                assert math.isclose(pattern_cost(costs, s).sum(), fr.cost[k], rel_tol=1e-12, abs_tol=1e-14)


def test_frontier_upper_bounds_rigorous_cost(setup):
    model, tables, consts = setup
    v = riemann_state(24, 1.1, -.4)
    costs = cell_costs(model, tables, consts, v, mode="both")
    fr = trust_frontier(costs)
    cbar = frontier_upper(fr, consts.lam, 24)
    for k in range(25):
        s = fr.select(k)
        exact = Q(consts.lam)*sum((Q(float(x)) for x in pattern_cost(costs, s)), Q(0))/24
        assert exact <= Q(float(cbar[k]))


# -------------------------------------------------------------------- rollout
@pytest.mark.parametrize("policy", [GreedyPolicy(), GreedyPolicy(cap=1e-3), AllOrNothingPolicy(),
                                    PacingPolicy(), ThresholdPolicy(1e2, 1e6),
                                    RobustifiedPolicy(ThresholdPolicy(1e2, 1e6), GreedyPolicy(), .3)])
def test_certified_rollout_bounds_and_budget(setup, policy):
    model, tables, _ = setup
    v0 = smooth_state(16, amplitude=1., offset=.2, phase=.5)
    r = certified_rollout_a2(model, tables, v0, steps=5, lam=.2, total_budget=2e-3,
                             policy=policy, mode="both", audit=True)
    assert r["spent"] <= r["total_budget"]
    assert all(x["step_certificate_holds"] for x in r["rows"])
    assert all(x["global_certificate_holds"] for x in r["rows"])


def test_rollout_without_audit_never_calls_exact_oracle(setup, monkeypatch):
    import stage_ab.stage_a2 as s2
    model, tables, _ = setup
    boom = lambda *a, **k: (_ for _ in ()).throw(AssertionError("oracle leakage"))
    monkeypatch.setattr(s2, "exact_step_error", boom)
    monkeypatch.setattr(s2, "exact_godunov", boom)
    r = certified_rollout_a2(model, tables, np.full(16, .4), steps=3, lam=.2, total_budget=1.,
                             policy=GreedyPolicy(), audit=False)
    assert r["mean_trusted_fraction"] == 1.0


def test_rollout_rejects_bad_contracts(setup):
    model, tables, _ = setup
    with pytest.raises(ValueError):
        certified_rollout_a2(model, tables, np.full(8, .1), steps=2, lam=1., total_budget=1., policy=GreedyPolicy())
    with pytest.raises(ValueError):
        certified_rollout_a2(model, tables, np.full(8, 2.5), steps=2, lam=.2, total_budget=1., policy=GreedyPolicy())
    with pytest.raises(ValueError):
        certified_rollout_a2(model, tables, np.full(8, .1), steps=2, lam=.2, total_budget=0., policy=GreedyPolicy())


def test_tables_roundtrip_and_tamper(setup, tmp_path):
    import json
    model, tables, _ = setup
    tables.save(tmp_path/"t.json")
    loaded = CertificateTables.load(tmp_path/"t.json", model)
    assert np.array_equal(loaded.delta, tables.delta)
    d = json.loads((tmp_path/"t.json").read_text())
    d["delta"][0][0] = 0.
    (tmp_path/"t.json").write_text(json.dumps(d))
    with pytest.raises(ValueError):
        CertificateTables.load(tmp_path/"t.json", model)


# --------------------------------------------------------------------- budget
def _random_menus(rng, T, L, U, max_w):
    menus = []
    for _ in range(T):
        m = int(rng.integers(1, 6))
        w = np.sort(rng.uniform(0, max_w, m))
        dens = rng.uniform(L, U, m)
        values = np.r_[0., w*dens]
        menus.append((values, np.r_[0., w]))
    return menus


def test_threshold_policy_competitive_bound_against_exact_lp():
    rng = np.random.default_rng(9)
    for trial in range(40):
        L, U = 1.0, float(10**rng.uniform(0.5, 4))
        B = 1.0
        menus = _random_menus(rng, int(rng.integers(5, 40)), L, U, max_w=float(rng.uniform(.01, .6)))
        pol = ThresholdPolicy(L, U)
        res = run_policy_on_menus(pol, menus, B)
        assert res["weight"] <= B*(1+1e-12)
        w_hat = max(float(np.max(w)) for _, w in menus)/B
        lp = lp_exact(menus, B)
        assert lp <= math.exp(pol.alpha*w_hat)*(pol.alpha*res["value"] + B*L/math.e)*(1+1e-9)
        assert lp <= lp_dual_bound(menus, B)*(1+1e-9)


def test_greedy_and_pacing_lower_bound_instances():
    L, U, m, B = 1.0, 1000.0, 100, 1.0
    menus = [(np.array([0., L*B/m]), np.array([0., B/m]))]*m + [(np.array([0., U*B/m]), np.array([0., B/m]))]*m
    g = run_policy_on_menus(GreedyPolicy(), menus, B)["value"]
    t = run_policy_on_menus(ThresholdPolicy(L, U), menus, B)["value"]
    opt = lp_exact(menus, B)
    assert opt/g >= .99*U/L
    assert opt <= math.exp((1+math.log(U/L))/m)*((1+math.log(U/L))*t + B*L/math.e)
    # Pacing: all value is offered at the first step only.
    T = 50
    burst = [(np.r_[0., U*np.linspace(B/m, B, m)], np.r_[0., np.linspace(B/m, B, m)])] + \
            [(np.array([0.]), np.array([0.]))]*(T-1)
    p = run_policy_on_menus(PacingPolicy(), burst, B)["value"]
    assert lp_exact(burst, B) >= .9*T*p


def test_robustified_policy_budget_and_best_of_both():
    rng = np.random.default_rng(10)
    for _ in range(30):
        L, U = 1.0, 300.0
        menus = _random_menus(rng, 30, L, U, .2)
        gamma = float(rng.uniform(.1, .9))
        advice = DensityThresholdPolicy(float(rng.uniform(L, U)))
        combo = RobustifiedPolicy(ThresholdPolicy(L, U), advice, gamma)
        res = run_policy_on_menus(combo, menus, 1.0)
        assert res["weight"] <= 1.0*(1+1e-12)
        a = run_policy_on_menus(DensityThresholdPolicy(advice.rho), menus, 1-gamma)["value"]
        r = run_policy_on_menus(ThresholdPolicy(L, U), menus, gamma)["value"]
        assert res["value"] >= max(a, r)*(1-1e-12)


def test_rollout_enforces_budget_even_for_a_rogue_policy(setup):
    from stage_ab.budget import Policy

    class Rogue(Policy):
        name = "rogue"

        def propose(self, values, weights):
            return len(values)-1          # ignores the budget entirely

    model, tables, _ = setup
    v0 = smooth_state(16, amplitude=1., offset=.2, phase=.5)
    r = certified_rollout_a2(model, tables, v0, steps=6, lam=.2, total_budget=5e-4,
                             policy=Rogue(), audit=True)
    assert r["spent"] <= r["total_budget"]
    assert any(x["budget_override"] for x in r["rows"])
    assert all(x["global_certificate_holds"] for x in r["rows"])
