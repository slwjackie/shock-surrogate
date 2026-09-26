import json
from fractions import Fraction as Q
import numpy as np
import pytest
import torch
from stage_ab.interval import Interval as I, Dual, upper_fraction
from stage_ab.burgers import (FrozenFlux, FluxCertificate, exact_godunov,
    certified_proposal, exact_local_error, certified_rollout, fourier_counterexample,
    fallback_step, train_flux)
from certified_burgers.godunov import godunov_flux
from certified_burgers.surrogate import ConservativeFluxSurrogate


def frozen():
    return FrozenFlux([(np.array([[1., 0.], [0., -1.], [1., -1.]]), np.array([0., 0., .1])),
                       (np.array([[.3, .4, .2]]), np.array([.01]))])


def test_intervals_enclose_exact_rational_arithmetic():
    rng = np.random.default_rng(1)
    for a, b in rng.uniform(-10, 10, (100, 2)):
        x, y = I(a), I(b)
        for result, exact in [(x+y,Q(a)+Q(b)), (x-y,Q(a)-Q(b)), (x*y,Q(a)*Q(b)), (x/y,Q(a)/Q(b))]:
            assert Q(result.lo) <= exact <= Q(result.hi)
    assert I(-2, 3)**2 == I(0, np.nextafter(9., np.inf))
    with pytest.raises(ArithmeticError): I(-1, 1).reciprocal()
    assert I(2).exp().contains(np.exp(2))
    assert I(2).log().contains(np.log(2))


def test_interval_dual_gradient():
    x, y = Dual.variable(I(2), 0, 2), Dual.variable(I(3), 1, 2)
    r = I(2)*x*y+x**2
    assert r.value.contains(16)
    assert r.grad[0].contains(10) and r.grad[1].contains(4)


def test_exact_flux_matches_existing_riemann_flux():
    for a in [-2., -1., 0., .5, 2.]:
        for b in [-2., -1., 0., .5, 2.]:
            assert float(exact_godunov(Q(a), Q(b))) == godunov_flux(a, b)


def test_fourier_simultaneous_blind_spot():
    row = fourier_counterexample()
    assert row['conservation'] < 1e-14
    assert row['weak_residual'] < 1e-12
    assert row['eta'] > .03
    with pytest.raises(ValueError): fourier_counterexample(n=16)


def test_interval_flux_table_encloses_float_inference_and_real_flux():
    f = frozen(); table = FluxCertificate.build(f, bins=4)
    rng = np.random.default_rng(3)
    samples = np.r_[rng.uniform(-2,2,(128,2)), [[-2,-2],[2,2],[0,0],[1,-1]]]
    for pair, prediction, delta in zip(samples, f.predict(samples), table.lookup(f,samples)):
        error = abs(Q(float(prediction))-exact_godunov(Q(float(pair[0])),Q(float(pair[1]))))
        assert error <= Q(float(delta))


def test_certificate_roundtrip_and_tampering(tmp_path):
    f=frozen(); t=FluxCertificate.build(f,bins=2)
    f.save(tmp_path/'m.json');t.save(tmp_path/'t.json')
    g=FrozenFlux.load(tmp_path/'m.json');loaded=FluxCertificate.load(tmp_path/'t.json',g)
    assert g.digest==f.digest
    assert np.array_equal(loaded.bounds,t.bounds)
    d=json.loads((tmp_path/'t.json').read_text());d['bounds'][0][0]=0
    (tmp_path/'t.json').write_text(json.dumps(d))
    with pytest.raises(ValueError):FluxCertificate.load(tmp_path/'t.json',g)


def test_certificate_mismatched_model_and_domain_fail_closed():
    f=frozen();t=FluxCertificate.build(f,bins=2)
    with pytest.raises(ValueError):t.lookup(f,np.array([[3.,0.]]))
    f.layers[0][1][0]+=.1
    with pytest.raises(ValueError):t.lookup(f,np.array([[0.,0.]]))


def test_one_step_and_global_certificate_including_roundoff():
    f=frozen();t=FluxCertificate.build(f,bins=4)
    rng=np.random.default_rng(4)
    for _ in range(20):
        v=rng.uniform(-1,1,16)
        p,bound,_=certified_proposal(f,t,v,.2,1/16)
        assert exact_local_error(p,v,.2,1/16)<=bound
    r=certified_rollout(f,t,v,steps=8,step_tolerance=10,global_budget=80)
    assert all(x['global_bound_holds'] for x in r['rows'])
    assert all(x['certificate_holds'] for x in r['rows'])
    reject=certified_rollout(f,t,v,steps=5,step_tolerance=0,global_budget=0)
    assert reject['acceptance_rate']==0
    assert all(x['global_bound_holds'] for x in reject['rows'])
    with pytest.raises(ValueError):certified_rollout(f,t,v,lam=1.)


def test_accepting_runtime_does_not_call_godunov(monkeypatch):
    import stage_ab.burgers as b
    f=frozen();t=FluxCertificate.build(f,bins=2)
    monkeypatch.setattr(b,'fallback_step',lambda *a: (_ for _ in ()).throw(AssertionError('oracle leakage')))
    r=certified_rollout(f,t,np.full(16,.5),steps=2,step_tolerance=100,global_budget=200,audit=False)
    assert r['acceptance_rate']==1


def test_existing_cnn_flux_theorem_and_gauge():
    torch.manual_seed(3)
    model=ConservativeFluxSurrogate(horizon=1,dropout=0).eval()
    u=torch.randn(1,1,32)*.3
    with torch.no_grad(): flux=model.flux_net(u)[0,0].double().numpy()
    v=u[0,0].double().numpy();g=godunov_flux(v,np.roll(v,-1));lam=.2;h=1/32
    p=v-lam*(flux-np.roll(flux,1));s=v-lam*(g-np.roll(g,1))
    assert h*np.abs(p-s).sum()<=2*lam*h*np.abs(flux-g).sum()+1e-15
    shifted=v-lam*((flux+10)-np.roll(flux+10,1))
    assert np.allclose(p,shifted)


def test_stage_a_smoke(tmp_path):
    from stage_ab.experiments import stage_a
    torch.set_num_threads(1)
    payload=stage_a(tmp_path,smoke=True,epochs=2,bins=2)
    assert payload['all_global_certificates_hold']
    assert (tmp_path/'results.json').is_file()
    assert any(x['acceptance']>0 for x in payload['certified_threshold_sweep']['id'])


@pytest.mark.parametrize('method',['interval','lipschitz','hybrid'])
def test_certificate_methods_and_refinement(method,tmp_path):
    f=frozen();table=FluxCertificate.build(f,bins=2,subdivisions=2,method=method)
    samples=np.random.default_rng(11).uniform(-2,2,(40,2))
    for pair,prediction,delta in zip(samples,f.predict(samples),table.lookup(f,samples)):
        assert abs(Q(float(prediction))-exact_godunov(Q(float(pair[0])),Q(float(pair[1]))))<=Q(float(delta))
    table.save(tmp_path/'t.json');loaded=FluxCertificate.load(tmp_path/'t.json',f)
    assert loaded.subdivisions==2


def test_legacy_proposal_verifier_budget_adapter():
    from certified_burgers.interfaces import Proposal
    from stage_ab.adapters import CertifiedFluxAdvice,CertifiedFluxVerifier,ErrorBudgetPolicy
    f=frozen();t=FluxCertificate.build(f,bins=2);lam=.2;h=1/16;v=np.full(16,.5)
    advice=CertifiedFluxAdvice(f,lam);verifier=CertifiedFluxVerifier(f,t,lam,h)
    proposal=Proposal(v,advice.predict(v),1,lam*h)
    result=verifier.evaluate(proposal)
    assert not result.hard_failure
    assert ErrorBudgetPolicy(10,10).decide(result).accept
    assert not ErrorBudgetPolicy(0,0).decide(result).accept
    bad=Proposal(v,proposal.candidate+.01,1,lam*h)
    assert verifier.evaluate(bad).hard_failure
