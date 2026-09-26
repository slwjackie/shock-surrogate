import numpy as np
import pytest
import torch
from stage_ab.chemistry import Chemistry,State,normal_shock,reflected_shock,ignition_metrics
from stage_ab.hydrogen import (make_cases,generate_data,train_chemistry,NeuralChemistry,
                              rollout,calibrate,local_audit,case_initial)
from stage_ab.kinetics_bounds import IntervalKinetics,certify_linear_slab,growth_bound
from stage_ab.interval import Interval as I


@pytest.fixture
def chem():
    return Chemistry()


def test_constant_volume_mass_elements_energy_and_rhs(chem):
    s=chem.fresh(T=1100)
    rhs=chem.rhs(s.vector,s.rho)
    assert abs(rhs[1:].sum())<1e-12
    assert np.max(np.abs(chem.elements@rhs[1:]))<1e-12
    out=chem.step(s,1e-5)
    assert chem.guard(s,out)[0]
    assert abs(chem.energy(out)-chem.energy(s))<1e-3
    assert out.rho==s.rho


def test_raw_physical_guard_does_not_normalize_invalid_candidate(chem):
    s=chem.fresh()
    wrong=State(s.T,s.rho,s.Y*1.01)
    assert not chem.guard(s,wrong)[0]
    negative=s.Y.copy();negative[1]=-.01
    assert not chem.guard(s,State(s.T,s.rho,negative))[0]
    assert not chem.guard(s,State(float('nan'),s.rho,s.Y))[0]
    assert not chem.guard(s,State(s.T,-s.rho,s.Y))[0]


def test_projection_preserves_elements_and_recovers_internal_energy(chem):
    s=chem.fresh(); y=s.Y.copy();y[chem.names.index('H2O')]+=.001
    p,correction=chem.project(y,s)
    assert chem.guard(s,p)[0]
    assert correction>0


def test_shock_rankine_hugoniot_and_reflected_wall(chem):
    s=chem.fresh(T=300)
    chem.set(s);p1=chem.gas.P;h1=chem.gas.enthalpy_mass
    p,meta=normal_shock(chem,s,3.)
    chem.set(p);p2=chem.gas.P;h2=chem.gas.enthalpy_mass
    u1,u2=meta['wave_speed_m_s'],meta['downstream_shock_frame_velocity_m_s']
    assert np.isclose(s.rho*u1,p.rho*u2,rtol=1e-12)
    assert np.isclose(p1+s.rho*u1**2,p2+p.rho*u2**2,rtol=1e-12)
    assert np.isclose(h1+.5*u1**2,h2+.5*u2**2,rtol=1e-11)
    reflected,wall=reflected_shock(chem,s,3.)
    assert reflected.T>p.T
    assert abs(wall['reflected_downstream_lab_velocity_m_s'])<1e-7
    assert np.array_equal(s.Y,reflected.Y)
    with pytest.raises(ValueError):normal_shock(chem,s,1.)


def test_no_ignition_is_censored_not_tend(chem):
    s=chem.fresh(T=600);times=np.array([0.,1e-8,1e-7])
    result=ignition_metrics(times,chem.trajectory(s,times),chem.names)
    assert result['censored'] and result['temperature_threshold_delay_s'] is None
    s=chem.fresh(T=1200);times=np.r_[0.,np.geomspace(1e-8,3e-4,80)]
    result=ignition_metrics(times,chem.trajectory(s,times),chem.names)
    assert result['ignited'] and result['temperature_threshold_delay_s']>0


def test_interval_rhs_matches_cantera_within_box(chem):
    s=chem.fresh(T=1100)
    s=chem.step(s,2e-5)
    k=IntervalKinetics(chem)
    for T in [900.,1100.,1800.,2500.]:
        z=np.r_[T,s.Y]
        box=[I(float(x)-1e-9,float(x)+1e-9) for x in z]
        ranges=k.rhs(box,I(s.rho));reference=chem.rhs(z,s.rho)
        assert all(v.contains(float(r)) for v,r in zip(ranges,reference))


def test_interval_certificate_and_fail_closed_thermo_breakpoint(chem):
    s=chem.fresh(T=1100);k=IntervalKinetics(chem)
    result=certify_linear_slab(k,s,chem.step(s,1e-10),1e-10,pieces=1,tube_radius=1e-5)
    assert result.available and result.bound>=0
    s=chem.fresh(T=1000)
    result=certify_linear_slab(k,s,s,1e-8,pieces=1,tube_radius=1e-3)
    assert not result.available and 'breakpoint' in result.reason
    s=chem.fresh(T=1100);wrong=State(1300,s.rho,s.Y)
    assert not certify_linear_slab(k,s,wrong,1e-6,tube_radius=1e-6,pieces=1).available


def test_gronwall_outward_growth():
    import mpmath as mp
    mp.mp.dps=60
    for mu in [-10.,0.,1.,10.]:
        expected=mp.e**(mp.mpf(mu)*mp.mpf(.2))*.01
        expected+=mp.mpf(.3)*(mp.expm1(mp.mpf(mu)*mp.mpf(.2))/mu if mu else mp.mpf(.2))
        assert mp.mpf(growth_bound(.01,.3,mu,.2))>=expected


def test_trajectory_split_checkpoint_and_actual_gates(chem,tmp_path):
    torch.set_num_threads(1)
    cases=make_cases(train_cases=2,other_cases=1,seed=11)
    data,manifest,_=generate_data(chem,cases,points=10,final_time=3e-4)
    assert len(manifest)==len(cases)
    assert not set(r[-1] for r in data['train']) & set(r[-1] for r in data['test_id'])
    with pytest.raises(ValueError):generate_data(chem,[cases[0],cases[0]],points=10)
    with pytest.raises(ValueError):train_chemistry(chem,data['test_id'],epochs=1)
    net,_=train_chemistry(chem,data['train'],epochs=1,width=8)
    net.save(tmp_path/'m.json');loaded=NeuralChemistry.load(tmp_path/'m.json',chem)
    s,_=case_initial(chem,cases[0]);times=np.linspace(0,3e-4,5)
    result=rollout(chem,loaded,s,times,policy='residual',threshold=-1,diagnostics=True)
    assert result['acceptance']==0
    baseline=rollout(chem,loaded,s,times,policy='always_solver_restarted')
    assert np.allclose(result['states'][-1].vector,baseline['states'][-1].vector)
    assert (tmp_path/'m.json').is_file()


def test_stage_b_smoke(tmp_path):
    from stage_ab.experiments import stage_b
    torch.set_num_threads(1)
    p=stage_b(tmp_path,smoke=True,epochs=1)
    assert len(p['results'])==5
    assert p['restricted_exact_ode_certificate_probe']['available']
    assert (tmp_path/'reference_trajectories.npz').is_file()
    assert p['mechanism']['species'][0]=='H2'


def test_complete_ode_trajectory_audit_requires_all_slabs(chem):
    from stage_ab.kinetics_bounds import audit_complete_trajectory
    s=chem.fresh(T=1100);times=np.array([0.,1e-10,2e-10])
    states=chem.trajectory(s,times)
    report=audit_complete_trajectory(IntervalKinetics(chem),states,times,tube_radius=1e-5)
    assert report['available'] and report['final_bound']>=0
    bad=[s,State(1400,s.rho,s.Y),states[-1]]
    report=audit_complete_trajectory(IntervalKinetics(chem),bad,times,tube_radius=1e-5)
    assert not report['available'] and report['final_bound'] is None


def test_interval_derivatives_contain_numerical_probe(chem):
    s=chem.step(chem.fresh(T=1100),2e-5);z=s.vector
    widths=np.r_[.001,np.full(len(s.Y),1e-7)]
    box=[I(float(x-w),float(x+w)) for x,w in zip(z,widths)]
    jac=IntervalKinetics(chem).jacobian_enclosure(box,s.rho)
    for j in [0,1,2,5]:
        step=1e-3 if j==0 else 1e-8
        left,right=z.copy(),z.copy();left[j]-=step;right[j]+=step
        deriv=(chem.rhs(right,s.rho)-chem.rhs(left,s.rho))/(2*step)
        for i,val in enumerate(deriv):
            assert jac[i][j].lo-1e-5<=val<=jac[i][j].hi+1e-5


def test_stiffness_report_not_a_certificate(chem):
    from stage_ab.hydrogen import stiffness_diagnostic
    r=stiffness_diagnostic(chem,chem.fresh(T=1100),1e-6)
    assert r['rigorous'] is False
    assert np.isfinite(r['spectral_radius'])
    assert r['max_real_eigenvalue']>=r['min_real_eigenvalue']
