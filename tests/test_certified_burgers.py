import numpy as np
from certified_burgers.godunov import godunov_flux,godunov_step
from certified_burgers.oracle_error import l1_error,oracle_gate
from certified_burgers.verifiers import conservation_defect,weak_residual_score

def test_godunov_flux_shock_and_rarefaction():
    assert np.isclose(godunov_flux(np.array([1.0]),np.array([0.0]))[0],0.5)
    assert np.isclose(godunov_flux(np.array([-1.0]),np.array([1.0]))[0],0.0)

def test_periodic_constant_state_and_mass_are_preserved():
    n=64; dx=1/n; u=np.full(n,0.4)
    out,_=godunov_step(u,dx,dt=0.5*dx); assert np.allclose(out,u)
    rng=np.random.default_rng(0); u=rng.uniform(-0.8,0.8,size=n)
    out,_=godunov_step(u,dx,dt=0.5*dx)
    assert np.isclose(np.sum(out),np.sum(u),atol=1e-12)
    assert conservation_defect(u[None],out[None],dx)[0]<1e-12

def test_numerical_l1_nonexpansiveness_under_cfl():
    rng=np.random.default_rng(1); n=128; dx=1/n
    u=rng.uniform(-0.8,0.8,size=n); v=rng.uniform(-0.8,0.8,size=n); dt=0.5*dx
    su,_=godunov_step(u,dx,dt=dt); sv,_=godunov_step(v,dx,dt=dt)
    assert l1_error(su[None],sv[None],dx)[0] <= l1_error(u[None],v[None],dx)[0]+1e-12

def test_weak_residual_proxy_zero_for_stationary_constant_state():
    n=64; dx=1/n; dt=0.5*dx; u=np.full((2,n),0.3)
    score=weak_residual_score(u,u.copy(),dx=dx,dt=dt)
    assert np.all(score<1e-10)

def test_oracle_gate_uses_reference_on_rejected_samples():
    dx=0.25; ref=np.zeros((2,4)); pred=np.array([[0.,0.,0.,0.],[1.,0.,0.,0.]])
    hybrid,accept,eta=oracle_gate(pred,ref,dx,threshold=0.1)
    assert accept.tolist()==[True,False]
    assert np.allclose(hybrid,ref)
    assert np.allclose(eta,[0.,0.25])
