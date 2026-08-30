import numpy as np
import torch
from certified_burgers.godunov import godunov_flux,godunov_step
from certified_burgers.initial_conditions import smooth_state
from certified_burgers.oracle_error import l1_error,oracle_gate
from certified_burgers.surrogate import TinyConvSurrogate
from certified_burgers.verifiers import conservation_defect,weak_residual_score
from certified_burgers.rollout import rollout_diagnostics
from certified_burgers.stability import stability_sweep
from certified_burgers.experiment import ExperimentConfig,run

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

def _identity_surrogate():
    model=TinyConvSurrogate(width=4,depth=1,dropout=0.0)
    with torch.no_grad():
        for p in model.parameters(): p.zero_()
    return model

def test_autoregressive_rollout_empirically_satisfies_theorem1_bound():
    n=64; dx=1/n; dt=0.4*dx/2.0
    u0=smooth_state(n,amplitude=0.7,phase=0.2)
    result=rollout_diagnostics(_identity_surrogate(),u0,dx=dx,dt=dt,horizon=1,macro_steps=12,policy="always_neural")
    assert result.theorem_holds
    assert result.trajectory_deviation[-1] <= result.cumulative_accepted_eta[-1] + 1e-10

def test_residual_gate_threshold_extremes_switch_between_solver_and_neural():
    n=48; dx=1/n; dt=0.4*dx/2.0; u0=smooth_state(n,amplitude=0.5)
    model=_identity_surrogate()
    reject=rollout_diagnostics(model,u0,dx=dx,dt=dt,macro_steps=4,policy="residual",threshold=-1.0)
    accept=rollout_diagnostics(model,u0,dx=dx,dt=dt,macro_steps=4,policy="residual",threshold=1e9)
    assert np.isclose(reject.fallback_rate,1.0)
    assert np.isclose(accept.accept_rate,1.0)

def test_stability_sweep_has_no_l1_expansion_in_sample():
    rows=stability_sweep(n_cells_values=(32,),cfl_values=(0.4,0.8,1.0),pairs=32,seed=3)
    assert all(r["violations_gt_1_plus_1e-12"]==0 for r in rows)
    assert max(r["max_l1_ratio"] for r in rows) <= 1.0+1e-12

def test_end_to_end_experiment_smoke(tmp_path):
    cfg=ExperimentConfig(n_cells=32,train_samples=32,calib_samples=12,test_samples=12,epochs=1,batch_size=8,
                         horizon=1,mc_samples=2,rollout_cases=2,rollout_steps=3,sweep_points=3,
                         runtime_repeats=1,primitive_repeats=2,make_plots=False,seed=5)
    payload=run(cfg,tmp_path/"smoke.json")
    assert set(payload["baselines"]["id"])=={"always_solver","always_neural","oracle","residual","uncertainty"}
    assert payload["baselines"]["id"]["always_solver"]["final_error"] < 1e-12
    assert payload["baselines"]["id"]["always_neural"]["theorem_holds_all"]
    assert set(payload["threshold_sweeps"]["ood"])=={"oracle","residual","uncertainty"}
    assert payload["primitive_wall_clock"]["godunov_macro_step_sec"] >= 0.0
