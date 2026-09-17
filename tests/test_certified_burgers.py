import numpy as np
import pytest
import torch
from certified_burgers.counterexamples import audit_verifier_counterexamples
from certified_burgers.fine_reference import (
    fine_godunov_reference,
    prolong_piecewise_constant,
    restrict_cell_average,
)
from certified_burgers.godunov import godunov_flux,godunov_step
from certified_burgers.horizon_study import run_horizon_study
from certified_burgers.initial_conditions import smooth_state
from certified_burgers.interfaces import ThresholdPolicy, VerifierResult
from certified_burgers.oracle_error import l1_error,oracle_gate
from certified_burgers.surrogate import TinyConvSurrogate
from certified_burgers.verifiers import conservation_defect,weak_residual_score
from certified_burgers.rollout import rollout_diagnostics
from certified_burgers.splits import make_data_splits
from certified_burgers.stability import stability_sweep
from certified_burgers.validation import godunov_validation_suite
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

def test_threshold_policy_hard_failure_overrides_a_small_score():
    policy=ThresholdPolicy(threshold=1.0)
    decision=policy.decide(VerifierResult("guarded",0.0,hard_failure=True))
    assert not decision.accept
    assert decision.reason=="hard_verifier_failure"

def test_refined_reference_preserves_constant_and_cell_average():
    n=32; dx=1/n; dt=0.2*dx
    state=np.full(n,0.4)
    prolonged=prolong_piecewise_constant(state,4)
    assert np.allclose(restrict_cell_average(prolonged,4),state)
    refined=fine_godunov_reference(state,dx=dx,dt=dt,coarse_steps=3,factor=4)
    assert np.allclose(refined,state)

def test_role_separated_splits_have_no_exact_overlap():
    _,manifest=make_data_splits(
        n_cells=32,train_samples=16,calib_samples=8,test_samples=8,
        rollout_cases=4,seed=11,
    )
    assert manifest["all_exact_overlap_counts_zero"]
    assert not any(manifest["exact_cross_split_overlap_counts"].values())

def test_mass_preserving_shift_exposes_conservation_blind_spot():
    n=64; dx=1/n; dt=0.2*dx
    current=smooth_state(n,amplitude=0.7)
    reference,_=godunov_step(current,dx,dt=dt)
    audit=audit_verifier_counterexamples(
        current,reference,dx=dx,elapsed_time=dt,
        thresholds={"conservation":1e-12,"residual":1e9},
    )
    shifted=next(row for row in audit["cases"] if row["name"]=="mass_preserving_shock_shift")
    assert shifted["oracle_eta_vs_same_grid_solver"] > 0.0
    assert shifted["conservation"] < 1e-12
    assert shifted["accepted_by_conservation_threshold"]

def test_config_enforces_same_final_time_horizon_rule():
    assert ExperimentConfig(horizon=4,reference_steps=24).macro_steps==6
    with pytest.raises(ValueError,match="divisible"):
        ExperimentConfig(horizon=5,reference_steps=24).validate()

def test_godunov_validation_checks_conservation_maximum_principle_and_tvd():
    report=godunov_validation_suite(n_cells=64,cfl=0.8,seed=3)
    assert report["periodic_mass_max_abs_error"] < 1e-12
    assert report["maximum_principle_violations"]==0
    assert report["tvd_violations"]==0

def test_stability_sweep_has_no_l1_expansion_in_sample():
    rows=stability_sweep(n_cells_values=(32,),cfl_values=(0.4,0.8,1.0),pairs=32,seed=3)
    assert all(r["violations_gt_1_plus_1e-12"]==0 for r in rows)
    assert max(r["max_l1_ratio"] for r in rows) <= 1.0+1e-12

def test_end_to_end_experiment_smoke(tmp_path):
    cfg=ExperimentConfig(n_cells=32,train_samples=32,calib_samples=12,test_samples=12,epochs=1,batch_size=8,
                         horizon=1,mc_samples=2,rollout_cases=2,reference_steps=3,
                         fine_reference_factor=2,sweep_points=3,runtime_repeats=1,
                         primitive_repeats=2,make_plots=False,seed=5)
    payload=run(cfg,tmp_path/"smoke.json")
    assert set(payload["baselines"]["id"])=={
        "always_solver","always_neural","oracle","conservation","residual","uncertainty"
    }
    assert payload["baselines"]["id"]["always_solver"]["final_error_vs_same_grid_solver"] < 1e-12
    assert payload["baselines"]["id"]["always_neural"]["theorem_holds_all"]
    assert set(payload["threshold_sweeps"]["ood"])=={
        "oracle","conservation","residual","uncertainty"
    }
    assert payload["primitive_wall_clock"]["godunov_macro_step_sec"] >= 0.0
    assert payload["comparison_contract"]["total_reference_steps"]==3
    assert payload["data_split_audit"]["all_exact_overlap_counts_zero"]
    assert payload["one_step_audit"]["id"]["triangle_inequality_violations"]==0
    assert "controlled_verifier_failure_probes" in payload

def test_horizon_study_keeps_split_and_final_time_fixed(tmp_path):
    cfg=ExperimentConfig(
        n_cells=16,train_samples=16,calib_samples=4,test_samples=4,epochs=1,
        batch_size=4,mc_samples=2,rollout_cases=1,reference_steps=2,
        fine_reference_factor=1,sweep_points=2,runtime_repeats=1,
        primitive_repeats=1,make_plots=False,seed=9,
    )
    summary=run_horizon_study(cfg,horizons=(1,2),out_dir=tmp_path/"horizons")
    assert summary["fairness_checks"]["identical_split_hashes"]
    assert summary["fairness_checks"]["identical_reference_step_count"]
    assert summary["fairness_checks"]["identical_final_physical_time"]
    assert [row["macro_steps"] for row in summary["rows"]]==[2,1]
