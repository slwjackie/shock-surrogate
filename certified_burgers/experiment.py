"""Full PoC experiment requested in the research note.

Pipeline:
1. train a deliberately small surrogate against S^H,
2. audit oracle local error eta and three cheap observable scores,
3. run true autoregressive trust-or-fallback trajectories,
4. compare Always Solver / Always Neural / Oracle / Residual / Uncertainty,
5. sweep gate thresholds, audit Theorem 1, and measure actual wall-clock cost.

Residual and uncertainty scores remain empirical proxies. The code does not call
them certificates unless a separate theorem is established.
"""
from __future__ import annotations
import argparse, json
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .godunov import advance
from .initial_conditions import cell_centers, sample_states
from .oracle_error import l1_error
from .surrogate import TinyConvSurrogate
from .verifiers import conservation_defect, weak_residual_score, mc_dropout_uncertainty
from .rollout import (
    POLICIES, benchmark_policy_runtime, benchmark_primitives, compressive_shock_strength,
    rollout_diagnostics,
)
from .stability import stability_sweep

@dataclass
class ExperimentConfig:
    n_cells:int=128
    train_samples:int=768
    calib_samples:int=192
    test_samples:int=192
    epochs:int=60
    batch_size:int=64
    lr:float=2e-3
    horizon:int=1
    cfl:float=0.8
    max_abs_global:float=4.0
    seed:int=0
    mc_samples:int=8
    rollout_cases:int=8
    rollout_steps:int=24
    sweep_points:int=9
    gate_accept_quantile:float=0.8
    empirical_coverage:float=0.95
    runtime_repeats:int=3
    primitive_repeats:int=50
    make_plots:bool=True


def _reference_batch(u,dx,dt,horizon):
    y,_=advance(u,dx,horizon,dt=dt,boundary="periodic")
    return y


def _pearson(x,y):
    x=np.asarray(x,dtype=np.float64); y=np.asarray(y,dtype=np.float64)
    if np.std(x)<1e-14 or np.std(y)<1e-14: return 0.0
    return float(np.corrcoef(x,y)[0,1])


def _rank_capture(score,eta,frac=0.1):
    score=np.asarray(score); eta=np.asarray(eta); n=len(eta); k=max(1,int(round(frac*n)))
    bad=set(np.argsort(eta)[-k:]); flagged=set(np.argsort(score)[-k:])
    return float(len(bad&flagged)/k)


def _empirical_scale(score,eta,coverage=0.95):
    """Calibration-only multiplicative scale C for empirical eta <= C q coverage.

    This is diagnostic calibration, not a mathematical certificate.
    """
    score=np.asarray(score,dtype=np.float64); eta=np.asarray(eta,dtype=np.float64)
    positive=score[score>0]
    floor=max(1e-12,1e-6*float(np.median(positive)) if positive.size else 1e-12)
    ratio=eta/np.maximum(score,floor)
    return float(np.quantile(ratio,float(coverage))), float(floor)


def _shock_mask(shock_strength):
    s=np.asarray(shock_strength,dtype=np.float64)
    if not len(s): return np.zeros(0,dtype=bool)
    threshold=float(np.quantile(s,0.75))
    return (s>=threshold) & (s>1e-12)


def _audit_batch(model,states,*,dx,dt,horizon,mc_samples,device,seed):
    ref=_reference_batch(states,dx,dt,horizon)
    xt=torch.as_tensor(states,dtype=torch.float32,device=device)
    model.eval()
    with torch.no_grad(): pred=model(xt).detach().cpu().numpy()
    eta=l1_error(pred,ref,dx)
    q_cons=conservation_defect(states,pred,dx)
    q_weak=weak_residual_score(states,pred,dx=dx,dt=dt*horizon)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    q_unc=mc_dropout_uncertainty(model,xt,samples=mc_samples).detach().cpu().numpy()
    shock=np.asarray([compressive_shock_strength(s) for s in states],dtype=np.float64)
    return {"eta":eta,"conservation":q_cons,"weak_residual":q_weak,"uncertainty":q_unc,"shock_strength":shock}


def _audit_summary(audit, calibration_scales=None):
    eta=np.asarray(audit["eta"]); shock_mask=_shock_mask(audit["shock_strength"])
    result={
        "mean_eta":float(np.mean(eta)),"median_eta":float(np.median(eta)),"p95_eta":float(np.quantile(eta,0.95)),
        "shock_heavy_fraction":float(np.mean(shock_mask)),"shock_heavy_mean_eta":float(np.mean(eta[shock_mask])) if np.any(shock_mask) else None,
        "verifiers":{},
    }
    for name in ("conservation","weak_residual","uncertainty"):
        score=np.asarray(audit[name])
        row={
            "pearson_with_eta":_pearson(score,eta),"top10_error_capture":_rank_capture(score,eta,0.10),
            "mean_score":float(np.mean(score)),
            "shock_heavy_pearson":_pearson(score[shock_mask],eta[shock_mask]) if np.sum(shock_mask)>=3 else None,
        }
        if calibration_scales and name in calibration_scales:
            scale=calibration_scales[name]["scale"]; floor=calibration_scales[name]["floor"]
            q_scaled=scale*np.maximum(score,floor)
            row.update({
                "empirical_scale_from_calibration":float(scale),
                "empirical_upper_coverage":float(np.mean(eta<=q_scaled)),
                "mean_scaled_bound_over_eta":float(np.mean(q_scaled/np.maximum(eta,1e-12))),
            })
        result["verifiers"][name]=row
    return result


def _threshold_grid(values,points):
    values=np.asarray(values,dtype=np.float64)
    qs=np.linspace(0.0,1.0,max(2,int(points)))
    return [float(x) for x in np.unique(np.quantile(values,qs))]


def _aggregate_rollouts(results):
    if not results: raise ValueError("Need at least one rollout")
    return {
        "final_error":float(np.mean([r.final_error for r in results])),
        "p95_final_error":float(np.quantile([r.final_error for r in results],0.95)),
        "accept_rate":float(np.mean([r.accept_rate for r in results])),
        "fallback_rate":float(np.mean([r.fallback_rate for r in results])),
        "mean_accepted_eta_sum":float(np.mean([r.accepted_eta_sum for r in results])),
        "theorem_holds_all":bool(all(r.theorem_holds for r in results)),
        "max_theorem_violation":float(max(r.max_theorem_violation for r in results)),
    }


def _run_rollout_set(model,states,*,dx,dt,config,policy,threshold):
    return [rollout_diagnostics(
        model,s,dx=dx,dt=dt,horizon=config.horizon,macro_steps=config.rollout_steps,
        policy=policy,threshold=threshold,mc_samples=config.mc_samples,seed=config.seed+10000+j,
    ) for j,s in enumerate(states)]


def _policy_sweep(model,states,thresholds,*,dx,dt,config,policy):
    rows=[]
    for threshold in thresholds:
        agg=_aggregate_rollouts(_run_rollout_set(model,states,dx=dx,dt=dt,config=config,policy=policy,threshold=threshold))
        rows.append({"threshold":float(threshold),**agg})
    return rows


def _train_model(config,train_x,train_y,device):
    model=TinyConvSurrogate().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=config.lr); loss_fn=nn.MSELoss()
    ds=TensorDataset(torch.tensor(train_x,dtype=torch.float32),torch.tensor(train_y,dtype=torch.float32))
    loader=DataLoader(ds,batch_size=config.batch_size,shuffle=True)
    losses=[]; model.train()
    for _ in range(config.epochs):
        running=0.0; count=0
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device)
            opt.zero_grad(set_to_none=True); loss=loss_fn(model(xb),yb); loss.backward(); opt.step()
            running += float(loss.detach().cpu())*len(xb); count += len(xb)
        losses.append(running/max(count,1))
    return model,losses


def run(config:ExperimentConfig,out_path=None):
    if not (0.0<config.gate_accept_quantile<1.0): raise ValueError("gate_accept_quantile must be in (0,1)")
    if not (0.0<config.empirical_coverage<1.0): raise ValueError("empirical_coverage must be in (0,1)")
    torch.manual_seed(config.seed); np.random.seed(config.seed)
    _,dx=cell_centers(config.n_cells)
    # Fixed timestep chosen from a declared global state envelope, as required by the reference setup.
    dt=config.cfl*dx/config.max_abs_global
    train_x=sample_states(config.train_samples,config.n_cells,seed=config.seed,max_abs=1.0)
    calib_x=sample_states(config.calib_samples,config.n_cells,seed=config.seed+10,max_abs=1.0)
    test_id=sample_states(config.test_samples,config.n_cells,seed=config.seed+20,max_abs=1.0)
    test_ood=sample_states(config.test_samples,config.n_cells,seed=config.seed+30,max_abs=1.8,strong_ood=True)
    train_y=_reference_batch(train_x,dx,dt,config.horizon)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model,losses=_train_model(config,train_x,train_y,device)

    calibration=_audit_batch(model,calib_x,dx=dx,dt=dt,horizon=config.horizon,mc_samples=config.mc_samples,device=device,seed=config.seed+40)
    id_audit=_audit_batch(model,test_id,dx=dx,dt=dt,horizon=config.horizon,mc_samples=config.mc_samples,device=device,seed=config.seed+41)
    ood_audit=_audit_batch(model,test_ood,dx=dx,dt=dt,horizon=config.horizon,mc_samples=config.mc_samples,device=device,seed=config.seed+42)
    calibration_scales={}
    for name in ("conservation","weak_residual","uncertainty"):
        scale,floor=_empirical_scale(calibration[name],calibration["eta"],config.empirical_coverage)
        calibration_scales[name]={"scale":scale,"floor":floor,"target_coverage":config.empirical_coverage}

    q=config.gate_accept_quantile
    selected_thresholds={
        "oracle":float(np.quantile(calibration["eta"],q)),
        "residual":float(np.quantile(calibration["weak_residual"],q)),
        "uncertainty":float(np.quantile(calibration["uncertainty"],q)),
    }
    threshold_grids={
        "oracle":_threshold_grid(calibration["eta"],config.sweep_points),
        "residual":_threshold_grid(calibration["weak_residual"],config.sweep_points),
        "uncertainty":_threshold_grid(calibration["uncertainty"],config.sweep_points),
    }

    rollout_sets={
        "id":sample_states(config.rollout_cases,config.n_cells,seed=config.seed+50,max_abs=1.0),
        "ood":sample_states(config.rollout_cases,config.n_cells,seed=config.seed+60,max_abs=1.8,strong_ood=True),
    }
    baselines={}; sweeps={}; representative={}
    for split,states in rollout_sets.items():
        baselines[split]={}
        diagnostic_cache={}
        for policy in POLICIES:
            threshold=selected_thresholds.get(policy)
            rolls=_run_rollout_set(model,states,dx=dx,dt=dt,config=config,policy=policy,threshold=threshold)
            diagnostic_cache[policy]=rolls
            baselines[split][policy]=_aggregate_rollouts(rolls)
        solver_runtime=benchmark_policy_runtime(
            model,states,dx=dx,dt=dt,horizon=config.horizon,macro_steps=config.rollout_steps,
            policy="always_solver",repeats=config.runtime_repeats,mc_samples=config.mc_samples,seed=config.seed+70,
        )
        for policy in POLICIES:
            threshold=selected_thresholds.get(policy)
            runtime=solver_runtime if policy=="always_solver" else benchmark_policy_runtime(
                model,states,dx=dx,dt=dt,horizon=config.horizon,macro_steps=config.rollout_steps,
                policy=policy,threshold=threshold,repeats=config.runtime_repeats,mc_samples=config.mc_samples,seed=config.seed+70,
            )
            baselines[split][policy]["actual_runtime_sec"]=float(runtime)
            baselines[split][policy]["speedup_vs_solver"]=float(solver_runtime/max(runtime,1e-15))
        sweeps[split]={
            policy:_policy_sweep(model,states,threshold_grids[policy],dx=dx,dt=dt,config=config,policy=policy)
            for policy in ("oracle","residual","uncertainty")
        }
        representative[split]=diagnostic_cache["residual"][0]

    primitives=benchmark_primitives(
        model,test_id[0],dx=dx,dt=dt,horizon=config.horizon,mc_samples=config.mc_samples,
        repeats=config.primitive_repeats,seed=config.seed+80,
    )
    stability=stability_sweep(pairs=128,seed=config.seed+90)

    plot_paths=[]
    if config.make_plots:
        from .plots import plot_verifier_scatter,plot_error_fallback,plot_error_speedup,plot_theorem_bound
        base=Path(out_path).with_suffix("") if out_path is not None else Path("outputs/certified_burgers_poc")
        plot_dir=base.parent/(base.name+"_plots")
        id_shock=_shock_mask(id_audit["shock_strength"]); ood_shock=_shock_mask(ood_audit["shock_strength"])
        for name in ("conservation","weak_residual","uncertainty"):
            plot_paths.append(plot_verifier_scatter(plot_dir,name,id_audit["eta"],id_audit[name],ood_audit["eta"],ood_audit[name],id_shock,ood_shock))
        for split in ("id","ood"):
            plot_paths.append(plot_error_fallback(plot_dir,sweeps[split],tag=split))
            plot_paths.append(plot_error_speedup(plot_dir,baselines[split],tag=split))
            plot_paths.append(plot_theorem_bound(plot_dir,representative[split],name=f"residual_{split}"))

    payload={
        "config":asdict(config),"dx":float(dx),"dt_per_reference_step":float(dt),"device":str(device),
        "training":{"first_epoch_mse":float(losses[0]),"final_epoch_mse":float(losses[-1])},
        "one_step_audit":{
            "calibration":_audit_summary(calibration),
            "id":_audit_summary(id_audit,calibration_scales),
            "ood":_audit_summary(ood_audit,calibration_scales),
            "empirical_scales":calibration_scales,
        },
        "selected_gate_thresholds":selected_thresholds,
        "baselines":baselines,
        "threshold_sweeps":sweeps,
        "primitive_wall_clock":primitives,
        "godunov_l1_stability_sweep":stability,
        "plots":plot_paths,
        "notes":[
            "eta is an oracle diagnostic because it evaluates S^H at the algorithm state.",
            "Residual, conservation, and MC-dropout quantities are empirical proxies, not proven certificates.",
            "The empirical multiplicative scale is calibrated only to measure potential coverage; it is not a theorem.",
            "Oracle actual runtime includes computing S^H to reveal eta, so Oracle Gate is not deployable and need not be faster than Always Solver.",
            "Theorem-1 audit compares observed hybrid/all-Godunov trajectory deviation with the cumulative accepted oracle eta.",
        ],
    }
    if out_path is not None:
        path=Path(out_path); path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    return payload


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--n_cells",type=int,default=128); p.add_argument("--train_samples",type=int,default=768)
    p.add_argument("--calib_samples",type=int,default=192); p.add_argument("--test_samples",type=int,default=192)
    p.add_argument("--epochs",type=int,default=60); p.add_argument("--batch_size",type=int,default=64)
    p.add_argument("--horizon",type=int,default=1); p.add_argument("--seed",type=int,default=0)
    p.add_argument("--rollout_cases",type=int,default=8); p.add_argument("--rollout_steps",type=int,default=24)
    p.add_argument("--sweep_points",type=int,default=9); p.add_argument("--runtime_repeats",type=int,default=3)
    p.add_argument("--out",default="outputs/certified_burgers_poc.json"); p.add_argument("--no_plots",action="store_true")
    a=p.parse_args()
    cfg=ExperimentConfig(
        n_cells=a.n_cells,train_samples=a.train_samples,calib_samples=a.calib_samples,test_samples=a.test_samples,
        epochs=a.epochs,batch_size=a.batch_size,horizon=a.horizon,seed=a.seed,rollout_cases=a.rollout_cases,
        rollout_steps=a.rollout_steps,sweep_points=a.sweep_points,runtime_repeats=a.runtime_repeats,make_plots=not a.no_plots,
    )
    print(json.dumps(run(cfg,a.out),indent=2))

if __name__=="__main__": main()
