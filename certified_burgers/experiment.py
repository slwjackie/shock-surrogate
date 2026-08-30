"""End-to-end PoC: train advice, compute oracle error, and audit cheap verifiers."""
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
from .oracle_error import l1_error, oracle_gate
from .surrogate import TinyConvSurrogate
from .verifiers import conservation_defect, weak_residual_score, mc_dropout_uncertainty

@dataclass
class ExperimentConfig:
    n_cells:int=128; train_samples:int=768; test_samples:int=192; epochs:int=60
    batch_size:int=64; lr:float=2e-3; horizon:int=1; cfl:float=0.8
    max_abs_global:float=2.0; seed:int=0; mc_samples:int=8

def _reference_batch(u,dx,dt,horizon):
    y,_=advance(u,dx,horizon,dt=dt,boundary="periodic")
    return y

def _pearson(x,y):
    x=np.asarray(x,dtype=np.float64); y=np.asarray(y,dtype=np.float64)
    if np.std(x)<1e-14 or np.std(y)<1e-14: return 0.0
    return float(np.corrcoef(x,y)[0,1])

def _rank_capture(score,eta,frac=0.1):
    n=len(eta); k=max(1,int(round(frac*n)))
    bad=set(np.argsort(eta)[-k:]); flagged=set(np.argsort(score)[-k:])
    return len(bad&flagged)/k

def run(config:ExperimentConfig,out_path=None):
    torch.manual_seed(config.seed); np.random.seed(config.seed)
    _,dx=cell_centers(config.n_cells)
    dt=config.cfl*dx/config.max_abs_global
    train_x=sample_states(config.train_samples,config.n_cells,seed=config.seed,max_abs=1.0)
    train_y=_reference_batch(train_x,dx,dt,config.horizon)
    test_id=sample_states(config.test_samples,config.n_cells,seed=config.seed+1,max_abs=1.0)
    test_ood=sample_states(config.test_samples,config.n_cells,seed=config.seed+2,max_abs=1.8,strong_ood=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=TinyConvSurrogate().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=config.lr); loss_fn=nn.MSELoss()
    ds=TensorDataset(torch.tensor(train_x,dtype=torch.float32),torch.tensor(train_y,dtype=torch.float32))
    loader=DataLoader(ds,batch_size=config.batch_size,shuffle=True)
    model.train()
    for _ in range(config.epochs):
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device)
            opt.zero_grad(set_to_none=True); loss=loss_fn(model(xb),yb); loss.backward(); opt.step()
    results={}; model.eval()
    for name,states in (("id",test_id),("ood",test_ood)):
        ref=_reference_batch(states,dx,dt,config.horizon)
        xt=torch.tensor(states,dtype=torch.float32,device=device)
        with torch.no_grad(): pred_t=model(xt)
        pred=pred_t.cpu().numpy()
        eta=l1_error(pred,ref,dx)
        q_cons=conservation_defect(states,pred,dx)
        q_weak=weak_residual_score(states,pred,dx=dx,dt=dt*config.horizon)
        q_unc=mc_dropout_uncertainty(model,xt,samples=config.mc_samples).cpu().numpy()
        oracle_threshold=float(np.median(eta))
        hybrid,accept,_=oracle_gate(pred,ref,dx,oracle_threshold)
        hybrid_error=l1_error(hybrid,ref,dx)
        def audit(score):
            return {"pearson_with_eta":_pearson(score,eta),"top10_error_capture":_rank_capture(score,eta,0.10),"mean":float(np.mean(score))}
        results[name]={
            "mean_eta":float(np.mean(eta)),"median_eta":float(np.median(eta)),"p95_eta":float(np.quantile(eta,0.95)),
            "oracle_gate":{"threshold":oracle_threshold,"accept_rate":float(np.mean(accept)),"fallback_rate":float(1-np.mean(accept)),"mean_hybrid_error":float(np.mean(hybrid_error))},
            "verifiers":{"conservation":audit(q_cons),"weak_residual":audit(q_weak),"mc_dropout":audit(q_unc)}
        }
    payload={"config":asdict(config),"dx":dx,"dt_per_reference_step":dt,"device":str(device),"results":results,
             "note":"Verifier scores are empirical proxies only. Oracle eta uses the Godunov reference and is not deployable."}
    if out_path is not None:
        path=Path(out_path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    return payload

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--n_cells",type=int,default=128); p.add_argument("--train_samples",type=int,default=768)
    p.add_argument("--test_samples",type=int,default=192); p.add_argument("--epochs",type=int,default=60)
    p.add_argument("--batch_size",type=int,default=64); p.add_argument("--horizon",type=int,default=1)
    p.add_argument("--seed",type=int,default=0); p.add_argument("--out",default="outputs/certified_burgers_poc.json")
    a=p.parse_args()
    cfg=ExperimentConfig(n_cells=a.n_cells,train_samples=a.train_samples,test_samples=a.test_samples,epochs=a.epochs,batch_size=a.batch_size,horizon=a.horizon,seed=a.seed)
    print(json.dumps(run(cfg,a.out),indent=2))
if __name__=="__main__": main()
