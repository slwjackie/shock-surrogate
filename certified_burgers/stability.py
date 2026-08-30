"""Numerical audit of the L1 non-expansiveness expected from monotone Godunov steps."""
from __future__ import annotations
import numpy as np
from .godunov import godunov_step
from .oracle_error import l1_error


def stability_sweep(*, n_cells_values=(64,128,256), cfl_values=(0.2,0.4,0.6,0.8,1.0), pairs=256,
                    max_abs=0.9, seed=0):
    rng=np.random.default_rng(seed); rows=[]
    for n in n_cells_values:
        dx=1.0/int(n)
        u=rng.uniform(-max_abs,max_abs,size=(pairs,int(n)))
        v=rng.uniform(-max_abs,max_abs,size=(pairs,int(n)))
        before=l1_error(u,v,dx)
        for cfl in cfl_values:
            dt=float(cfl)*dx/max_abs
            su,_=godunov_step(u,dx,dt=dt); sv,_=godunov_step(v,dx,dt=dt)
            after=l1_error(su,sv,dx)
            ratio=after/np.maximum(before,1e-15)
            rows.append({
                "n_cells":int(n),"cfl":float(cfl),"pairs":int(pairs),
                "max_l1_ratio":float(np.max(ratio)),"p99_l1_ratio":float(np.quantile(ratio,0.99)),
                "mean_l1_ratio":float(np.mean(ratio)),"violations_gt_1_plus_1e-12":int(np.sum(ratio>1.0+1e-12)),
            })
    return rows
