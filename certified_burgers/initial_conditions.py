"""Initial-condition generators for ID and deliberately shifted Burgers tests."""
from __future__ import annotations
import numpy as np

def cell_centers(n_cells: int, domain=(0.0, 1.0)):
    a, b = map(float, domain)
    dx = (b-a)/n_cells
    x = a + (np.arange(n_cells)+0.5)*dx
    return x, dx

def riemann_state(n_cells, left, right, *, location=0.5, domain=(0.0,1.0)):
    x,_ = cell_centers(n_cells, domain)
    return np.where(x < location, left, right).astype(np.float64)

def three_state(n_cells,left,middle,right,*,x1=0.35,x2=0.65,domain=(0.0,1.0)):
    x,_=cell_centers(n_cells,domain)
    return np.where(x<x1,left,np.where(x<x2,middle,right)).astype(np.float64)

def smooth_state(n_cells, *, amplitude=0.8, offset=0.0, phase=0.0, mode=1, domain=(0.0,1.0)):
    x,_ = cell_centers(n_cells, domain)
    length = float(domain[1]-domain[0])
    xi=(x-domain[0])/length
    return offset + amplitude*np.sin(2*np.pi*mode*xi+phase)

def sample_states(count, n_cells, *, seed=0, max_abs=1.0, strong_ood=False):
    rng=np.random.default_rng(seed); states=[]
    amp_hi=max_abs
    amp_lo=(0.65 if strong_ood else 0.35)*max_abs
    n_kinds=4 if strong_ood else 3
    for j in range(count):
        kind=j%n_kinds
        if kind in (0,1):
            a=rng.uniform(-amp_hi,amp_hi); b=rng.uniform(-amp_hi,amp_hi)
            if abs(a-b)<amp_lo:
                b=np.clip(a+rng.choice([-1.0,1.0])*amp_lo,-amp_hi,amp_hi)
            left,right=(max(a,b),min(a,b)) if kind==0 else (min(a,b),max(a,b))
            states.append(riemann_state(n_cells,left,right,location=rng.uniform(0.2,0.8)))
        elif kind==2:
            mode=int(rng.integers(1,5 if strong_ood else 3))
            amplitude=rng.uniform(0.25*max_abs,max_abs)
            offset=rng.uniform(-0.15*max_abs,0.15*max_abs)
            phase=rng.uniform(0,2*np.pi)
            u=smooth_state(n_cells,amplitude=amplitude,offset=offset,phase=phase,mode=mode)
            states.append(np.clip(u,-max_abs,max_abs))
        else:
            # OOD-only interacting pattern: a compressive jump followed by an expansive jump.
            middle=rng.uniform(-0.35*max_abs,0.35*max_abs)
            left=rng.uniform(max(middle+0.4*max_abs,0.45*max_abs),max_abs)
            right=rng.uniform(max(middle+0.35*max_abs,0.4*max_abs),max_abs)
            x1=rng.uniform(0.2,0.4); x2=rng.uniform(0.6,0.8)
            states.append(three_state(n_cells,left,middle,right,x1=x1,x2=x2))
    return np.stack(states,axis=0)
