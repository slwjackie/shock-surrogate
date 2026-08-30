"""Smooth-test-function weak residual proxy for one Burgers time slab.

This does NOT call the Godunov solver. It is an observable candidate score,
not a rigorous certificate until a theorem is proved.
"""
from __future__ import annotations
import numpy as np
from ..godunov import burgers_flux

def weak_residual_score(current,prediction,*,dx:float,dt:float,num_modes:int=4):
    u0=np.asarray(current,dtype=np.float64); u1=np.asarray(prediction,dtype=np.float64)
    if u0.shape!=u1.shape: raise ValueError("current and prediction must have identical shapes.")
    if dt<=0 or dx<=0: raise ValueError("dt and dx must be positive.")
    n=u0.shape[-1]
    x=(np.arange(n)+0.5)/n
    u_mid=0.5*(u0+u1); f_mid=burgers_flux(u_mid); du=u1-u0
    scores=[]
    for k in range(1,num_modes+1):
        omega=2*np.pi*k
        for trig in ("sin","cos"):
            if trig=="sin":
                phi=np.sin(omega*x); dphi=omega*np.cos(omega*x)
            else:
                phi=np.cos(omega*x); dphi=-omega*np.sin(omega*x)
            temporal=dx*np.sum(du*phi,axis=-1)
            flux_term=dt*dx*np.sum(f_mid*dphi,axis=-1)
            residual=temporal-flux_term
            scale=dx*np.sum(np.abs(du),axis=-1)+dt*dx*np.sum(np.abs(f_mid*dphi),axis=-1)+1e-12
            scores.append(np.abs(residual)/scale)
    return np.max(np.stack(scores,axis=0),axis=0)
