"""Train a conservative local neural operator for eight shallow-water steps.

A shared pointwise ReLU encoder on state jumps feeds a spatial convolution
with 15 offsets. The readout predicts time-averaged fluxes and one conservative
difference gives the endpoint. All trainable arrays are exported to the native
predictor. Training, replay selection and development use whole trajectories.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from .macro8_data import G, LAMBDA, STEPS, ReferenceSolver, make_split

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "shallow_water_macro8"
HEADER = ROOT / "shallow_water" / "macro8_weights.h"
C = np.sqrt(G)
COURANT = C*LAMBDA*STEPS
RADIUS = STEPS - 1
TAPS = 2*RADIUS+1
WIDTH = 8
INPUTS = 6
CHANNELS = 2*(WIDTH+1)
SCALES = np.array([1., C])
# Keep these physical engineering tolerances fixed before drawing the test data.
TRAJECTORY_TOL = np.array([0.01, 0.04])
ROWS_PER_STATE = 48
SEED = 216071


def encode_state(q):
    """Jump inputs on n+1 faces, normalized by h0=1m and c0=sqrt(g*h0)."""
    h, m = (q/SCALES).T
    u = m/h
    physical = np.stack((m, m*u+0.5*h*h), axis=-1)
    central = np.empty((len(q)+1, 2))
    central[[0,-1]] = physical[[0,-1]]
    central[1:-1] = 0.5*(physical[1:]+physical[:-1])
    dh, dm, du = np.diff(h), np.diff(m), np.diff(u)
    x = np.zeros((len(q)+1, INPUTS))
    x[1:-1] = np.stack((np.sqrt((h[:-1]+h[1:])*.5)-1, (u[:-1]+u[1:])*.5,
                        dh, du, np.abs(dh), np.abs(du)), axis=-1)
    jump = np.zeros((len(q)+1,2))
    jump[1:-1] = np.stack((dh,dm),axis=-1)
    return central, x, jump


def embedding(x, jump, w, b):
    hidden = np.maximum(x@w+b, 0.)
    raw = np.concatenate((jump, (jump[..., :, None]*hidden[..., None, :]).reshape(
        *jump.shape[:-1], 2*WIDTH)), axis=-1)
    return hidden, raw


def predict(q, params):
    """Vectorized independent implementation of the native predictor."""
    w, b, kernel = params
    central, x, jump = encode_state(q)
    _, raw = embedding(x, jump, w,b)
    padded = np.pad(raw, ((RADIUS,RADIUS),(0,0)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, TAPS, axis=0).transpose(0,2,1)
    flux = central + np.einsum("ntc,tco->no", windows, kernel, optimize=True)
    flux[[0,-1]] = central[[0,-1]]
    return q+COURANT*(flux[:-1]-flux[1:])*SCALES


def patches(batch, ids, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for i in ids:
        q, target = batch.q[i], batch.target[i]
        central,x,jump = encode_state(q)
        # Integrate q^8 - q^0 to obtain unique training fluxes with the
        # undisturbed transmissive left boundary. Training buffers are >8.
        flux = np.empty_like(central)
        flux[0] = central[0]
        flux[1:] = central[0] + np.cumsum((q-target)/SCALES/COURANT, axis=0)
        resid = flux-central
        x = np.pad(x, ((RADIUS,RADIUS),(0,0)))
        jump = np.pad(jump, ((RADIUS,RADIUS),(0,0)))
        patch_j = np.lib.stride_tricks.sliding_window_view(jump,TAPS+1,axis=0).transpose(0,2,1)
        amp = np.max(np.abs(patch_j),axis=(1,2))
        active = np.flatnonzero(amp>1.e-12)
        if not active.size:
            continue
        idx = rng.choice(active, ROWS_PER_STATE, replace=len(active)<ROWS_PER_STATE)
        xi = x[idx[:,None] + np.arange(TAPS+1)]
        # Cell differences are the supervised endpoint signal. Face labels
        # constrain its long-wavelength component and share the convolution.
        rows.append((xi,patch_j[idx],resid[idx],resid[idx]-resid[idx+1],
                     1/np.sqrt(.08+amp[idx])))
    return tuple(np.concatenate(z) for z in zip(*rows))


def init():
    rng = np.random.default_rng(SEED)
    w = rng.normal(0,.55,(INPUTS,WIDTH))
    b = rng.uniform(-.08,.30,WIDTH)
    b[:3] = (.4,.4,.4)
    return [w,b,np.zeros((TAPS,CHANNELS,2))]


def design(raw):
    a = raw[:,:-1]
    return a,a-raw[:,1:]


def readout(params, data):
    w,b,_ = params
    x,j,flux,change,weights = data
    width = TAPS*CHANNELS
    gram, rhs = np.zeros((width,width)), np.zeros((width,2))
    for k in range(0,len(x),512):
        s = slice(k,k+512)
        _,raw = embedding(x[s],j[s],w,b)
        for mat, label, factor in zip(design(raw),(flux[s],change[s]),(.3,1.)):
            weight = weights[s,None]*factor
            a = mat.reshape(len(mat),width)*weight
            gram += a.T@a
            rhs += a.T@(label*weight)
    # Tikhonov regularization stabilizes correlated filters on smooth states.
    gram.flat[::width+1] += 2e-5*len(x)/10000
    return np.linalg.solve(gram,rhs).reshape(TAPS,CHANNELS,2)


def fit(params, data, seed, cycles):
    """Alternate four/fixed-count Adam epochs on encoder and exact ridge readouts."""
    w,b,kernel = [p.copy() for p in params]
    x,j,flux,change,weights = data
    rng = np.random.default_rng(seed)
    m = [np.zeros_like(w),np.zeros_like(b)]
    v = [np.zeros_like(w),np.zeros_like(b)]
    step = 0
    for _ in range(cycles):
        kernel = readout((w,b,kernel),data)
        for _ in range(36):
            step += 1
            idx = rng.integers(len(x),size=128)
            hidden,raw = embedding(x[idx],j[idx],w,b)
            a,d = design(raw)
            flat = kernel.reshape(-1,2)
            wd = weights[idx,None]**2 / len(idx)
            e0 = ((a.reshape(len(idx),-1)@flat-flux[idx])*.09)*wd
            e1 = ((d.reshape(len(idx),-1)@flat-change[idx]))*wd
            dr = np.zeros_like(raw)
            dr[:,:-1] = ((e0+e1)@flat.T).reshape(a.shape)
            dr[:,1:] -= (e1@flat.T).reshape(a.shape)
            dj = dr[:,:,2:].reshape(len(idx),TAPS+1,2,WIDTH)
            dh = np.sum(dj*j[idx,:,:,None],axis=2)*(hidden>0)
            dx = x[idx].reshape(-1,INPUTS)
            dz = dh.reshape(-1,WIDTH)
            for i,(param,grad) in enumerate(((w,dx.T@dz),(b,dz.sum(axis=0)))):
                m[i] = .9*m[i]+.1*grad
                v[i] = .999*v[i]+.001*grad**2
                param -= .0015*(m[i]/(1-.9**step))/(np.sqrt(v[i]/(1-.999**step))+1e-10)
    return w,b,readout((w,b,kernel),data)


def errors(batch, params):
    def one(a,b):
        pred = predict(a,params)
        return np.mean(np.abs(pred-b),axis=0)
    return np.array([one(a,b) for a,b in zip(batch.q,batch.target)])


def summary(e):
    ratio = np.max(e/TRAJECTORY_TOL,axis=1)
    return dict(count=len(e), mean_mae=e.mean(axis=0).tolist(),
                fail_count=int(np.sum(ratio>1)), worst_ratio=float(ratio.max()))


def export(models, out, header):
    out.mkdir(parents=True,exist_ok=True)
    w,b,k = (np.stack(z) for z in zip(*models))
    np.savez(out/"weights.npz",encoder=w,bias=b,kernel=k)
    def literal(a):
        if not a.ndim: return float(a).hex()
        return "{"+",\n".join(literal(row) for row in a)+"}"
    with header.open('w') as f:
        f.write("/* Frozen macro-operator parameters, regenerated by macro8_model.py. */\n")
        for name,array in (("sw8_w",w),("sw8_b",b),("sw8_k",k.transpose(0,2,1,3))):
            dims = ''.join(f'[{v}]' for v in array.shape)
            f.write(f"static const double {name}{dims} = {literal(array)};\n")


def main(out=OUT, header=HEADER):
    out.mkdir(parents=True,exist_ok=True)
    with ReferenceSolver() as solver:
        train = make_split("train",256,solver)
        pool = make_split("mine",256,solver)
        dev = make_split("dev",96,solver)
    base = patches(train, np.arange(len(train.q)), SEED+1)
    m0 = fit(init(),base,SEED+2,4)
    err = errors(pool,m0)
    ratio = (err/TRAJECTORY_TOL).max(axis=1)
    rng = np.random.default_rng(SEED+3)
    add = 64
    uniform = rng.choice(len(pool.q), add, replace=False)
    hard = np.argsort(-ratio,kind="stable")[:add//2]
    hard = np.r_[hard,rng.choice(np.setdiff1d(np.arange(len(pool.q)), hard),add-len(hard),replace=False)]
    models=[m0]
    selected = {}
    for name,indices in (("uniform",uniform),("hard",hard)):
        extra = patches(pool,indices,SEED+4)
        both = tuple(np.concatenate((a,b)) for a,b in zip(base,extra))
        models.append(fit(m0,both,SEED+5,3))
        selected[name] = dict(trajectory_ids=[pool.ids[i] for i in indices],
                              true_failures=int(np.sum(ratio[indices]>1)),
                              mean_rank_score=float(np.mean(ratio[indices])))
    names = ("initial","uniform","hard")
    desc = dict(training_seed=SEED, n_train=len(train.q),n_mine=len(pool.q),n_dev=len(dev.q),
                extra_states=add, rows_per_state=ROWS_PER_STATE, width=WIDTH,
                inputs=["sqrt(mean(h))-1","mean(u)/sqrt(g)","delta h","delta u/sqrt(g)",
                        "abs(delta h)","abs(delta u)/sqrt(g)"], taps=TAPS,
                learned_parameters=int(sum(a.size for a in models[0])), native_modes=names,
                initial_encoder_steps=4*36, continuation_encoder_steps=3*36,
                initial_ridge_fits=5, continuation_ridge_fits=4,
                error_thresholds=TRAJECTORY_TOL.tolist(), initial_pool=summary(err),
                replay_definition="Select half by decreasing maximum normalized block endpoint MAE, then draw half at random from remaining trajectories.",
                selected=selected, dev={n:summary(errors(dev,m)) for n,m in zip(names,models)})
    export(models,out,header)
    (out/"training.json").write_text(json.dumps(desc,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:desc[k] for k in ("initial_pool","dev")},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=OUT)
    parser.add_argument("--header",type=Path,default=HEADER)
    args=parser.parse_args()
    main(args.out,args.header)
