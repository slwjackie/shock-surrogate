"""Train and export inexpensive dimensionless shallow-water star-depth proposals.

The network corrects the classical two-rarefaction estimate. Failures affect
offline sampling only; the solver's signed depth check controls acceptance.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "reports/shallow_water"
DEFAULT_HEADER = ROOT / "shallow_water/model.h"
GRAVITY = 9.81
WIDTH = 12
TOLERANCE = 0.01
NAMES = ("initial", "uniform", "bootstrap")


def features(states, gravity=GRAVITY):
    hl, ul, hr, ur = np.asarray(states, dtype=np.float64).T
    big = np.maximum(hl, hr)
    low = np.sqrt(gravity * np.minimum(hl, hr))
    high = np.sqrt(gravity * big)
    r = low / high
    w = (0.5 * (low + high) + 0.25 * (ul - ur)) / high
    # The caller handles dry/vacuum states. A zero w is safe in this offline
    # helper; only wet, positive-w states are used in the experiment.
    wp = np.where(w > 0.0, w, 1.0)
    q = np.maximum(0.0, (w - r) / wp)
    p = np.maximum(0.0, (w - 1.0) / wp)
    ratio = p / np.where(q > 0.0, q, 1.0)
    x = np.stack((q, p, ratio), axis=1)
    scale = q*q*q + p*p*p
    return x, scale, big*w*w


def propose(states, params):
    x, scale, base = features(states)
    weights, bias, output, offset = params
    correction = np.maximum(x @ weights + bias, 0.0) @ output + offset.item()
    return base * np.maximum(0.0, 1.0 - scale * correction)


def training_arrays(states, depth):
    x, scale, base = features(states)
    # Tiny waves yield cancellation in the target correction, but their
    # absolute difference from the analytic proposal is already negligible.
    keep = (scale > 1e-6) & (base > 0.0)
    y = (1.0 - depth[keep] / base[keep]) / scale[keep]
    return x[keep], y, (scale * base)[keep]


def initialize(seed):
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.7, (3, WIDTH))
    centers = rng.uniform(0.03, 0.94, (3, WIDTH))
    b = -(centers*w).sum(axis=0)
    return [w, b, rng.normal(0.0, 0.03, WIDTH), np.array(0.4)]


def fit(params, x, y, importance, steps, seed, lr):
    """Mini-batch Adam. Both continuation arms reset moments and use one seed."""
    params = [np.copy(p) for p in params]
    first = [np.zeros_like(p) for p in params]
    second = [np.zeros_like(p) for p in params]
    rng = np.random.default_rng(seed)
    # The target is in correction units. Upweight examples where a unit
    # correction error would more strongly affect depth, without making a few
    # extreme high-depth cases dominate every mini-batch.
    iw = np.sqrt(np.clip(importance, 0.01, 50.0))
    iw /= np.mean(iw)
    for i in range(1, steps+1):
        j = rng.integers(len(x), size=384)
        xb = x[j]
        weights, bias, output, offset = params
        hidden = np.maximum(xb @ weights + bias, 0.0)
        dy = iw[j] * (hidden @ output + offset.item() - y[j]) / len(j)
        dh = (dy[:, None] * output) * (hidden > 0.0)
        grad = (xb.T @ dh, dh.sum(axis=0), hidden.T @ dy, dy.sum())
        rate = lr*(0.1+0.9*(1-i/steps))
        for k, (p, d) in enumerate(zip(params, grad)):
            first[k] = .9*first[k] + .1*d
            second[k] = .999*second[k] + .001*d*d
            p -= rate*(first[k]/(1-.9**i))/(np.sqrt(second[k]/(1-.999**i))+1e-8)
    return params


def save_weights(path, models):
    np.savez(path, **{key: np.stack([m[i] for m in models])
                     for i, key in enumerate(("weights", "bias", "output", "offset"))})


def load_weights(path=DEFAULT_OUT/"models.npz"):
    with np.load(path) as data:
        return [[data[k][i] for k in ("weights", "bias", "output", "offset")] for i in range(3)]


def literal(x):
    if np.ndim(x) == 0:
        return float(x).hex()
    return "{" + ",".join(literal(i) for i in x) + "}"


def write_header(path, models):
    declarations = []
    for i, (name, sizes) in enumerate((("w", f"[3][3][{WIDTH}]"), ("b", f"[3][{WIDTH}]"),
                                        ("v", f"[3][{WIDTH}]"), ("c", "[3]"))):
        a = np.stack([m[i] for m in models])
        declarations.append(f"static const double sw_nn_{name}{sizes} = {literal(a)};")
    header = """/* Generated from frozen training snapshots by python -m shallow_water.model. */
#ifndef SW_DEPTH_MODEL_H
#define SW_DEPTH_MODEL_H
#include <math.h>
"""
    header += "\n".join(declarations)
    header += """
static inline double sw_model_guess(int model, double hmax, double r, double w)
{
    const double base = hmax*w*w;
    if (!(w > r)) return base;
    const double q = (w-r)/w;
    const double p = w > 1.0 ? (w-1.0)/w : 0.0;
    const double t = p/q;
    double correction = sw_nn_c[model];
    for (int i=0; i<SW_NET_WIDTH; ++i) {
        const double activation = sw_nn_b[model][i] + q*sw_nn_w[model][0][i]
                              + p*sw_nn_w[model][1][i] + t*sw_nn_w[model][2][i];
        correction += (activation > 0.0 ? activation : 0.0)*sw_nn_v[model][i];
    }
    const double factor = 1.0 - (q*q*q+p*p*p)*correction;
    return base*(factor > 0.0 ? factor : 0.0);
}
#endif
"""
    path.write_text(header.replace("SW_NET_WIDTH",str(WIDTH)))


def challenge_sample(n, seed):
    """Independent wet states for offline mining, with a broader depth/speed range."""
    rng = np.random.default_rng(seed)
    hl, hr = np.exp(rng.uniform(np.log(.15), np.log(9.0), (2, n)))
    cl, cr = np.sqrt(GRAVITY*hl), np.sqrt(GRAVITY*hr)
    du = rng.uniform(-1.85,1.8,n)*(cl+cr)
    advection = .5*rng.uniform(-1,1,n)*(cl+cr)
    return np.column_stack((hl,advection-.5*du,hr,advection+.5*du))


def mining_rejections(states, params):
    """Evaluate the frozen first-pass C proposal with the actual certificate."""
    shim = r"""
#include "model.h"
#include "certificate.h"
void check(int n, const double *v, unsigned char *accepted, double tol, double g) {
    for (int i=0; i<n; ++i) {
        double hl=v[4*i], ul=v[4*i+1], hr=v[4*i+2], ur=v[4*i+3];
        double hmax=fmax(hl,hr), cl=sqrt(g*hl), cr=sqrt(g*hr);
        double cmax=fmax(cl,cr), r=fmin(cl,cr)/cmax;
        double w=(0.5*(cl+cr)+0.25*(ul-ur))/cmax;
        double proposal=sw_model_guess(0,hmax,r,w);
        accepted[i]=(unsigned char)sw_certify(hl,ul,hr,ur,g,proposal,tol);
    }
}
"""
    with TemporaryDirectory() as directory:
        tmp = Path(directory)
        write_header(tmp/"model.h",[params]*3)
        (tmp/"check.c").write_text(shim)
        subprocess.run(["cc","-std=c11","-O3","-fno-fast-math","-ffp-contract=off","-shared","-fPIC",
                        "-I",str(ROOT/"shallow_water"),str(tmp/"check.c"),"-lm","-o",str(tmp/"check.so")],check=True)
        lib = ctypes.CDLL(str(tmp/"check.so"))
        check = lib.check
        check.argtypes = [ctypes.c_int,np.ctypeslib.ndpointer(dtype=np.float64,flags="C_CONTIGUOUS"),
                          np.ctypeslib.ndpointer(dtype=np.uint8,flags="C_CONTIGUOUS"),
                          ctypes.c_double,ctypes.c_double]
        check.restype = None
        accepted = np.zeros(len(states),dtype=np.uint8)
        check(len(states),np.ascontiguousarray(states),accepted,TOLERANCE,GRAVITY)
    return accepted == 0


def train(out=DEFAULT_OUT, header=DEFAULT_HEADER):
    from .data import sample, reference

    base = sample(8192, 1391)
    pool = np.concatenate((sample(16384,2791),challenge_sample(16384,2792)))
    valid = sample(8192, 7079)
    y, yp, yv = [reference(a) for a in (base, pool, valid)]
    init = fit(initialize(9041), *training_arrays(base, y), 3500, 1396, .004)
    pool_err = np.abs(propose(pool, init)-yp)
    rejected = mining_rejections(pool,init)
    failures = np.flatnonzero(rejected)
    rng = np.random.default_rng(7743)
    count = 4096
    uniform_ids = rng.choice(len(pool), count, replace=False)
    # Use a fixed hard-example quota. Tolerance misses rank first, followed
    # by samples that crossed closest to the same acceptance boundary.
    # Fill the other half uniformly; every new example is unique.
    mined_ids = np.lexsort((-pool_err,~rejected))[:count//2]
    rest = rng.choice(np.setdiff1d(np.arange(len(pool)),mined_ids), count-len(mined_ids), replace=False)
    hard_ids = np.r_[mined_ids, rest]
    models = [init]
    for ids in (uniform_ids, hard_ids):
        states = np.concatenate((base, pool[ids]))
        depth = np.concatenate((y, yp[ids]))
        models.append(fit(init, *training_arrays(states,depth), 3000, 1397, .0018))
    stats = {}
    for name, params in zip(NAMES, models):
        err = np.abs(propose(valid,params)-yv)
        stats[name] = {"mean_abs_m":float(err.mean()),"p95_abs_m":float(np.quantile(err,.95)),
                       "fraction_within_tol":float(np.mean(err<=TOLERANCE))}
    metadata = dict(gravity=GRAVITY,tolerance_m=TOLERANCE,
                    architecture="3 dimensionless inputs, 12 ReLU hidden units, 1 output; multiplicative shock correction to two-rarefaction depth",
                    training_set=8192,mining_pool=32768,train_seed=1391,iid_pool_seed=2791,challenge_pool_seed=2792,
                    validation_seed=7079,common_pool_iid_rows=16384,common_pool_challenge_rows=16384,
                    challenge_distribution="independent depths log-uniform [0.15,9] m, velocity jump/(cL+cR) uniform [-1.85,1.8]; finite wet states",
                    mining_certificate="shallow_water/certificate.h sw_certify with the first-pass native proposal",
                    initial_steps=3500,continuation_steps=3000,added_rows_each=count,
                    mining_rule="half the new rows prioritize first-model certificate rejections, then largest offline absolute depth errors; fill remainder uniformly without replacement.",
                    pool_failures=len(failures),pool_oracle_misses=int(np.sum(pool_err>TOLERANCE)),
                    pool_iid_failures=int(rejected[:16384].sum()),pool_challenge_failures=int(rejected[16384:].sum()),
                    mined_quota_rows=len(mined_ids),mined_quota_actual_rejections=int(rejected[mined_ids].sum()),
                    mined_quota_near_misses=int((~rejected[mined_ids]).sum()),
                    selected_failures_uniform=int(rejected[uniform_ids].sum()),
                    selected_failures_bootstrap=int(rejected[hard_ids].sum()),
                    validation=stats)
    out.mkdir(parents=True,exist_ok=True)
    save_weights(out/"models.npz",models)
    write_header(header,models)
    (out/"training.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps(metadata,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--header", type=Path, default=DEFAULT_HEADER)
    args = parser.parse_args()
    train(args.out,args.header)
