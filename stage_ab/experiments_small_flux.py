"""Paired CPU comparison of narrower frozen ReLU fluxes on the saved A2 cases."""
import json
import os
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import numpy as np

from certified_burgers.godunov import advance, godunov_flux
from .anchor_refinement import AnchorTable
from .benchmark_flux_routing import CASES, MODEL, THREAD_VARS, algebraic_godunov, rollout, sample_times
from .cheap_gate import gate_rollout, train_gate
from .experiments_a2 import discretize
from .experiments_refinement import make_cases
from .frozen_flux import FrozenFlux
from .stage_a2 import CertificateTables, CertifiedFlux, sample_training_pairs


def fit(width, steps=3500):
    init = np.random.default_rng(317)
    data = np.random.default_rng(811)
    dims = [2, width, width, 1]
    params = []
    for a, b in zip(dims, dims[1:]):
        params.extend([init.normal(0, (2/a)**.5, (a, b)), init.uniform(-.1, .1, b)])
    first = [np.zeros_like(p) for p in params]
    second = [np.zeros_like(p) for p in params]
    w1, b1, w2, b2, w3, b3 = params
    for it in range(1, steps+1):
        x = sample_training_pairs(data, 512, 2.)
        y = godunov_flux(*x.T)[:, None]
        h = np.maximum(0., x @ w1+b1)
        j = np.maximum(0., h @ w2+b2)
        r = (j @ w3+b3-y) * (2/len(x))
        dj = (r @ w3.T)*(j > 0)
        dh = (dj @ w2.T)*(h > 0)
        grads = [x.T@dh, dh.sum(0), h.T@dj, dj.sum(0), j.T@r, r.sum(0)]
        rate = .003*(.1+.9*(1+np.cos(np.pi*it/steps))/2)
        for p, m, v, g in zip(params, first, second, grads):
            m *= .9
            m += .1*g
            v *= .999
            v += .001*g*g
            p -= rate*(m/(1-.9**it))/(np.sqrt(v/(1-.999**it))+1e-8)
    return FrozenFlux([(w.T,b) for w,b in zip(params[::2],params[1::2])])


def run(out):
    if any(os.environ.get(k) != "1" for k in THREAD_VARS):
        raise ValueError("Use one numerical thread")
    out.mkdir(parents=True, exist_ok=True)
    models = {"original16": FrozenFlux.load(MODEL)}
    for width in (16, 8, 4, 2):
        model = fit(width)
        model.save(out/f"width{width}.json")
        models[f"width{width}"] = model
    axis = np.linspace(-2,2,301)
    xy = np.array(np.meshgrid(axis,axis)).reshape(2,-1).T
    target = godunov_flux(*xy.T)
    records = json.loads(CASES.read_text())["results"]
    cases = [(discretize(r["ic"],128), r["budget"]) for r in records if r["method"]=="selective_anchor/all/128"]
    assert len(cases)==9
    original = CertifiedFlux(models["original16"])
    training = [(v,b) for _, v,b in make_cases(original,CertificateTables.build(original),1100,12)]
    results = {}
    gates = {}
    for name,model in models.items():
        m = CertifiedFlux(model)
        base = CertificateTables.build(m)
        anchor = AnchorTable.build(m,base,bins=128)
        _, gate = train_gate(m,base,anchor,training,steps=60,lam=.2)
        gates[name] = m,base,anchor,gate
        err = np.abs(model.predict(xy)-target)
        results[name] = dict(grid_flux_mae=float(np.mean(err)), grid_flux_max=float(np.max(err)))
    rows = []
    rng = np.random.default_rng(184)
    for i,(state,budget) in enumerate(cases):
        methods = {"public":lambda v: advance(v,1.,60,dt=.2)[0],
                   "formula":lambda v: rollout(v,60,.2,None,algebraic_godunov,0)}
        for name,model in models.items():
            methods[name] = lambda v,m=model: rollout(v,60,.2,m,None,len(v))
        warm = {k:f(state) for k,f in methods.items()}
        assert np.array_equal(warm["public"],warm["formula"])
        times = sample_times(state,methods,warm,rng,9)
        order = rng.permutation(list(gates))
        for name in order:
            kwargs = dict(steps=60,lam=.2,total_budget=budget)
            g = gates[name]
            ref = gate_rollout(*g,state,**kwargs,audit=True)
            assert ref["audited_steps"]==60 and ref["spent"]<=budget
            gt = []
            for _ in range(3):
                start = perf_counter()
                r = gate_rollout(*g,state,**kwargs)
                gt.append(perf_counter()-start)
                assert np.array_equal(r["state"],ref["state"])
            rows.append(dict(index=i, name=name, accepted=ref["accepted"], budget=budget, spent=ref["spent"],
                             gate_s=median(gt), nn_s=median(times[name]),
                             terminal_l1=float(np.abs(warm[name]-warm["public"]).mean()),
                             public_s=median(times["public"]), formula_s=median(times["formula"])))
    for name,summary in results.items():
        group = [r for r in rows if r["name"]==name]
        for key in ("gate_s","nn_s","terminal_l1","public_s","formula_s"):
            summary[key]=mean(r[key] for r in group)
        summary["coverage"]=mean(r["accepted"]/(128*60) for r in group)
    (out/"results.json").write_text(json.dumps(dict(results=results,rows=rows,numpy=np.__version__),indent=2)+"\n")
    print(json.dumps(results,indent=2))


if __name__=="__main__":
    run(Path("reports/small_flux"))
