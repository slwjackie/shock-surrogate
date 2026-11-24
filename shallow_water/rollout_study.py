"""Matched-time first-order wet shallow-water rollouts with exact cell-average targets."""
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter_ns

import numpy as np

from shallow_water.benchmark import FLAGS, METHODS
from shallow_water.data import G, reference

METHOD_NAMES = (*METHODS, "hlle", "rusanov")
# left depth, velocity, right depth, velocity. The discontinuity is at x=1/2.
CASES = {"wet_dam": (4., 0., 1., 0.), "colliding": (1.5, 1.3, 1.5, -1.3),
         "separating": (1.2, -1.3, 1.2, 1.3)}
DURATION = .045
LENGTH = 1.
CFL = .70
ROOT_TOLERANCE = .01  # m
CONTROL_DEPTH_TOLERANCE = 1e-8
ERROR_FACTOR = 1.05


def initial(case, n):
    hl, ul, hr, ur = case
    if n % 2:
        raise ValueError("Use even grids with a face at the initial jump")
    q = np.empty((n, 2))
    q[:n//2] = hl, hl*ul
    q[n//2:] = hr, hr*ur
    return q


def wave_term(h, hi):
    if h > hi:
        return (h-hi)*math.sqrt(.5*G*(1/h+1/hi))
    return 2*(math.sqrt(G*h)-math.sqrt(G*hi))


def physical_flux(h, u):
    return np.array([h*u, h*u*u + .5*G*h*h])


def exact_cell_averages(case, n, duration=DURATION, jump_cell=None):
    """Piecewise exactly integrate the constant/fan/shock entropy solution in FP64.

    In a fan h(x/t) is quadratic and hu(x/t) cubic, so two-point Gauss
    integration is exact after splitting each coarse cell at each wave edge.
    """
    if jump_cell is None:
        if n % 2:
            raise ValueError("A centred discontinuity requires an even grid")
        jump_cell = n//2
    if not 0 < jump_cell < n or not duration > 0:
        raise ValueError("The discontinuity must be internal and the duration positive")
    jump_x = LENGTH * jump_cell/n
    hl, ul, hr, ur = case
    hs = float(reference(np.array([case]))[0])
    us = .5*(ul+ur + wave_term(hs,hr)-wave_term(hs,hl))
    cl, cr, cs = (math.sqrt(G*h) for h in (hl,hr,hs))
    if hs > hl:
        la = lb = ul-math.sqrt(.5*G*hs*(hs+hl)/hl)
    else:
        la, lb = ul-cl, us-cs
    if hs > hr:
        ra = rb = ur+math.sqrt(.5*G*hs*(hs+hr)/hr)
    else:
        ra, rb = us+cs, ur+cr
    max_speed = max(abs(ul)+cl, abs(ur)+cr, abs(us)+cs, abs(la), abs(rb))
    if min(jump_x, LENGTH-jump_x) <= max_speed*duration:
        raise ValueError("The analytical waves may reach a boundary")
    region = np.array([la, lb, ra, rb])
    assert np.all(np.diff(region) >= 0)

    def evaluate(z):
        if z < la:
            return hl, hl*ul
        if z < lb:
            k = ul + 2*cl
            h, u = (k-z)**2/(9*G), (k+2*z)/3
            return h, h*u
        if z < ra:
            return hs, hs*us
        if z < rb:
            k = ur - 2*cr
            h, u = (z-k)**2/(9*G), (k+2*z)/3
            return h, h*u
        return hr, hr*ur

    edges = (np.linspace(0, LENGTH, n+1)-jump_x)/duration
    ref = np.zeros((n, 2))
    for i, (left,right) in enumerate(zip(edges[:-1], edges[1:])):
        pieces = np.unique(np.r_[left, region[(left < region) & (region < right)], right])
        for a,b in zip(pieces[:-1],pieces[1:]):
            mid, offset = (a+b)/2, (b-a)/math.sqrt(12)
            ref[i] += ((b-a)/(2*(right-left)) *
                       (np.array(evaluate(mid-offset)) + np.array(evaluate(mid+offset))))
    mean_initial = (jump_cell*np.array([hl, hl*ul]) + (n-jump_cell)*np.array([hr, hr*ur]))/n
    balance = mean_initial + duration/LENGTH*(physical_flux(hl,ul)-physical_flux(hr,ur))
    np.testing.assert_allclose(ref.mean(axis=0), balance, rtol=5e-13, atol=5e-13)
    return ref, math.ceil(duration*n*max_speed/(LENGTH*CFL))


def compile_rollout(path):
    import shallow_water
    package = Path(shallow_water.__file__).resolve().parent
    source = Path(__file__).resolve().with_name("rollout.c")
    binary = path / "rollout.so"
    subprocess.run(["gcc", *FLAGS, str(source), str(package/"solver.c"),
                    "-I", str(package), "-lm", "-o", str(binary)], check=True)
    lib = ctypes.CDLL(str(binary))
    run = lib.sw_rollout
    array = np.ctypeslib.ndpointer(np.float64, ndim=2, flags="C_CONTIGUOUS")
    run.argtypes = [ctypes.c_int, array, ctypes.c_double, ctypes.c_double, ctypes.c_int,
                    ctypes.c_double, ctypes.c_int]
    run.restype = ctypes.c_int
    return run


def provenance():
    package = Path(__file__).resolve().parent
    return dict(
        compiler=subprocess.check_output(["gcc", "--version"], text=True).splitlines()[0],
        flags=FLAGS,
        sha256={name: hashlib.sha256((package/name).read_bytes()).hexdigest()
                for name in ("rollout.c", "solver.c", "model.h", "certificate.h")},
        grids=[128, 256, 512, 1024],
        accuracy_target=("Per-case componentwise terminal mean absolute errors of the "
                         "filtered Newton N=256 control with 1e-8 m depth tolerance, "
                         "multiplied by 1.05"),
        timing_statistic=("Median across seven shuffled single-trajectory calls by default; "
                          "p25/p75 are within-run interquartiles. C allocation, "
                          "interface work, cell updates and the Python-to-C call are "
                          "included. Python input reset and reference are untimed."))


def study(out, cpu, rounds):
    if rounds<1:
        raise ValueError("Positive timing sample count required")
    if hasattr(os, "sched_getaffinity"):
        cores = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {cores[0] if cpu is None else cpu})
    metadata = provenance()
    ns = metadata["grids"]
    rng = np.random.default_rng(20681)
    thresholds, rows = {}, []
    with TemporaryDirectory() as tmp:
        advance = compile_rollout(Path(tmp))
        for name, params in CASES.items():
            tasks = []
            for n in ns:
                q0 = initial(params,n)
                exact, steps = exact_cell_averages(params,n)
                if n==256:
                    control = q0.copy()
                    ret = advance(n, control, LENGTH, DURATION, steps, CONTROL_DEPTH_TOLERANCE, 6)
                    if ret:
                        raise RuntimeError(f"Failed to establish {name} accuracy target at {n}: {ret}")
                    err = np.abs(control-exact).mean(axis=0)
                    thresholds[name] = (ERROR_FACTOR*err).tolist()
                for mode, label in enumerate(METHOD_NAMES):
                    q = q0.copy()
                    code = advance(n, q, LENGTH, DURATION, steps, ROOT_TOLERANCE, mode)
                    if code or not np.isfinite(q).all() or np.min(q[:,0]) <= 0:
                        raise RuntimeError(f"{name} {label} {n} produced invalid rollout ({code})")
                    # Discrete mass & momentum balance at fixed outflow boundaries.
                    # Numerical viscosity can make the outflow differ from its
                    # initial value before the physical wave reaches it.
                    row = dict(case=name, n=n, steps=steps, method=label,
                               err_h=float(np.abs(q[:,0]-exact[:,0]).mean()),
                               err_hu=float(np.abs(q[:,1]-exact[:,1]).mean()),
                               mean_balance_residual=(q.mean(axis=0)-exact.mean(axis=0)).tolist(),
                               min_depth=float(q[:,0].min()), samples_us=[])
                    tasks.append((q0, n, steps, mode, row))
                    rows.append(row)
            for _ in range(rounds):
                for idx in rng.permutation(len(tasks)):
                    q0, n, steps, mode, row = tasks[idx]
                    q = q0.copy()
                    start = perf_counter_ns()
                    code = advance(n, q, LENGTH, DURATION, steps, ROOT_TOLERANCE, mode)
                    elapsed = (perf_counter_ns()-start)/1000
                    if code:
                        raise RuntimeError(f"Timed rollout failed ({code})")
                    row["samples_us"].append(elapsed)
    for row in rows:
        target = thresholds[row["case"]]
        row["median_us"] = float(np.median(row["samples_us"]))
        row["iqr_us"] = np.quantile(row["samples_us"], [.25, .75]).tolist()
        row["meets_both"] = row["err_h"]<=target[0] and row["err_hu"]<=target[1]
    matched = {}
    for name in CASES:
        matched[name] = {}
        for method in METHOD_NAMES:
            eligible = [r for r in rows if r["case"]==name and r["method"]==method and r["meets_both"]]
            pick = min(eligible, key=lambda r:r["median_us"], default=None)
            matched[name][method] = {k:pick[k] for k in ("n","median_us","err_h","err_hu")} if pick else None
    obj = dict(provenance=metadata, description="State-start to terminal state, single trajectory, warmed shuffled timings",
               cases=CASES, domain_length=LENGTH, T=DURATION, cfl_target=CFL,
               initialization="Exact two constant states on either side of x=0.5",
               cfl_method="Same domain/grid/case-dependent step count for each numerical flux",
               outflow="Transmissive on both ends; analytic waves do not reach the boundary",
               depth_tolerance_m=ROOT_TOLERANCE, control_depth_tolerance_m=CONTROL_DEPTH_TOLERANCE,
               error_factor=ERROR_FACTOR, thresholds=thresholds, rounds=rounds,
               platform=platform.platform(), affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None,
               rows=rows, fastest_meeting_both=matched)
    out.write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n")
    return obj


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "reports" / "shallow_water" / "rollout.json")
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--rounds", type=int, default=7)
    args = parser.parse_args()
    result = study(args.out, args.cpu, args.rounds)
    print(json.dumps(result["fastest_meeting_both"], indent=2))
