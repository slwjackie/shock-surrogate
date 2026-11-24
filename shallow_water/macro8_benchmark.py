"""Benchmark a learned eight-step update against native wet shallow-water solvers."""
import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter_ns

import numpy as np

from .benchmark import FLAGS
from .data import reference
from .macro8_data import (DURATION, FAMILIES, G, LAMBDA, N, SPLIT_SEEDS, STEPS,
                          TOL, ReferenceSolver, analytical_dam, make_split)

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'reports' / 'shallow_water_macro8'
METHODS = ('teacher8', 'godunov8', 'godunov_cfl', 'engineering8', 'engineering_cfl',
           'hlle8', 'hlle_cfl', 'rusanov8', 'rusanov_cfl', 'initial', 'uniform',
           'hard', 'routed_initial', 'routed_uniform', 'routed_hard')
MODEL_NAMES = ('initial', 'uniform', 'hard')
ENDPOINT_BUDGET = np.array([0.01, 0.04])
FINE_ERROR_FACTOR = 1.05
WARMUP = 2


def build(directory):
    binary = Path(directory) / 'macro8.so'
    sources = ('macro8_workload.c', 'macro8_predict.c', 'macro8_guard.c',
               'macro8_classical.c', 'rollout.c')
    subprocess.run(['gcc', *FLAGS, *(str(HERE / s) for s in sources),
                    '-lm', '-o', str(binary)], check=True)
    lib = ctypes.CDLL(str(binary.resolve()))
    real, integer = ctypes.c_double, ctypes.c_int
    def matrix(dtype, ndim):
        return np.ctypeslib.ndpointer(dtype=dtype, ndim=ndim, flags='C_CONTIGUOUS')
    grid, batch = matrix(np.float64,2), matrix(np.float64,3)
    risk, record, records = matrix(np.float64,1), matrix(np.int32,1), matrix(np.int32,2)
    lib.sw_macro8_predict.argtypes = [integer, grid, grid, integer]
    lib.sw_macro8_check.argtypes = [integer, grid, grid, real, risk]
    lib.sw_macro8_routed.argtypes = [integer, grid, grid, integer, real, risk]
    lib.sw_rollout.argtypes = [integer, grid, real, real, integer, real, integer]
    lib.sw_engineering_rollout.argtypes = lib.sw_rollout.argtypes
    lib.sw_macro8_classical.argtypes = [integer, grid, grid, integer, record]
    lib.sw_macro8_batch.argtypes = [integer,integer,batch,batch,integer,real,records]
    return lib


def read_thresholds(path=OUT / 'guard.json'):
    with open(path) as f:
        data = json.load(f)
    return {int(k): float(v) for k,v in data['thresholds'].items()}


def _grid(q):
    q = np.ascontiguousarray(q, dtype=np.float64)
    if q.shape != (N, 2):
        raise ValueError(f'Expected a ({N}, 2) grid of depth and discharge')
    return q


def classical_step(lib, method, q):
    q = _grid(q)
    out = np.empty_like(q)
    record = np.empty(3, np.int32)
    status = lib.sw_macro8_classical(len(q), q, out, METHODS.index(method), record)
    if status:
        raise RuntimeError(f'{method} returned {status}')
    return out, int(record[1])


def apply(lib, method, q, thresholds):
    if method in METHODS[:9]:
        return classical_step(lib, method, q)[0], -1
    q = _grid(q)
    out = np.empty_like(q)
    name = method.removeprefix('routed_')
    model = MODEL_NAMES.index(name)
    if name == method:
        status = lib.sw_macro8_predict(len(q),q,out,model)
        if status:
            raise RuntimeError(f'{method} returned {status}')
        return out,-1
    risk = np.empty(1)
    route = lib.sw_macro8_routed(len(q),q,out,model,thresholds[model],risk)
    if route < 0:
        raise RuntimeError(f'{method} returned {route}')
    return out,route


def budget_pass(errors):
    return np.all(errors <= ENDPOINT_BUDGET, axis=1)


def true_shock(q):
    """Coarse-cell position and midpoint depth of the unique compressive dam wave."""
    (hl,ml),(hr,mr) = q[[0,-1]]
    ul,ur = ml/hl,mr/hr
    star = float(reference(np.array([[hl,ul,hr,ur]]))[0])
    sign = 1 if hl>hr else -1
    h,u = (hr,ur) if sign == 1 else (hl,ul)
    speed = u + sign*math.sqrt(.5*G*star*(star+h)/h)
    jump = np.flatnonzero(q[1:,0] != q[:-1,0])
    if len(jump) != 1:
        raise ValueError('The shock diagnostic expects one original dam jump')
    return jump[0]+1+speed*DURATION*len(q), .5*(h+star), sign


def shock_error(q, prediction):
    position, level, sign = true_shock(q)
    h = prediction[:,0]
    dh = h[1:]-h[:-1]
    # Depth gradients across the compressive wave have the opposite sign to
    # the propagation direction in the common-background dam frame. In an
    # oscillatory profile choose the steepest crossing of this same level.
    crossings = (sign*dh < 0) & ((h[:-1]-level)*(h[1:]-level) <= 0)
    cells = np.flatnonzero(crossings)
    if not len(cells):
        return None
    j = cells[np.argmax(np.abs(dh[cells]))]
    estimate = j+.5+(level-h[j])/dh[j]
    return float(abs(estimate-position))


def error_statistics(values):
    return dict(mean=values.mean(axis=0).tolist(),
                p95=np.quantile(values,.95,axis=0).tolist(),
                maximum=values.max(axis=0).tolist())


def checksum(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def measure_split(lib, teacher, split, cases, repeats, thresholds, timer_seed):
    data = make_split(split,cases,teacher)
    q, target = data.q, data.target
    fine = np.empty_like(target)
    method_by_case = []
    for j, family in enumerate(data.families):
        if family == 'dam':
            fine[j] = analytical_dam(q[j])
            method_by_case.append('exact_riemann')
        else:
            fine[j] = teacher.refined(q[j],refinement=4)
            method_by_case.append('4x_godunov')
    teacher_fine = np.abs(target-fine).mean(axis=1)
    # Components must BOTH satisfy this cohort threshold. The casewise count
    # uses each individual baseline error and is reported separately.
    fine_target = FINE_ERROR_FACTOR * teacher_fine.mean(axis=0)
    timings = {m: [] for m in METHODS}
    outputs = {}
    routes = {}
    scratch = np.empty_like(q)
    records = np.empty((len(q),3),np.int32)
    rng = np.random.default_rng(timer_seed)
    for rep in range(-WARMUP,repeats):
        for mode in rng.permutation(len(METHODS)):
            mode = int(mode)
            m = METHODS[mode]
            threshold = thresholds[MODEL_NAMES.index(m[7:])] if m.startswith('routed_') else 0.
            t0 = perf_counter_ns()
            status = lib.sw_macro8_batch(len(q),N,q,scratch,mode,threshold,records)
            ns = perf_counter_ns()-t0
            if status:
                raise RuntimeError(f'Whole batch failed for {split}/{m}: status {status}')
            if rep >= 0:
                timings[m].append(ns/(1000*len(q)))
            if rep == -1:
                if not np.isfinite(scratch).all():
                    raise ArithmeticError(f'Nonfinite output from {m} in {split}')
                outputs[m] = scratch.copy()
                routes[m] = records.copy()
    np.testing.assert_array_equal(outputs['teacher8'],target)
    for name in MODEL_NAMES:
        routed = f'routed_{name}'
        accepted = routes[routed][:,0] == 1
        np.testing.assert_array_equal(outputs[routed][accepted],outputs[name][accepted])
        np.testing.assert_array_equal(outputs[routed][~accepted],target[~accepted])
    results, details = {}, {}
    for m in METHODS:
        predicted, rec = outputs[m], routes[m]
        delta = np.abs(predicted-target).mean(axis=1)
        fine_residual = np.abs(predicted-fine)
        fine_delta = fine_residual.mean(axis=1)
        max_coarse = np.abs(predicted-target).max(axis=1)
        max_fine = fine_residual.max(axis=1)
        passed = budget_pass(delta)
        shock = [shock_error(q[j], predicted[j]) for j in range(cases)
                 if data.families[j]=='dam']
        located = [s for s in shock if s is not None]
        fpass = np.all(fine_delta <= FINE_ERROR_FACTOR*teacher_fine + 1e-13,axis=1)
        vals = np.asarray(timings[m])
        desc = dict(time_us=float(np.median(vals)), timing_iqr_us=np.quantile(vals,[.25,.75]).tolist(),
                    timing_samples_us=vals.tolist(), coarse_error=error_statistics(delta),
                    endpoint_budget_pass=int(passed.sum()), fine_error=error_statistics(fine_delta),
                    meets_cohort_fine_target=bool(np.all(fine_delta.mean(axis=0) <= fine_target+1e-13)),
                    within_casewise_fine_target=int(fpass.sum()),
                    shock_mean_abs_cells=float(np.mean(located)) if located else None,
                    shock_dams_located=len(located), negative_depth_cases=int(np.any(predicted[:,:,0]<=0,axis=1).sum()),
                    mean_classical_steps=float(rec[:,1].mean()), retried_cases=int(rec[:,2].sum()))
        desc['max_cell_coarse_per_profile'] = error_statistics(max_coarse)
        desc['max_cell_fine_per_profile'] = error_statistics(max_fine)
        desc['by_family'] = {}
        for family in FAMILIES:
            ix = np.asarray(data.families)==family
            desc['by_family'][family] = dict(count=int(ix.sum()),coarse_mean=delta[ix].mean(axis=0).tolist(),
                                              fine_mean=fine_delta[ix].mean(axis=0).tolist(),
                                              worst_cell_fine=max_fine[ix].max(axis=0).tolist(),
                                              worst_cell_coarse=max_coarse[ix].max(axis=0).tolist(),
                                              endpoint_budget_pass=int(passed[ix].sum()))
        if m.startswith('routed_'):
            accepted = rec[:,0] == 1
            bare = outputs[m[7:]]
            safe_if_accepted = budget_pass(np.abs(bare-target).mean(axis=1))
            desc['guard'] = dict(accepted=int(accepted.sum()), fallback=int((~accepted).sum()),
                                 accepted_over_budget=int((accepted & ~safe_if_accepted).sum()),
                                 rejected_within_budget=int((~accepted & safe_if_accepted).sum()))
        results[m] = desc
        details[m] = dict(coarse_mean_abs=delta.tolist(), fine_mean_abs=fine_delta.tolist(),
                          dam_shock_absolute_cells=shock, route=rec[:,0].tolist())
    eligible = [m for m in METHODS if results[m]['meets_cohort_fine_target']]
    for name, row in results.items():
        row['teacher_time_over_method_time'] = results['teacher8']['time_us']/row['time_us']
    persistence = np.abs(q-target).mean(axis=1)
    persistence_ok = budget_pass(persistence)
    persistence_by_family = {}
    for family in FAMILIES:
        ix = np.asarray(data.families)==family
        persistence_by_family[family] = dict(count=int(ix.sum()), within_budget=int(persistence_ok[ix].sum()),
                                              coarse_mae=persistence[ix].mean(axis=0).tolist())
    return dict(ids=list(data.ids), families=list(data.families), reference=method_by_case,
                initial_sha256=checksum(q),teacher_sha256=checksum(target),reference_sha256=checksum(fine),
                fine_error_targets=fine_target.tolist(), methods=results, per_case=details,
                fastest_meeting_cohort_fine_target=min(eligible,key=lambda m:results[m]['time_us']),
                persistence_coarse_mean=persistence.mean(axis=0).tolist(),
                persistence_budget_pass=int(persistence_ok.sum()), persistence_by_family=persistence_by_family)


def benchmark(path=OUT/'benchmark.json', cases=128, rounds=13, cpu=None):
    if cases < 4 or rounds < 1:
        raise ValueError('Use at least four profiles and one timing round')
    affinity = None
    if hasattr(os, 'sched_getaffinity'):
        cores = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {cores[-1] if cpu is None else cpu})
        affinity = sorted(os.sched_getaffinity(0))
    elif cpu is not None:
        raise ValueError('CPU pinning is unavailable on this platform')
    thresholds = read_thresholds()
    source_paths = [HERE/n for n in ('macro8_workload.c', 'macro8_data.py', 'macro8_model.py',
                    'macro8_predict.c', 'macro8_guard.c', 'macro8_classical.c', 'rollout.c',
                    'solver.c', 'certificate.h')]
    source_paths += [p for p in sorted(HERE.glob('macro8*.h'))]
    source_paths += [HERE.parent / 'reports/shallow_water/engineering_control/newton.c',OUT/'guard.json']
    source_hashes = {str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in source_paths if p.exists()}
    with TemporaryDirectory() as d, ReferenceSolver() as solver:
        lib = build(Path(d))
        results = {split:measure_split(lib, solver, split, cases, rounds, thresholds, 34493+i)
                   for i,split in enumerate(('test','shift'))}
    desc = dict(cases_per_split=cases, repeats=rounds, warmups=WARMUP, grid_size=N,
                steps=STEPS, dt_over_dx=LAMBDA, duration_s=DURATION, gravity=G,
                seeds={s:SPLIT_SEEDS[s] for s in results}, endpoint_budget=ENDPOINT_BUDGET.tolist(),
                fine_error_factor=FINE_ERROR_FACTOR, thresholds=thresholds, compiler_flags=FLAGS,
                compiler=subprocess.check_output(['gcc','--version'],text=True).splitlines()[0],
                cpu_affinity=affinity, host=platform.platform(), python=platform.python_version(),
                numpy=np.__version__, source_sha256=source_hashes,
                coarse_teacher=f'Eight mode-6 Godunov steps with {TOL:g} m depth tolerance',
                pde_reference=('Analytic at cell-average level on dam inputs; fourfold spatial and temporal '
                               'refinement of the original Godunov solver starting from the same supplied, '
                               'piecewise-constant coarse profile for all other families.'),
                timings=('Median across randomized, warmed, single-core native full-batch elapsed times '
                         'divided by profile count. Includes input preparation internal to each method, '
                         'prediction, routing, and all fallback solves. Cached offline references excluded.'),
                cfl_policy=('Within a block choose ceil(8*dt/dx * max(|u|+sqrt(gh)) / .90) steps, clamped '
                            'to [1,8]. If the evolution reports a CFL or positivity failure with a reduced step count, '
                            'retry with eight substeps from the original profile. Both attempts are included.'),
                shock_metric=('Absolute shock-position error in coarse cells. Interpolate the crossing of the '
                              'midpoint between the analytic intermediate depth and the downstream depth, '
                              'choosing the steepest same-sign crossing when oscillatory. Dam cases only.'),
                splits=results)
    proc = Path('/proc/cpuinfo')
    if proc.exists():
        desc['cpu_model'] = next((line.split(':',1)[-1].strip() for line in proc.read_text().splitlines()
                                  if line.startswith('model name')), None)
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(desc,indent=2,allow_nan=False)+'\n')
    return desc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT/'benchmark.json')
    parser.add_argument('--cases',type=int,default=128)
    parser.add_argument('--rounds',type=int,default=13)
    parser.add_argument('--cpu',type=int)
    args = parser.parse_args()
    report = benchmark(args.out,args.cases,args.rounds,args.cpu)
    for split,panel in report['splits'].items():
        print(split, 'fastest eligible', panel['fastest_meeting_cohort_fine_target'])
        for name,row in panel['methods'].items():
            print(f"{name:16} {row['time_us']:8.2f} us  {row['endpoint_budget_pass']:3} within budget  "
                  f"field MAE {row['fine_error']['mean']}")
