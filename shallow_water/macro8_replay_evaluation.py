"""Freeze replay-model thresholds, then evaluate new, disjoint one-block profiles."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter_ns

import numpy as np

from .benchmark import FLAGS
from .macro8_benchmark import (
    ENDPOINT_BUDGET, FINE_ERROR_FACTOR, HERE, OUT, METHODS, budget_pass,
    checksum, error_statistics, read_thresholds, shock_error,
)
from .macro8_data import Batch, FAMILIES, N, SPLIT_SEEDS, ReferenceSolver, analytical_dam, make_split
from .macro8_guard import fit_cutoff
from .macro8_replay_native import replay_source

MODEL_NAMES = ('original_hard', 'onpolicy_uniform', 'onpolicy_hard')
GUARD = OUT/'replay_guard.json'
RESULT = OUT/'replay_evaluation.json'
FIRST_NEW_ID = 320
DEFAULT_MODEL = 'onpolicy_uniform'


def build_replay_workload(folder):
    """Same native benchmark ABI, with three frozen replay models substituted."""
    folder = Path(folder)
    predictor = replay_source(folder)
    binary = folder/'replay_workload.so'
    subprocess.run(['gcc', *FLAGS, str(predictor), str(HERE/'macro8_workload.c'),
                    str(HERE/'macro8_guard.c'), str(HERE/'rollout.c'),
                    str(HERE/'macro8_classical.c'), '-lm', '-o', str(binary)], check=True)
    lib = ctypes.CDLL(str(binary.resolve()))
    def matrix(t, rank):
        return np.ctypeslib.ndpointer(t, ndim=rank, flags='C_CONTIGUOUS')
    grid, batch = matrix(np.float64, 2), matrix(np.float64, 3)
    risk, record, records = matrix(np.float64, 1), matrix(np.int32, 1), matrix(np.int32, 2)
    i, f = ctypes.c_int, ctypes.c_double
    lib.sw_macro8_predict.argtypes = [i, grid, grid, i]
    lib.sw_macro8_check.argtypes = [i, grid, grid, f, risk]
    lib.sw_macro8_routed.argtypes = [i, grid, grid, i, f, risk]
    lib.sw_rollout.argtypes = [i, grid, f, f, i, f, i]
    lib.sw_engineering_rollout.argtypes = lib.sw_rollout.argtypes
    lib.sw_macro8_classical.argtypes = [i, grid, grid, i, record]
    lib.sw_macro8_batch.argtypes = [i, i, batch, batch, i, f, records]
    return lib


def confirmatory_split(split, count, teacher, start=FIRST_NEW_ID):
    """Disjoint ICs indexed by child SeedSequences of the original holdout seed."""
    if split not in ('test', 'shift') or start < FIRST_NEW_ID or count < 4:
        raise ValueError('Use test/shift with at least 4 cases and a start index >=320')
    b = make_split(split, start + count, teacher)
    return Batch(q=b.q[start:], target=b.target[start:], families=b.families[start:], ids=b.ids[start:])


def write_snapshot(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def calibrate_replay(lib, teacher, path=GUARD):
    b = make_split('calibrate',256,teacher)
    thresholds, summary = {}, {}
    y = np.empty_like(b.q[0])
    risk = np.empty(1,np.float64)
    for model, name in enumerate(MODEL_NAMES):
        scores = np.full(len(b.q),np.inf)
        errors = np.full((len(b.q),2),np.inf)
        for j,(q, target) in enumerate(zip(b.q,b.target)):
            if lib.sw_macro8_predict(N,q,y,model):
                continue
            lib.sw_macro8_check(N,q,y,np.inf,risk)
            scores[j] = risk[0]
            errors[j] = np.abs(y-target).mean(axis=0)
        accurate = budget_pass(errors)
        threshold = fit_cutoff(scores, accurate)
        accepted = np.isfinite(scores) & (scores<=threshold)
        thresholds[str(model)] = threshold
        summary[name] = dict(accepted=int(accepted.sum()), within_budget=int(accurate.sum()),
                             accepted_over_budget=int((accepted & ~accurate).sum()),
                             score_sha256=checksum(scores))
    expected = read_thresholds()[2]
    if thresholds['0'] != expected:
        raise RuntimeError('Original frozen hard model must reproduce primary cutoff')
    obj = dict(split='calibrate', seed=SPLIT_SEEDS['calibrate'], sample_count=len(b.q),
               calibrated_before_new_holdout=True, ids=b.ids, input_sha256=checksum(b.q),
               endpoint_budget=ENDPOINT_BUDGET.tolist(), model_order=MODEL_NAMES, thresholds=thresholds,
               models=summary, deployment_choice_before_testing=DEFAULT_MODEL,
               calibration_rule='Use the original fit_cutoff at the original mean absolute h and hu tolerances.',
               weights_sha256=hashlib.sha256((OUT/'replay.npz').read_bytes()).hexdigest())
    write_snapshot(path,obj)
    return {int(k):v for k,v in thresholds.items()}


def entries():
    for m in ('teacher8','engineering8','engineering_cfl','godunov_cfl','hlle8','hlle_cfl','rusanov8','rusanov_cfl'):
        yield m, METHODS.index(m), None
    for j,name in enumerate(MODEL_NAMES):
        yield name, METHODS.index('initial')+j, None
        yield 'routed_'+name, METHODS.index('routed_initial')+j, j


def run_panel(split, lib, teacher, thresholds, count, rounds, seed):
    b = confirmatory_split(split,count,teacher)
    reference = np.stack([analytical_dam(q) if f=='dam' else teacher.refined(q,4)
                          for q,f in zip(b.q,b.families)])
    teacher_accuracy = np.abs(reference-b.target).mean(axis=1)
    target = FINE_ERROR_FACTOR*teacher_accuracy.mean(axis=0)
    methods = list(entries())
    samples = {m:[] for m,_,_ in methods}
    out = np.empty_like(b.q)
    records = np.empty((count,3),np.int32)
    frozen, routes = {}, {}
    rng = np.random.default_rng(seed)
    for rep in range(-2,rounds):
        for order in rng.permutation(len(methods)):
            name, mode, model = methods[order]
            threshold = thresholds[model] if model is not None else 0.
            start = perf_counter_ns()
            bad = lib.sw_macro8_batch(count,N,b.q,out,mode,threshold,records)
            ns = perf_counter_ns()-start
            if bad or not np.isfinite(out).all():
                raise RuntimeError(f'Native {split}/{name} batch failed: {bad}')
            if rep == -1:
                frozen[name] = out.copy()
                routes[name] = records[:,0].copy()
            if rep >= 0:
                samples[name].append(ns/(1000*count))
    np.testing.assert_array_equal(frozen['teacher8'], b.target)
    dam = [i for i,f in enumerate(b.families) if f=='dam']
    report = {}
    for name,_,model in methods:
        pred = frozen[name]
        elapsed = samples[name]
        error = np.abs(pred-b.target).mean(axis=1)
        field = np.abs(pred-reference).mean(axis=1)
        within = budget_pass(error)
        casewise = np.all(field <= FINE_ERROR_FACTOR*teacher_accuracy+1e-13, axis=1)
        shock = [shock_error(b.q[i],pred[i]) for i in dam]
        located = [s for s in shock if s is not None]
        row = dict(time_us=float(np.median(elapsed)), timing_iqr_us=np.quantile(elapsed,[.25,.75]).tolist(),
                   timing_samples_us=elapsed, endpoint_budget_pass=int(within.sum()), coarse_mae=error_statistics(error),
                   pde_mae=error_statistics(field), meets_cohort_pde_target=bool(np.all(field.mean(axis=0)<=target+1e-13)),
                   within_casewise_pde_target=int(casewise.sum()), mean_abs_shock_cells=float(np.mean(located)) if located else None,
                   dams_located=len(located), input_count=count, output_sha256=checksum(pred))
        if model is not None:
            accepted = routes[name]==1
            bare = frozen[MODEL_NAMES[model]]
            np.testing.assert_array_equal(pred[accepted],bare[accepted])
            np.testing.assert_array_equal(pred[~accepted],b.target[~accepted])
            bare_ok = budget_pass(np.abs(bare-b.target).mean(axis=1))
            row['guard'] = dict(accepted=int(accepted.sum()), fallback=int((~accepted).sum()),
                                accepted_over_budget=int((accepted & ~bare_ok).sum()),
                                rejected_within_budget=int((~accepted & bare_ok).sum()))
            row['accepted_error_ids'] = [b.ids[i] for i in np.flatnonzero(accepted & ~bare_ok)]
        report[name]=row
    eligible=[m for m,row in report.items() if row['meets_cohort_pde_target']]
    return dict(ids=b.ids, families=b.families, input_sha256=checksum(b.q),
                fine_reference_sha256=checksum(reference), teacher_sha256=checksum(b.target),
                pde_cohort_error_budgets=target.tolist(), fastest_eligible=min(eligible,key=lambda m:report[m]['time_us']),
                methods=report)


def run(calibration_path, report_path, cases, rounds, cpu=None):
    if cases<4 or rounds<1:
        raise ValueError('Use at least four cases and one timing sample')
    core=None
    if hasattr(os,'sched_getaffinity'):
        core=max(os.sched_getaffinity(0)) if cpu is None else cpu
        os.sched_setaffinity(0,{core})
    with TemporaryDirectory() as folder, ReferenceSolver() as teacher:
        lib=build_replay_workload(Path(folder))
        thresholds=calibrate_replay(lib,teacher,calibration_path)
        data={s:run_panel(s,lib,teacher,thresholds,cases,rounds,26458+i)
              for i,s in enumerate(('test','shift'))}
    files = ('macro8_replay_native.py','macro8_rollout_replay.py','macro8_predict.c','macro8_guard.c',
             'macro8_classical.c','macro8_workload.c','macro8_data.py','rollout.c','solver.c',
             'macro8_replay_evaluation.py')
    report=dict(default_model_preselected=DEFAULT_MODEL, frozen_weights=MODEL_NAMES, counts=cases, warmups=2, rounds=rounds,
                start_index=FIRST_NEW_ID, cpu=core, compiler_flags=FLAGS, compiler=subprocess.check_output(
                    ['gcc','--version'],text=True).splitlines()[0], thresholds=thresholds,
                calibration_sha256=hashlib.sha256(Path(calibration_path).read_bytes()).hexdigest(),
                code_sha256={f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in files},
                endpoint_budget=ENDPOINT_BUDGET.tolist(), pde_factor=FINE_ERROR_FACTOR,
                split_notes='The case range starting at 320 was held out of all prior model training and development. Each profile is an independent initial condition.',
                pde_reference='Cell averages from analytic Riemann solution for dams, fourfold-refined Godunov solution starting from the same supplied initial profile otherwise.',
                timing='Single native batch; mean time per full initialized/output profile. Warmed, interleaved median across rounds.',
                splits=data)
    write_snapshot(report_path,report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=RESULT)
    parser.add_argument('--guard-out',type=Path,default=GUARD)
    parser.add_argument('--cases',type=int,default=64)
    parser.add_argument('--rounds',type=int,default=13)
    parser.add_argument('--cpu',type=int)
    args=parser.parse_args()
    result=run(args.guard_out,args.out,args.cases,args.rounds,args.cpu)
    for split,panel in result['splits'].items():
        print(split,panel['fastest_eligible'])
        for name,row in panel['methods'].items():
            print(name, round(row['time_us'],2),row['endpoint_budget_pass'],row['meets_cohort_pde_target'],row.get('guard'))
