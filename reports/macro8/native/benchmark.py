"""Compare optimized native FP64 rollouts at equal physical horizons.

Run `OPENBLAS_NUM_THREADS=1 python reports/macro8/native/benchmark.py`.
With a shared machine use taskset to pin the process to a single available core.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
from statistics import mean, median
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from certified_burgers.godunov import godunov_step
from stage_ab.experiments_a2 import discretize, ic_family
from stage_ab.macro_controls import matched_macro
from stage_ab.macro_flux import MacroFlux, WEIGHTS as STRUCTURED_WEIGHTS
from stage_ab.macro_network import WEIGHTS, load_network

N = 128
MODES = ('godunov', 'neural', 'structured', 'large_godunov', 'fixed')
REPEATS = 21
TEST_SETS = ((241124, 9), (79023, 30))
HORIZONS = (8, 60, 64)
# Batches of independently initialized trajectories amortize the two native
# timer reads equally and keep short and long horizons at similar runtime.
BATCH_SIZE = {8: 256, 60: 32, 64: 32}
FLAGS = ['-std=c11', '-O3', '-march=native', '-fno-fast-math', '-ffp-contract=off', '-shared', '-fPIC']


def c_literal(a):
    if a.ndim == 0:
        return float(a).hex()
    return '{' + ','.join(c_literal(v) for v in a) + '}'


def compile_kernel(path):
    fields = [('W', 'weights', (10, 16)), ('b', 'bias', (16,)),
              ('v', 'weights_out', (16,)), ('c', 'bias_out', ())]
    header = []
    with np.load(WEIGHTS) as model:
        for key, name, shape in fields:
            a = model[key]
            assert a.shape == shape and a.dtype == np.float64
            dims = ''.join(f'[{n}]' for n in shape)
            header.append(f'static const double {name}{dims} = {c_literal(a)};')
    for name, model in [('structured_weights', MacroFlux.load()), ('fixed_weights', matched_macro())]:
        coeffs = model.weights
        assert coeffs.shape == (5,) and coeffs.dtype == np.float64
        header.append(f'static const double {name}[5] = {c_literal(coeffs)};')
    (path / 'weights.h').write_text('\n'.join(header) + '\n')
    lib = path / 'kernel.so'
    subprocess.run(['gcc', *FLAGS, '-I', str(path), str(HERE / 'kernel.c'), '-o', str(lib)], check=True)
    run = ctypes.CDLL(str(lib)).batch
    array = np.ctypeslib.ndpointer(np.float64, ndim=1, shape=(N,), flags='C_CONTIGUOUS')
    run.argtypes = [array, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, array]
    run.restype = ctypes.c_int64
    return run


def call(run, u, mode, horizon, reps, out):
    macros, tail = divmod(horizon, 8)
    return run(u, mode, macros, tail, reps, out)


def validate(cases, run, result):
    errors = {name: {h: 0. for h in HORIZONS} for name in MODES}
    predictors = (load_network(), MacroFlux.load().step, matched_macro().step)
    max_cfl = 0.
    for case, u in cases:
        ref, large = u.copy(), u.copy()
        learned = [u.copy() for _ in predictors]
        for step in range(1, max(HORIZONS) + 1):
            ref, _ = godunov_step(ref, 1.0, dt=0.2)
            if step % 4 == 0:
                max_cfl = max(max_cfl, .8 * np.max(np.abs(large)))
                large, _ = godunov_step(large, 1.0, dt=0.8)
            if step % 8 == 0:
                learned = [advance(s) for advance, s in zip(predictors, learned)]
            if step in HORIZONS:
                learned_tail = learned.copy()
                for _ in range(step % 8):
                    learned_tail = [godunov_step(s, 1.0, dt=0.2)[0] for s in learned_tail]
                for mode, expected in enumerate((ref, learned_tail[0], learned_tail[1], large, learned_tail[2])):
                    name = MODES[mode]
                    call(run, u, mode, step, 1, result)
                    np.testing.assert_allclose(result, expected, atol=5e-12, rtol=0, err_msg=str(case))
                    errors[name][step] = max(errors[name][step], float(np.max(np.abs(expected - result))))
    return errors, float(max_cfl)


def main():
    cases = [({'seed': seed, 'index': j, 'kind': ic['kind']}, discretize(ic, N))
             for seed, size in TEST_SETS for j, ic in enumerate(ic_family(seed, size))]
    rng = np.random.default_rng(8901)
    result = np.empty(N)
    configs = [(mode, h) for mode in range(len(MODES)) for h in HORIZONS]
    out = {'cells': N, 'fine_dt_dx': .2, 'fine_steps_per_macro_call': 8,
           'model_sha256': hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),
           'structured_sha256': hashlib.sha256(STRUCTURED_WEIGHTS.read_bytes()).hexdigest(),
           'compiler': subprocess.check_output(['gcc', '--version'], text=True).splitlines()[0],
           'flags': FLAGS, 'machine': platform.machine(), 'cpu_affinity': sorted(os.sched_getaffinity(0)),
           'cpu_model': next(s.split(':', 1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines()
                             if s.startswith('model name')),
           'batch_size': BATCH_SIZE, 'timed_batches_per_case': REPEATS,
           'aggregation': 'Per-case median batch wall ns/trajectory; arithmetic mean of case medians.',
           'cases': []}
    with TemporaryDirectory() as tmp:
        run = compile_kernel(Path(tmp))
        out['numpy_parity_max_abs'], out['max_cfl_large_godunov'] = validate(cases, run, result)
        for case, u in cases:
            sample = {cfg: [] for cfg in configs}
            for mode, h in configs:
                call(run, u, mode, h, BATCH_SIZE[h], result)
            for _ in range(REPEATS):
                for j in rng.permutation(len(configs)):
                    mode, h = configs[j]
                    wall_ns = call(run, u, mode, h, BATCH_SIZE[h], result)
                    assert wall_ns > 0
                    sample[mode, h].append(wall_ns / BATCH_SIZE[h])
            out['cases'].append({**case, 'ns_per_trajectory': {
                name: {h: sample[mode, h] for h in HORIZONS}
                for mode, name in enumerate(MODES)}})
    summary = {name: {h: mean(median(case['ns_per_trajectory'][name][h]) for case in out['cases'])
                      for h in HORIZONS} for name in MODES}
    out['mean_ns_per_trajectory'] = summary
    out['speedup_vs_standard_godunov'] = {name: {h: summary['godunov'][h] / summary[name][h]
                                                 for h in HORIZONS} for name in MODES}
    (HERE / 'timing.json').write_text(json.dumps(out, indent=1) + '\n')
    for key in ('numpy_parity_max_abs', 'mean_ns_per_trajectory', 'speedup_vs_standard_godunov'):
        print(key, out[key])


if __name__ == '__main__':
    main()
