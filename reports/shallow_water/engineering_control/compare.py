"""Empirical Newton control with no sign certificate, timed against certified methods."""
import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter_ns

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from shallow_water.benchmark import FLAGS, G, SEEDS
from shallow_water.data import reference, sample


def run(cpu, count, rounds):
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        cores = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {cores[0] if cpu is None else cpu})
        affinity = sorted(os.sched_getaffinity(0))
    elif cpu is not None:
        raise ValueError("CPU affinity selection is unavailable on this platform")
    timer_rng = np.random.default_rng(731289)
    methods = [("newton_rr", "engineering_batch", 5),
               ("newton_two_shock", "engineering_batch", 1),
               ("certified_rr", "sw_solve_batch", 5),
               ("certified_hard_nn", "sw_solve_batch", 3)]
    report = dict(affinity=affinity, seeds=SEEDS, count=count,
                  rounds=rounds, tol_m=0.01, newton_step_threshold_m=0.001,
                  flags=FLAGS, splits={})
    with TemporaryDirectory() as d:
        libpath = Path(d) / "control.so"
        subprocess.run(["gcc", *FLAGS, str(HERE / "newton.c"), "-lm", "-o", str(libpath)], check=True)
        lib = ct.CDLL(str(libpath))
        double = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
        integer = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
        for _, kernel, _ in methods:
            getattr(lib,kernel).argtypes = [ct.c_int,double,ct.c_double,ct.c_double,ct.c_int,double,integer]
        for split, seed in SEEDS.items():
            faces = sample(count, seed, split)
            oracle = reference(faces)
            depth = np.empty(count)
            stats = np.empty((count,4), dtype=np.int32)
            measurements = {name: [] for name, _, _ in methods}
            errors = {}
            for rep in range(-2, rounds):
                for j in timer_rng.permutation(len(methods)):
                    name, kernel, mode = methods[j]
                    start = perf_counter_ns()
                    invalid = getattr(lib,kernel)(count, faces, G, .01, mode, depth, stats)
                    elapsed = perf_counter_ns()-start
                    if invalid:
                        raise ValueError(f"{invalid} invalid outputs for {split}/{name}")
                    if rep>=0:
                        measurements[name].append(elapsed/count)
                    if rep==0:
                        err=np.abs(depth-oracle)
                        errors[name]=dict(max_abs_error_m=float(err.max()),
                                          count_over_0_01m=int(np.sum(err>.01)))
            report["splits"][split] = {
                name: {**errors[name], "median_ns_per_face":float(np.median(times)), "samples_ns_per_face":times}
                for name,times in measurements.items()}
    report["equal_split_mean_ns"] = {
        name:float(np.mean([v[name]["median_ns_per_face"] for v in report["splits"].values()]))
        for name, _, _ in methods}
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu",type=int)
    parser.add_argument("--count",type=int,default=8192)
    parser.add_argument("--rounds",type=int,default=41)
    args=parser.parse_args()
    result=run(args.cpu,args.count,args.rounds)
    (HERE/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(result["equal_split_mean_ns"])
