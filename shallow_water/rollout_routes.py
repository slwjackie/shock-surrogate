"""Untimed route audit of the 256-cell wet shallow-water trajectories.

The monitored translation unit wraps the real sw_solve call made by rollout.c.
A separately built uninstrumented solver must return identical terminal states.
"""
import argparse
import ctypes
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np

from .benchmark import FLAGS, METHODS
from .rollout_study import (CASES, DURATION, LENGTH, ROOT_TOLERANCE,
                            compile_rollout, exact_cell_averages, initial, provenance)

MODES = ("two_rarefaction", "nn_initial", "nn_uniform", "nn_hard")
PACKAGE = Path(__file__).resolve().parent
REPORT = PACKAGE.parent / "reports" / "shallow_water" / "rollout_routes.json"

AUDIT_SOURCE = r"""
int sw_solve(double, double, double, double, double, double, int, double *, int *);

/* This diagnostic library is called from one thread and a single trajectory at
   a time. The independently timed production library has no counter at all. */
static long long route_counts[6];
static int audit_sw_solve(double hl, double ul, double hr, double ur,
                          double g, double tol, int mode, double *h, int *stats) {
    int status = sw_solve(hl, ul, hr, ur, g, tol, mode, h, stats);
    if (status >= 0) {
        route_counts[stats[0]]++;
        route_counts[3] += stats[1];
        route_counts[4] += stats[2];
        route_counts[5] += stats[3];
    }
    return status;
}

#define sw_solve audit_sw_solve
#include "rollout.c"
#undef sw_solve

int sw_audit(int n, double *q, double length, double duration, int steps,
             double tol, int mode, long long *out) {
    memset(route_counts, 0, sizeof(route_counts));
    int status = sw_rollout(n, q, length, duration, steps, tol, mode);
    memcpy(out, route_counts, sizeof(route_counts));
    return status;
}
"""


def audit(out):
    with TemporaryDirectory() as tmp:
        work = Path(tmp)
        regular = compile_rollout(work)
        source = work / "audit.c"
        source.write_text(AUDIT_SOURCE)
        library = work / "audit.so"
        subprocess.run(["gcc", *FLAGS, str(source), str(PACKAGE/"solver.c"),
                        "-I", str(PACKAGE), "-lm", "-o", str(library)], check=True)
        monitored = ctypes.CDLL(str(library)).sw_audit
        monitor_arg = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1,
                                              shape=(6,), flags="C_CONTIGUOUS")
        monitored.argtypes = [*regular.argtypes, monitor_arg]
        monitored.restype = ctypes.c_int
        rows = []
        for name, case in CASES.items():
            n = 256
            _, steps = exact_cell_averages(case, n)
            for method in MODES:
                state = initial(case,n)
                check = state.copy()
                counts = np.zeros(6, dtype=np.int64)
                args = (LENGTH, DURATION, steps, ROOT_TOLERANCE, METHODS.index(method))
                if monitored(n, state, *args, counts) or regular(n, check, *args):
                    raise RuntimeError(f"Route audit could not complete {name} with {method}")
                np.testing.assert_array_equal(state, check)
                analytic, direct, corrected, newton, bisection, certificates = map(int, counts)
                faces = (n-1)*steps
                bypassed = faces - analytic - direct - corrected
                assert bypassed >= 0
                rows.append(dict(case=name, method=method, interior_faces=faces,
                                 equal_or_upwind_bypasses=bypassed, analytic_accepted=analytic,
                                 proposed_accepted=direct, corrected=corrected,
                                 newton_updates=newton, bisections=bisection,
                                 certificate_calls=certificates, terminal_bitwise_parity=True))
    obj = dict(grid_cells=256, duration=DURATION, local_tolerance_m=ROOT_TOLERANCE,
               definition="Interior-face events on each mode's own trajectory; bypasses include equal states and unequivocally upwind fluxes.",
               provenance=provenance(), results=rows)
    out.write_text(json.dumps(obj, indent=2)+"\n")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPORT)
    args = parser.parse_args()
    for row in audit(args.out):
        print(row)
