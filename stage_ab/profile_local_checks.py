"""Measure the two numerical kernels of the original sparse local certificate."""
from pathlib import Path
from statistics import median
import argparse
import json
import time

import numpy as np

from .burgers import FrozenFlux
from .experiments_a2 import LAM, discretize, ic_family
from .experiments_refinement import COUNT, N, MODEL_PATH
from .stage_a2 import (
    CertifiedFlux, CertificateTables, FPConstants, cell_costs,
    _difference_bounds_on_stencils, _local_face_bounds,
)


def profile(repeats):
    model = CertifiedFlux(FrozenFlux.load(MODEL_PATH))
    tables = CertificateTables.build(model, bins=16)
    consts = FPConstants.build(LAM, tables.envelope)
    rows = []
    order = np.random.default_rng(4131)
    for index, ic in enumerate(ic_family(9702, 9)):
        state = discretize(ic, N)
        bins = tables.bin_index(state)
        table = cell_costs(model, tables, consts, state, mode="table")
        centers = np.argsort(-table[:, 1, 1], kind="stable")[:COUNT]
        left, right = (centers-1) % N, (centers+1) % N
        faces = np.union1d(left, centers)
        face_inputs = (state[faces], state[(faces+1) % N])
        stencils = state[left], state[centers], state[right]
        stencil_bins = bins[left], bins[centers], bins[right]
        kernels = {
            "point_values": lambda: _local_face_bounds(model, *face_inputs),
            "local_difference": lambda: _difference_bounds_on_stencils(
                model, tables, stencils, stencil_bins, "local"),
        }
        for fn in kernels.values():
            fn()
        times = {name: [] for name in kernels}
        for _ in range(repeats):
            for name in order.permutation(list(kernels)):
                start = time.perf_counter()
                kernels[name]()
                times[name].append(time.perf_counter() - start)
        rows.append({"index": index, "kind": ic["kind"], "cells": COUNT, "faces": len(faces),
                     "seconds": times})
    mean = {name: float(np.mean([median(r["seconds"][name]) for r in rows])) for name in kernels}
    return {"seed": 9702, "repeats": repeats, "step": 0, "n": N,
            "mean_seconds": mean, "results": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    report = profile(args.repeats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(report["mean_seconds"])
