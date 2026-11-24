"""Single-thread, compute-only native comparison on the nine confirmation states.

Call `python reports/cheap_routing/native/benchmark.py` from the repository.
There is no gate, training, error certificate, or budget in this diagnostic.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
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
from certified_burgers.godunov import godunov_flux
from stage_ab.frozen_flux import FrozenFlux
from stage_ab.experiments_a2 import discretize

N, STEPS, REPEATS = 128, 60, 17
COUNTS = [0, 64, 128]
FLAGS = ["-std=c11", "-O3", "-march=native", "-fno-fast-math", "-ffp-contract=off", "-fPIC", "-shared"]


def c_array(x):
    if x.ndim == 1:
        return "{" + ",".join(float(t).hex() for t in x) + "}"
    return "{" + ",".join(c_array(row) for row in x) + "}"


def compile_kernel(temp, model):
    header = []
    for depth, (weights, biases) in enumerate(model.layers):
        for name, array in [(f"w{depth}", weights), (f"b{depth}", biases)]:
            dims = "".join(f"[{n}]" for n in array.shape)
            header.append(f"static const double {name}{dims}={c_array(array)};")
    (temp / "weights.h").write_text("\n".join(header))
    library = temp / "flux.so"
    subprocess.run(["gcc", *FLAGS, "-I", str(temp), str(HERE / "kernel.c"), "-o", str(library)], check=True)
    compiled = ctypes.CDLL(str(library)).rollout
    array = np.ctypeslib.ndpointer(np.float64, ndim=1, shape=(N,), flags="C_CONTIGUOUS")
    compiled.argtypes = [array, ctypes.c_int, ctypes.c_int, array]
    compiled.restype = ctypes.c_int64
    return compiled


def ref_step(state, count, model):
    right = np.roll(state, -1)
    flux = np.empty(N)
    if count:
        flux[:count] = model.predict(np.column_stack([state[:count], right[:count]]))
    if count < N:
        flux[count:] = godunov_flux(state[count:], right[count:])
    return np.clip(state - .2 * (flux - np.roll(flux, 1)), -2., 2.)


def main():
    model_path = ROOT / "reports/stage_a2/flux_w16_s0.json"
    model = FrozenFlux.load(model_path)
    assert [w.shape for w, _ in model.layers] == [(16, 2), (16, 16), (1, 16)]
    source = ROOT / "reports/verifier_allocation/confirmation.json"
    cases = {r["index"]: r["ic"] for r in json.loads(source.read_text())["results"]}
    assert sorted(cases) == list(range(9))
    out = {"n": N, "steps": STEPS, "fixed_trusted_prefix": COUNTS, "repeats": REPEATS,
           "seconds_aggregation": "per-case median and arithmetic mean over 9 cases",
           "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
           "confirmation_source": str(source.relative_to(ROOT)),
           "compiler": subprocess.check_output(["gcc", "--version"], text=True).splitlines()[0],
           "flags": FLAGS, "architecture": platform.machine(), "results": []}
    rng = np.random.default_rng(67)
    dest = np.empty(N)
    with TemporaryDirectory() as tmp:
        run = compile_kernel(Path(tmp), model)
        states = {i: discretize(cases[i], N) for i in sorted(cases)}
        for i, initial in states.items():
            for count in COUNTS:
                reference = initial.copy()
                for step in range(1, STEPS+1):
                    reference = ref_step(reference, count, model)
                    if step in (1, 8, STEPS):
                        run(initial, count, step, dest)
                        np.testing.assert_array_equal(reference.view(np.uint64), dest.view(np.uint64))
        for i, initial in states.items():
            samples = {k: [] for k in COUNTS}
            for k in COUNTS:
                run(initial, k, STEPS, dest)
            for _ in range(REPEATS):
                for k in rng.permutation(COUNTS):
                    elapsed_ns = run(initial, int(k), STEPS, dest)
                    assert elapsed_ns > 0
                    samples[int(k)].append(1.e-9 * elapsed_ns)
            out["results"].append({"case": i, "kind": cases[i]["kind"], "wall_s": samples})
    out["bitwise_parity_checks"] = {"cases": 9, "routes": 3, "checked_steps": [1, 8, 60]}
    out["mean_s"] = {k: mean(median(r["wall_s"][k]) for r in out["results"]) for k in COUNTS}
    (HERE / "timing.json").write_text(json.dumps(out, indent=2) + "\n")
    print(out["mean_s"])


if __name__ == "__main__":
    main()
