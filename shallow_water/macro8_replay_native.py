"""Build and run the frozen whole-field models trained on their own predictions.

Arms 0, 1 and 2 select the original hard-case model, uniform on-policy replay
and hard-case on-policy replay, respectively. The primary deployment stays
unchanged. A temporary build directory belongs to the caller.
"""
import ctypes
from hashlib import sha256
import json
from pathlib import Path
from shutil import copyfile
from tempfile import mkdtemp

import numpy as np

from .macro8_data import N
from .macro8_guard import build_guard
from .macro8_model import CHANNELS, HEADER, INPUTS, OUT, TAPS, WIDTH, export

ARMS = ("previous_hard", "on_policy_uniform", "on_policy_hard")


def _arrays(path, expected_sha, count):
    if sha256(path.read_bytes()).hexdigest() != expected_sha:
        raise ValueError(f"Model archive does not match its frozen hash: {path.name}")
    arrays = []
    with np.load(path, allow_pickle=False) as archive:
        for key, shape in (("encoder", (INPUTS,WIDTH)), ("bias", (WIDTH,)),
                           ("kernel", (TAPS,CHANNELS,2))):
            v = archive[key]
            if v.shape != (count,*shape) or v.dtype != np.float64 or not np.isfinite(v).all():
                raise ValueError(f"Invalid {key} weights in {path.name}")
            arrays.append(v)
    return list(zip(*arrays))


def replay_source(directory):
    """Return the native predictor .c with a sibling header in private build storage.

    The original .c is copied verbatim. Compile the returned source instead of
    the original predictor when linking a custom benchmark or application.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    folder = Path(mkdtemp(prefix="macro8_replay_",dir=directory))
    report = json.loads((OUT/"replay.json").read_text())
    if tuple(report["frozen_modes"]) != ARMS[1:]:
        raise ValueError("Unexpected on-policy model order")
    primary = _arrays(OUT/"weights.npz",report["source_weights_sha256"],3)
    replay = _arrays(OUT/"replay.npz",report["weight_sha256"],2)
    export([primary[2], *replay],folder,folder/"macro8_weights.h")
    source = folder/"macro8_predict.c"
    copyfile(HEADER.with_name("macro8_predict.c"),source)
    return source


def build_replay_native(directory):
    """Load a native predictor, guard and Godunov fallback with replay weights."""
    predictor = replay_source(directory)
    return build_guard(predictor.parent,predictor)


def replay_step(lib, state, arm=1, threshold=None):
    """Return the eight-step field and route (-1 bare, 1 accepted, 0 fallback).

    A threshold is supplied by a calibration for these weights and the caller's
    error target. Negative thresholds force fallback and infinity disables
    only the score ceiling; hard physical checks still run on the routed path.
    """
    if isinstance(arm,(bool,np.bool_)) or not isinstance(arm,(int,np.integer)) or not 0<=arm<len(ARMS):
        raise ValueError(f"arm must be 0, 1 or 2 for {ARMS}")
    q = np.ascontiguousarray(state,dtype=np.float64)
    if q.shape != (N,2) or not np.isfinite(q).all() or np.any(q[:,0]<=0):
        raise ValueError(f"Expected {N} finite rows of (positive depth, discharge)")
    out = np.empty_like(q)
    if threshold is None:
        status = lib.sw_macro8_predict(N,q,out,int(arm))
        if status:
            raise RuntimeError(f"Native eight-step predictor returned {status}")
        route = -1
    else:
        if not np.isscalar(threshold) or np.isnan(float(threshold)):
            raise ValueError("The guard threshold must be a real scalar or infinity")
        risk = ctypes.c_double()
        route = lib.sw_macro8_routed(N,q,out,int(arm),float(threshold),ctypes.byref(risk))
        if route<0:
            raise RuntimeError(f"Native eight-step route returned {route}")
    if not np.isfinite(out).all() or np.any(out[:,0]<=0):
        raise FloatingPointError("Eight-step prediction produced a nonfinite field or nonpositive depth")
    return out, int(route)
