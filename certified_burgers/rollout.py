"""Autoregressive trust-or-fallback rollouts for the certified Burgers PoC.

The diagnostic rollout intentionally computes the trusted Godunov update even
on accepted steps so the oracle local advice error eta_t is observable for
research evaluation. Runtime benchmarking is performed in a separate pass so
these diagnostic solver calls do not contaminate deployable-policy timings.
"""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
from typing import Iterable
import numpy as np
import torch
from .godunov import advance
from .oracle_error import l1_error
from .verifiers import weak_residual_score, mc_dropout_uncertainty

POLICIES = ("always_solver", "always_neural", "oracle", "residual", "uncertainty")

@dataclass
class RolloutResult:
    policy: str
    threshold: float | None
    final_error: float
    accept_rate: float
    fallback_rate: float
    accepted_eta_sum: float
    theorem_holds: bool
    max_theorem_violation: float
    trajectory_deviation: list[float]
    cumulative_accepted_eta: list[float]
    eta: list[float]
    scores: list[float]
    accepted: list[bool]
    shock_strength: list[float]


def _device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def predict_numpy(model, state):
    device=_device(model)
    x=torch.as_tensor(np.asarray(state)[None],dtype=torch.float32,device=device)
    was_training=model.training
    model.eval()
    with torch.no_grad():
        y=model(x)[0]
    model.train(was_training)
    return y.detach().cpu().numpy().astype(np.float64,copy=False)


def uncertainty_numpy(model, state, samples=8, seed=None):
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    device=_device(model)
    x=torch.as_tensor(np.asarray(state)[None],dtype=torch.float32,device=device)
    q=mc_dropout_uncertainty(model,x,samples=samples)
    return float(q.detach().cpu().numpy()[0])


def compressive_shock_strength(state):
    """Cheap Burgers shock sensor: largest positive left-to-right jump u_i-u_{i+1}."""
    u=np.asarray(state,dtype=np.float64)
    return float(np.max(np.maximum(u-np.roll(u,-1),0.0)))


def _score_and_decision(policy, current, pred, fallback, *, dx, macro_dt, threshold, model, mc_samples, seed):
    eta=float(l1_error(pred[None],fallback[None],dx)[0])
    if policy=="always_solver":
        return float("nan"), False, eta
    if policy=="always_neural":
        return float("nan"), True, eta
    if threshold is None:
        raise ValueError(f"policy={policy} requires a threshold")
    if policy=="oracle":
        score=eta
    elif policy=="residual":
        score=float(weak_residual_score(current[None],pred[None],dx=dx,dt=macro_dt)[0])
    elif policy=="uncertainty":
        score=uncertainty_numpy(model,current,samples=mc_samples,seed=seed)
    else:
        raise ValueError(f"Unknown policy {policy!r}")
    return score, bool(score<=float(threshold)), eta


def rollout_diagnostics(model, u0, *, dx, dt, horizon=1, macro_steps=20, policy="always_neural",
                        threshold=None, boundary="periodic", mc_samples=8, seed=0, theorem_tol=1e-10):
    """Run one autoregressive trajectory and audit Theorem-1's oracle eta bound."""
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    current=np.asarray(u0,dtype=np.float64).copy()
    classical=current.copy()
    cumulative_eta=0.0
    deviations=[0.0]; bounds=[0.0]
    etas=[]; scores=[]; accepted=[]; shocks=[]
    for t in range(int(macro_steps)):
        pred=predict_numpy(model,current)
        fallback,_=advance(current,dx,horizon,dt=dt,boundary=boundary)
        score,accept,eta=_score_and_decision(
            policy,current,pred,fallback,dx=dx,macro_dt=dt*horizon,threshold=threshold,
            model=model,mc_samples=mc_samples,seed=seed+t,
        )
        shocks.append(compressive_shock_strength(current))
        etas.append(eta); scores.append(score); accepted.append(accept)
        if accept:
            cumulative_eta += eta
            current=pred
        else:
            current=fallback
        classical,_=advance(classical,dx,horizon,dt=dt,boundary=boundary)
        deviation=float(l1_error(current[None],classical[None],dx)[0])
        deviations.append(deviation); bounds.append(cumulative_eta)
    violations=np.asarray(deviations)-np.asarray(bounds)
    return RolloutResult(
        policy=policy, threshold=None if threshold is None else float(threshold),
        final_error=float(deviations[-1]), accept_rate=float(np.mean(accepted)) if accepted else 0.0,
        fallback_rate=float(1.0-np.mean(accepted)) if accepted else 0.0,
        accepted_eta_sum=float(cumulative_eta), theorem_holds=bool(np.max(violations)<=theorem_tol),
        max_theorem_violation=float(np.max(violations)), trajectory_deviation=[float(x) for x in deviations],
        cumulative_accepted_eta=[float(x) for x in bounds], eta=[float(x) for x in etas],
        scores=[float(x) for x in scores], accepted=[bool(x) for x in accepted],
        shock_strength=[float(x) for x in shocks],
    )


def _runtime_one(model, u0, *, dx, dt, horizon, macro_steps, policy, threshold, boundary, mc_samples, seed):
    current=np.asarray(u0,dtype=np.float64).copy()
    for t in range(int(macro_steps)):
        if policy=="always_solver":
            current,_=advance(current,dx,horizon,dt=dt,boundary=boundary)
            continue
        pred=predict_numpy(model,current)
        if policy=="always_neural":
            current=pred; continue
        if policy=="oracle":
            fallback,_=advance(current,dx,horizon,dt=dt,boundary=boundary)
            eta=float(l1_error(pred[None],fallback[None],dx)[0])
            current=pred if eta<=float(threshold) else fallback
        elif policy=="residual":
            score=float(weak_residual_score(current[None],pred[None],dx=dx,dt=dt*horizon)[0])
            if score<=float(threshold): current=pred
            else: current,_=advance(current,dx,horizon,dt=dt,boundary=boundary)
        elif policy=="uncertainty":
            score=uncertainty_numpy(model,current,samples=mc_samples,seed=seed+t)
            if score<=float(threshold): current=pred
            else: current,_=advance(current,dx,horizon,dt=dt,boundary=boundary)
        else:
            raise ValueError(f"Unknown policy {policy!r}")
    return current


def benchmark_policy_runtime(model, initial_states: Iterable[np.ndarray], *, dx, dt, horizon, macro_steps,
                             policy, threshold=None, boundary="periodic", mc_samples=8, repeats=3, seed=0):
    """Median actual wall-clock runtime for a fixed workload, excluding research-only oracle diagnostics."""
    states=[np.asarray(s,dtype=np.float64) for s in initial_states]
    device=_device(model)
    if policy!="always_solver" and states:
        _=predict_numpy(model,states[0])
    _sync(device)
    elapsed=[]
    for r in range(int(repeats)):
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed+r)
        torch.manual_seed(seed+r)
        _sync(device); start=perf_counter()
        for j,u0 in enumerate(states):
            _runtime_one(model,u0,dx=dx,dt=dt,horizon=horizon,macro_steps=macro_steps,
                         policy=policy,threshold=threshold,boundary=boundary,mc_samples=mc_samples,
                         seed=seed+1000*r+100*j)
        _sync(device); elapsed.append(perf_counter()-start)
    return float(np.median(elapsed))


def benchmark_primitives(model, sample_state, *, dx, dt, horizon=1, mc_samples=8, repeats=50, seed=0):
    """Measure per-call wall-clock costs c_S, c_N, weak c_V, and uncertainty c_V."""
    u=np.asarray(sample_state,dtype=np.float64)
    device=_device(model)
    pred=predict_numpy(model,u)
    def median_time(fn):
        times=[]
        for _ in range(int(repeats)):
            _sync(device); start=perf_counter(); fn(); _sync(device); times.append(perf_counter()-start)
        return float(np.median(times))
    return {
        "godunov_macro_step_sec": median_time(lambda: advance(u,dx,horizon,dt=dt,boundary="periodic")),
        "neural_prediction_sec": median_time(lambda: predict_numpy(model,u)),
        "weak_residual_sec": median_time(lambda: weak_residual_score(u[None],pred[None],dx=dx,dt=dt*horizon)),
        "mc_dropout_uncertainty_sec": median_time(lambda: uncertainty_numpy(model,u,samples=mc_samples,seed=seed)),
    }
