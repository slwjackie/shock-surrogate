"""Autoregressive trust-or-fallback rollouts for the Burgers study.

Diagnostic rollouts intentionally evaluate the trusted Godunov proposal at
every macro step.  That makes the oracle local error observable and permits an
audit of the discrete contraction bound.  Runtime measurements use a separate
execution path and only call the trusted solver after a deployable gate rejects.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Iterable

import numpy as np
import torch

from .components import GodunovMacroStepper, TorchSurrogateAdapter, make_verifier
from .fine_reference import prolong_piecewise_constant, restrict_cell_average
from .interfaces import (
    AlwaysAcceptPolicy,
    AlwaysFallbackPolicy,
    Proposal,
    ThresholdPolicy,
    VerifierResult,
)
from .oracle_error import l1_error


POLICIES = (
    "always_solver",
    "always_neural",
    "oracle",
    "conservation",
    "residual",
    "uncertainty",
)


@dataclass
class RolloutResult:
    policy: str
    verifier_name: str | None
    threshold: float | None
    final_error: float
    accept_rate: float
    fallback_rate: float
    hard_rejection_rate: float
    accepted_eta_sum: float
    theorem_holds: bool
    max_theorem_violation: float
    cfl_assumptions_hold: bool
    max_algorithm_state_abs: float
    max_algorithm_cfl_number: float
    reference_steps: int
    final_physical_time: float
    trajectory_deviation: list[float]
    cumulative_accepted_eta: list[float]
    eta: list[float]
    scores: list[float]
    accepted: list[bool]
    decision_reasons: list[str]
    hard_rejections: list[bool]
    shock_strength: list[float]
    fine_reference_factor: int
    final_error_vs_fine: float
    classical_discretization_proxy: float
    trajectory_error_vs_fine: list[float]
    trajectory_classical_vs_fine: list[float]

    def to_dict(self) -> dict:
        return asdict(self)


def _device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def predict_numpy(model, state):
    """Backward-compatible convenience wrapper around the advice interface."""

    return TorchSurrogateAdapter(model).predict(state)


def uncertainty_numpy(model, state, samples=8, seed=None):
    """Backward-compatible convenience wrapper around the uncertainty verifier."""

    proposal = Proposal(
        state=np.asarray(state, dtype=np.float64),
        candidate=np.asarray(state, dtype=np.float64),
        horizon=1,
        elapsed_time=1.0,
        metadata={} if seed is None else {"seed": int(seed)},
    )
    verifier = make_verifier(
        "uncertainty", model=model, dx=1.0, mc_samples=samples, max_abs=None
    )
    return float(verifier.evaluate(proposal).score)


def compressive_shock_strength(state):
    """Largest positive left-to-right jump, a cheap periodic shock sensor."""

    values = np.asarray(state, dtype=np.float64)
    return float(np.max(np.maximum(values - np.roll(values, -1), 0.0)))


def _decision_components(policy: str, *, threshold: float | None, model, dx: float,
                         mc_samples: int, max_abs: float | None):
    if policy == "always_solver":
        return None, AlwaysFallbackPolicy()
    if policy == "always_neural":
        return None, AlwaysAcceptPolicy()
    if threshold is None:
        raise ValueError(f"policy={policy!r} requires a threshold.")
    return (
        make_verifier(
            policy, model=model, dx=dx, mc_samples=mc_samples, max_abs=max_abs
        ),
        ThresholdPolicy(float(threshold)),
    )


def rollout_diagnostics(
    model,
    u0,
    *,
    dx,
    dt,
    horizon=1,
    macro_steps=20,
    policy="always_neural",
    threshold=None,
    boundary="periodic",
    mc_samples=8,
    seed=0,
    theorem_tol=1e-10,
    max_abs=None,
    fine_reference_factor=1,
):
    """Run one trajectory and separate advice and discretization errors.

    ``final_error`` is the hybrid deviation from the same-grid all-Godunov
    trajectory. ``classical_discretization_proxy`` compares that all-Godunov
    trajectory with a restricted refined-grid Godunov trajectory.  The latter
    is a numerical proxy, not error against the exact entropy solution.
    """

    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}.")
    if int(horizon) < 1 or int(macro_steps) < 0:
        raise ValueError("horizon must be positive and macro_steps nonnegative.")
    fine_reference_factor = int(fine_reference_factor)
    if fine_reference_factor < 1:
        raise ValueError("fine_reference_factor must be at least one.")

    stepper = GodunovMacroStepper(dx, dt, int(horizon), boundary)
    advice = TorchSurrogateAdapter(model)
    verifier, decision_policy = _decision_components(
        policy,
        threshold=threshold,
        model=model,
        dx=dx,
        mc_samples=mc_samples,
        max_abs=max_abs,
    )

    current = np.asarray(u0, dtype=np.float64).copy()
    classical = current.copy()
    fine_state = prolong_piecewise_constant(current, fine_reference_factor)
    fine_stepper = GodunovMacroStepper(
        float(dx) / fine_reference_factor,
        float(dt) / fine_reference_factor,
        int(horizon) * fine_reference_factor,
        boundary,
    )

    cumulative_eta = 0.0
    deviations = [0.0]
    bounds = [0.0]
    hybrid_vs_fine = [0.0]
    classical_vs_fine = [0.0]
    etas: list[float] = []
    scores: list[float] = []
    accepted: list[bool] = []
    reasons: list[str] = []
    hard_rejections: list[bool] = []
    shocks: list[float] = []
    max_algorithm_state_abs = float(np.max(np.abs(current)))

    for macro_index in range(int(macro_steps)):
        candidate = advice.predict(current)
        # Research-only oracle call. Runtime benchmarking below does not make
        # this call on accepted deployable-gate steps.
        fallback = stepper.step(current)
        eta = float(l1_error(candidate[None], fallback[None], dx)[0])
        proposal = Proposal(
            state=current,
            candidate=candidate,
            horizon=int(horizon),
            elapsed_time=stepper.elapsed_time,
            reference_candidate=fallback,
            metadata={"seed": int(seed) + macro_index},
        )

        if verifier is None:
            result = VerifierResult(policy, float("nan"))
        else:
            result = verifier.evaluate(proposal)
        decision = decision_policy.decide(result)

        shocks.append(compressive_shock_strength(current))
        etas.append(eta)
        scores.append(float(result.score))
        accepted.append(bool(decision.accept))
        reasons.append(decision.reason)
        hard_rejections.append(bool(result.hard_failure))

        if decision.accept:
            cumulative_eta += eta
            current = candidate
        else:
            current = fallback
        max_algorithm_state_abs = max(
            max_algorithm_state_abs, float(np.max(np.abs(current)))
        )
        classical = stepper.step(classical)
        fine_state = fine_stepper.step(fine_state)
        fine_restricted = restrict_cell_average(fine_state, fine_reference_factor)

        deviations.append(float(l1_error(current[None], classical[None], dx)[0]))
        bounds.append(cumulative_eta)
        hybrid_vs_fine.append(float(l1_error(current[None], fine_restricted[None], dx)[0]))
        classical_vs_fine.append(
            float(l1_error(classical[None], fine_restricted[None], dx)[0])
        )

    violations = np.asarray(deviations) - np.asarray(bounds)
    accept_rate = float(np.mean(accepted)) if accepted else 0.0
    hard_rate = float(np.mean(hard_rejections)) if hard_rejections else 0.0
    return RolloutResult(
        policy=policy,
        verifier_name=None if verifier is None else verifier.name,
        threshold=None if threshold is None else float(threshold),
        final_error=float(deviations[-1]),
        accept_rate=accept_rate,
        fallback_rate=float(1.0 - accept_rate),
        hard_rejection_rate=hard_rate,
        accepted_eta_sum=float(cumulative_eta),
        theorem_holds=bool(np.max(violations) <= theorem_tol),
        max_theorem_violation=float(np.max(violations)),
        cfl_assumptions_hold=bool(
            float(dt) * max_algorithm_state_abs / float(dx) <= 1.0 + 1e-12
        ),
        max_algorithm_state_abs=max_algorithm_state_abs,
        max_algorithm_cfl_number=float(dt) * max_algorithm_state_abs / float(dx),
        reference_steps=int(horizon) * int(macro_steps),
        final_physical_time=float(dt) * int(horizon) * int(macro_steps),
        trajectory_deviation=[float(value) for value in deviations],
        cumulative_accepted_eta=[float(value) for value in bounds],
        eta=[float(value) for value in etas],
        scores=[float(value) for value in scores],
        accepted=[bool(value) for value in accepted],
        decision_reasons=reasons,
        hard_rejections=[bool(value) for value in hard_rejections],
        shock_strength=[float(value) for value in shocks],
        fine_reference_factor=fine_reference_factor,
        final_error_vs_fine=float(hybrid_vs_fine[-1]),
        classical_discretization_proxy=float(classical_vs_fine[-1]),
        trajectory_error_vs_fine=hybrid_vs_fine,
        trajectory_classical_vs_fine=classical_vs_fine,
    )


def _runtime_one(
    model,
    u0,
    *,
    dx,
    dt,
    horizon,
    macro_steps,
    policy,
    threshold,
    boundary,
    mc_samples,
    seed,
    max_abs,
):
    current = np.asarray(u0, dtype=np.float64).copy()
    stepper = GodunovMacroStepper(dx, dt, int(horizon), boundary)
    advice = TorchSurrogateAdapter(model)
    verifier, decision_policy = _decision_components(
        policy,
        threshold=threshold,
        model=model,
        dx=dx,
        mc_samples=mc_samples,
        max_abs=max_abs,
    )

    for macro_index in range(int(macro_steps)):
        if policy == "always_solver":
            current = stepper.step(current)
            continue

        candidate = advice.predict(current)
        if policy == "always_neural":
            current = candidate
            continue

        fallback = stepper.step(current) if policy == "oracle" else None
        proposal = Proposal(
            state=current,
            candidate=candidate,
            horizon=int(horizon),
            elapsed_time=stepper.elapsed_time,
            reference_candidate=fallback,
            metadata={"seed": int(seed) + macro_index},
        )
        result = verifier.evaluate(proposal)
        decision = decision_policy.decide(result)
        if decision.accept:
            current = candidate
        elif fallback is not None:
            current = fallback
        else:
            current = stepper.step(current)
    return current


def benchmark_policy_runtime(
    model,
    initial_states: Iterable[np.ndarray],
    *,
    dx,
    dt,
    horizon,
    macro_steps,
    policy,
    threshold=None,
    boundary="periodic",
    mc_samples=8,
    repeats=3,
    seed=0,
    max_abs=None,
):
    """Median wall time for a fixed workload, excluding diagnostic oracle calls."""

    states = [np.asarray(state, dtype=np.float64) for state in initial_states]
    device = _device(model)
    if policy != "always_solver" and states:
        predict_numpy(model, states[0])
    _sync(device)
    elapsed = []
    for repeat in range(int(repeats)):
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + repeat)
        torch.manual_seed(seed + repeat)
        _sync(device)
        start = perf_counter()
        for case_index, initial in enumerate(states):
            _runtime_one(
                model,
                initial,
                dx=dx,
                dt=dt,
                horizon=horizon,
                macro_steps=macro_steps,
                policy=policy,
                threshold=threshold,
                boundary=boundary,
                mc_samples=mc_samples,
                seed=seed + 1000 * repeat + 100 * case_index,
                max_abs=max_abs,
            )
        _sync(device)
        elapsed.append(perf_counter() - start)
    return float(np.median(elapsed))


def benchmark_primitives(
    model,
    sample_state,
    *,
    dx,
    dt,
    horizon=1,
    mc_samples=8,
    repeats=50,
    seed=0,
    max_abs=None,
):
    """Measure primitive costs and the gate's minimum break-even accept rate."""

    state = np.asarray(sample_state, dtype=np.float64)
    device = _device(model)
    prediction = predict_numpy(model, state)
    proposal = Proposal(
        state=state,
        candidate=prediction,
        horizon=int(horizon),
        elapsed_time=float(dt) * int(horizon),
        metadata={"seed": int(seed)},
    )

    def median_time(function):
        times = []
        for _ in range(int(repeats)):
            _sync(device)
            start = perf_counter()
            function()
            _sync(device)
            times.append(perf_counter() - start)
        return float(np.median(times))

    solver_cost = median_time(
        lambda: GodunovMacroStepper(dx, dt, int(horizon)).step(state)
    )
    neural_cost = median_time(lambda: predict_numpy(model, state))
    verifier_cost = {
        name: median_time(
            lambda verifier=make_verifier(
                name,
                model=model,
                dx=dx,
                mc_samples=mc_samples,
                max_abs=max_abs,
            ): verifier.evaluate(proposal)
        )
        for name in ("conservation", "residual", "uncertainty")
    }
    break_even = {
        name: float((neural_cost + cost) / max(solver_cost, 1e-15))
        for name, cost in verifier_cost.items()
    }
    return {
        "godunov_macro_step_sec": solver_cost,
        "neural_prediction_sec": neural_cost,
        "conservation_sec": verifier_cost["conservation"],
        "weak_residual_sec": verifier_cost["residual"],
        "mc_dropout_uncertainty_sec": verifier_cost["uncertainty"],
        "minimum_accept_rate_for_speedup": break_even,
        "break_even_interpretation": (
            "Expected hybrid cost is c_N+c_V+(1-a)c_S; speedup requires "
            "a>(c_N+c_V)/c_S. A value >=1 means no per-sample break-even on "
            "this workload without batching, a larger horizon, or lower overhead."
        ),
    }
