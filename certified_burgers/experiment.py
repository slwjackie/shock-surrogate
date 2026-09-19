"""Reproducible Burgers/Godunov verifier experiment.

This runner implements the professor's six-task workflow while fixing four
pieces needed for later extensions:

* comparison horizons share a fixed number of fine Godunov steps and therefore
  the same final physical time;
* advice error and a refined-grid discretization-error proxy are reported
  separately;
* cheap scores are audited for dangerous false accepts and controlled blind
  spots, not just correlation;
* solver, advice, verifier, and decision policy use replaceable interfaces.

Cheap scores and calibration envelopes remain empirical diagnostics.  They are
not described as mathematical certificates.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .counterexamples import audit_verifier_counterexamples
from .fine_reference import fine_godunov_reference
from .godunov import advance
from .initial_conditions import cell_centers
from .oracle_error import l1_error
from .rollout import (
    POLICIES,
    benchmark_policy_runtime,
    benchmark_primitives,
    compressive_shock_strength,
    rollout_diagnostics,
)
from .splits import make_data_splits
from .stability import stability_sweep
from .surrogate import build_surrogate
from .validation import godunov_validation_suite
from .verifiers import conservation_defect, mc_dropout_uncertainty, weak_residual_score


CHEAP_VERIFIERS = ("conservation", "residual", "uncertainty")
SCORE_FIELD = {
    "oracle": "eta",
    "conservation": "conservation",
    "residual": "weak_residual",
    "uncertainty": "uncertainty",
}


@dataclass
class ExperimentConfig:
    n_cells: int = 128
    train_samples: int = 768
    calib_samples: int = 192
    test_samples: int = 192
    epochs: int = 60
    batch_size: int = 64
    lr: float = 2e-3
    horizon: int = 1
    cfl: float = 0.8
    max_abs_global: float = 4.0
    surrogate_kind: str = "state"
    seed: int = 0
    mc_samples: int = 8
    rollout_cases: int = 8
    reference_steps: int = 24
    fine_reference_factor: int = 4
    sweep_points: int = 9
    gate_accept_quantile: float = 0.8
    unsafe_eta_quantile: float = 0.9
    empirical_coverage: float = 0.95
    runtime_repeats: int = 3
    primitive_repeats: int = 50
    make_plots: bool = True

    @property
    def macro_steps(self) -> int:
        if int(self.horizon) < 1:
            raise ValueError("horizon must be positive.")
        if int(self.reference_steps) < 1:
            raise ValueError("reference_steps must be positive.")
        if int(self.reference_steps) % int(self.horizon):
            raise ValueError(
                "reference_steps must be divisible by horizon so horizon studies "
                "end at exactly the same physical time."
            )
        return int(self.reference_steps) // int(self.horizon)

    def validate(self) -> None:
        _ = self.macro_steps
        if self.n_cells < 8:
            raise ValueError("n_cells must be at least 8.")
        for name in ("train_samples", "calib_samples", "test_samples", "rollout_cases"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive.")
        if not (0.0 < self.cfl <= 1.0):
            raise ValueError("cfl must be in (0,1].")
        if self.max_abs_global < 1.8:
            raise ValueError(
                "max_abs_global must cover the generated OOD envelope (at least 1.8)."
            )
        if self.lr <= 0.0:
            raise ValueError("lr must be positive.")
        if self.surrogate_kind not in {"state", "flux"}:
            raise ValueError("surrogate_kind must be 'state' or 'flux'.")
        if self.fine_reference_factor < 1:
            raise ValueError("fine_reference_factor must be at least one.")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive.")
        if self.mc_samples < 2:
            raise ValueError("mc_samples must be at least two.")
        if self.sweep_points < 2:
            raise ValueError("sweep_points must be at least two.")
        if self.runtime_repeats < 1 or self.primitive_repeats < 1:
            raise ValueError("runtime repeat counts must be positive.")
        for name in ("gate_accept_quantile", "unsafe_eta_quantile", "empirical_coverage"):
            if not (0.0 < float(getattr(self, name)) < 1.0):
                raise ValueError(f"{name} must be in (0,1).")


def _reference_batch(states, dx, dt, horizon):
    reference, _ = advance(states, dx, int(horizon), dt=dt, boundary="periodic")
    return reference


def _pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if np.std(x) < 1e-14 or np.std(y) < 1e-14:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _spearman(x, y):
    return _pearson(_average_ranks(np.asarray(x)), _average_ranks(np.asarray(y)))


def _rank_capture(score, eta, frac=0.1):
    score = np.asarray(score)
    eta = np.asarray(eta)
    k = max(1, int(round(float(frac) * len(eta))))
    bad = set(np.argsort(eta)[-k:])
    flagged = set(np.argsort(score)[-k:])
    return float(len(bad & flagged) / k)


def _empirical_scale(score, eta, coverage=0.95):
    """Calibration-only multiplier C for empirical eta <= C*q coverage."""

    score = np.asarray(score, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    positive = score[score > 0]
    floor = max(1e-12, 1e-6 * float(np.median(positive)) if positive.size else 1e-12)
    ratio = eta / np.maximum(score, floor)
    return float(np.quantile(ratio, float(coverage))), float(floor)


def _shock_mask(shock_strength, threshold):
    values = np.asarray(shock_strength, dtype=np.float64)
    return (values >= float(threshold)) & (values > 1e-12)


def _predict_batch(model, states, device):
    tensor = torch.as_tensor(states, dtype=torch.float32, device=device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        prediction = model(tensor).detach().cpu().numpy()
    model.train(was_training)
    return prediction


def _audit_batch(
    model,
    states,
    *,
    dx,
    dt,
    horizon,
    mc_samples,
    device,
    seed,
    fine_reference_factor,
):
    reference = _reference_batch(states, dx, dt, horizon)
    prediction = _predict_batch(model, states, device)
    refined_reference = fine_godunov_reference(
        states,
        dx=dx,
        dt=dt,
        coarse_steps=horizon,
        factor=fine_reference_factor,
    )

    eta = l1_error(prediction, reference, dx)
    solver_discretization_proxy = l1_error(reference, refined_reference, dx)
    advice_vs_refined_proxy = l1_error(prediction, refined_reference, dx)
    conservation = conservation_defect(states, prediction, dx)
    weak = weak_residual_score(states, prediction, dx=dx, dt=dt * horizon)

    tensor = torch.as_tensor(states, dtype=torch.float32, device=device)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    uncertainty = (
        mc_dropout_uncertainty(model, tensor, samples=mc_samples).detach().cpu().numpy()
    )
    shock = np.asarray(
        [compressive_shock_strength(state) for state in states], dtype=np.float64
    )
    return {
        "eta": eta,
        "conservation": conservation,
        "weak_residual": weak,
        "uncertainty": uncertainty,
        "shock_strength": shock,
        "solver_discretization_proxy": solver_discretization_proxy,
        "advice_vs_refined_proxy": advice_vs_refined_proxy,
    }


def _risk_metrics(score, eta, *, score_threshold, unsafe_eta_threshold):
    score = np.asarray(score, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    accepted = score <= float(score_threshold)
    unsafe = eta > float(unsafe_eta_threshold)
    accepted_count = int(np.sum(accepted))
    unsafe_count = int(np.sum(unsafe))
    dangerous = accepted & unsafe
    return {
        "score_threshold_from_calibration": float(score_threshold),
        "unsafe_eta_threshold_from_calibration": float(unsafe_eta_threshold),
        "acceptance_rate": float(np.mean(accepted)),
        "false_acceptance_rate_given_accept": (
            float(np.sum(dangerous) / accepted_count) if accepted_count else None
        ),
        "unsafe_acceptance_fraction_all_cases": float(np.mean(dangerous)),
        "unsafe_case_miss_rate": (
            float(np.sum(dangerous) / unsafe_count) if unsafe_count else None
        ),
        "unsafe_case_count": unsafe_count,
    }


def _coverage_metrics(eta, scaled_bound, mask=None):
    eta = np.asarray(eta, dtype=np.float64)
    bound = np.asarray(scaled_bound, dtype=np.float64)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        eta = eta[mask]
        bound = bound[mask]
    if not len(eta):
        return None
    return float(np.mean(eta <= bound))


def _audit_summary(
    audit,
    *,
    shock_threshold,
    selected_thresholds=None,
    unsafe_eta_threshold=None,
    calibration_scales=None,
):
    eta = np.asarray(audit["eta"])
    shock_mask = _shock_mask(audit["shock_strength"], shock_threshold)
    nonshock_mask = ~shock_mask
    discretization = np.asarray(audit["solver_discretization_proxy"])
    advice_refined = np.asarray(audit["advice_vs_refined_proxy"])
    result = {
        "count": int(len(eta)),
        "mean_eta_same_grid_advice_error": float(np.mean(eta)),
        "median_eta_same_grid_advice_error": float(np.median(eta)),
        "p95_eta_same_grid_advice_error": float(np.quantile(eta, 0.95)),
        "mean_solver_discretization_proxy": float(np.mean(discretization)),
        "p95_solver_discretization_proxy": float(np.quantile(discretization, 0.95)),
        "mean_advice_vs_refined_proxy": float(np.mean(advice_refined)),
        "p95_advice_vs_refined_proxy": float(np.quantile(advice_refined, 0.95)),
        "mean_triangle_upper_bound": float(np.mean(eta + discretization)),
        "triangle_inequality_violations": int(
            np.sum(advice_refined > eta + discretization + 1e-12)
        ),
        "shock_threshold_from_calibration": float(shock_threshold),
        "shock_heavy_fraction": float(np.mean(shock_mask)),
        "shock_heavy_mean_eta": (
            float(np.mean(eta[shock_mask])) if np.any(shock_mask) else None
        ),
        "verifiers": {},
    }

    for name in CHEAP_VERIFIERS:
        score = np.asarray(audit[SCORE_FIELD[name]])
        row = {
            "pearson_with_eta": _pearson(score, eta),
            "spearman_with_eta": _spearman(score, eta),
            "top10_error_capture": _rank_capture(score, eta, 0.10),
            "mean_score": float(np.mean(score)),
            "raw_score_upper_coverage": float(np.mean(eta <= score)),
            "shock_heavy_pearson": (
                _pearson(score[shock_mask], eta[shock_mask])
                if np.sum(shock_mask) >= 3
                else None
            ),
        }
        if selected_thresholds is not None and unsafe_eta_threshold is not None:
            row["risk_at_selected_threshold"] = _risk_metrics(
                score,
                eta,
                score_threshold=selected_thresholds[name],
                unsafe_eta_threshold=unsafe_eta_threshold,
            )
        if calibration_scales and name in calibration_scales:
            scale = calibration_scales[name]["scale"]
            floor = calibration_scales[name]["floor"]
            scaled_bound = scale * np.maximum(score, floor)
            row["empirical_bound_diagnostic"] = {
                "scale_from_calibration": float(scale),
                "coverage_all": _coverage_metrics(eta, scaled_bound),
                "coverage_shock_heavy": _coverage_metrics(
                    eta, scaled_bound, shock_mask
                ),
                "coverage_nonshock": _coverage_metrics(
                    eta, scaled_bound, nonshock_mask
                ),
                "mean_bound_over_eta": float(
                    np.mean(scaled_bound / np.maximum(eta, 1e-12))
                ),
            }
        result["verifiers"][name] = row
    return result


def _threshold_grid(values, points):
    values = np.asarray(values, dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, max(2, int(points)))
    return [float(value) for value in np.unique(np.quantile(values, quantiles))]


def _aggregate_rollouts(results):
    if not results:
        raise ValueError("Need at least one rollout.")
    return {
        "final_error_vs_same_grid_solver": float(np.mean([r.final_error for r in results])),
        "p95_final_error_vs_same_grid_solver": float(
            np.quantile([r.final_error for r in results], 0.95)
        ),
        # Backward-compatible alias used by existing plots.
        "final_error": float(np.mean([r.final_error for r in results])),
        "final_error_vs_refined_solver_proxy": float(
            np.mean([r.final_error_vs_fine for r in results])
        ),
        "p95_final_error_vs_refined_solver_proxy": float(
            np.quantile([r.final_error_vs_fine for r in results], 0.95)
        ),
        "same_grid_solver_discretization_proxy": float(
            np.mean([r.classical_discretization_proxy for r in results])
        ),
        "accept_rate": float(np.mean([r.accept_rate for r in results])),
        "fallback_rate": float(np.mean([r.fallback_rate for r in results])),
        "hard_rejection_rate": float(
            np.mean([r.hard_rejection_rate for r in results])
        ),
        "mean_accepted_eta_sum": float(
            np.mean([r.accepted_eta_sum for r in results])
        ),
        "theorem_holds_all": bool(all(r.theorem_holds for r in results)),
        "discrete_contraction_bound_audit_passed_all": bool(
            all(r.theorem_holds for r in results)
        ),
        "theorem_assumptions_hold_all": bool(
            all(r.cfl_assumptions_hold for r in results)
        ),
        "max_algorithm_cfl_number": float(
            max(r.max_algorithm_cfl_number for r in results)
        ),
        "max_theorem_violation": float(
            max(r.max_theorem_violation for r in results)
        ),
        "reference_steps": int(results[0].reference_steps),
        "final_physical_time": float(results[0].final_physical_time),
    }


def _run_rollout_set(model, states, *, dx, dt, config, policy, threshold):
    return [
        rollout_diagnostics(
            model,
            state,
            dx=dx,
            dt=dt,
            horizon=config.horizon,
            macro_steps=config.macro_steps,
            policy=policy,
            threshold=threshold,
            mc_samples=config.mc_samples,
            seed=config.seed + 10000 + index,
            max_abs=config.max_abs_global,
            fine_reference_factor=config.fine_reference_factor,
        )
        for index, state in enumerate(states)
    ]


def _policy_sweep(model, states, thresholds, *, dx, dt, config, policy):
    rows = []
    for threshold in thresholds:
        aggregate = _aggregate_rollouts(
            _run_rollout_set(
                model,
                states,
                dx=dx,
                dt=dt,
                config=config,
                policy=policy,
                threshold=threshold,
            )
        )
        rows.append({"threshold": float(threshold), **aggregate})
    return rows


def _train_model(config, train_x, train_y, device):
    model = build_surrogate(
        config.surrogate_kind,
        horizon=config.horizon,
        dt_over_dx=config.cfl / config.max_abs_global,
        channels=1,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    loss_function = nn.MSELoss()
    dataset = TensorDataset(
        torch.tensor(train_x, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    losses = []
    model.train()
    for _ in range(config.epochs):
        running = 0.0
        count = 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(inputs), targets)
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu()) * len(inputs)
            count += len(inputs)
        losses.append(running / max(count, 1))
    return model, losses


def _plot_results(out_path, id_audit, ood_audit, shock_threshold, sweeps, baselines,
                  representative):
    from .plots import (
        plot_error_fallback,
        plot_error_speedup,
        plot_theorem_bound,
        plot_verifier_scatter,
    )

    base = Path(out_path).with_suffix("") if out_path is not None else Path(
        "outputs/certified_burgers_poc"
    )
    plot_dir = base.parent / (base.name + "_plots")
    paths = []
    id_shock = _shock_mask(id_audit["shock_strength"], shock_threshold)
    ood_shock = _shock_mask(ood_audit["shock_strength"], shock_threshold)
    for name in CHEAP_VERIFIERS:
        field = SCORE_FIELD[name]
        paths.append(
            plot_verifier_scatter(
                plot_dir,
                name,
                id_audit["eta"],
                id_audit[field],
                ood_audit["eta"],
                ood_audit[field],
                id_shock,
                ood_shock,
            )
        )
    for split in ("id", "ood"):
        paths.append(plot_error_fallback(plot_dir, sweeps[split], tag=split))
        paths.append(plot_error_speedup(plot_dir, baselines[split], tag=split))
        paths.append(
            plot_theorem_bound(
                plot_dir,
                representative[split],
                name=f"residual_{split}",
            )
        )
    return paths


def run(config: ExperimentConfig, out_path=None):
    config.validate()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    _, dx = cell_centers(config.n_cells)
    # Fixed across every sample and policy, based on the declared global state
    # envelope rather than per-sample adaptive time steps.
    dt = config.cfl * dx / config.max_abs_global
    splits, split_manifest = make_data_splits(
        n_cells=config.n_cells,
        train_samples=config.train_samples,
        calib_samples=config.calib_samples,
        test_samples=config.test_samples,
        rollout_cases=config.rollout_cases,
        seed=config.seed,
    )

    train_targets = _reference_batch(splits["train"], dx, dt, config.horizon)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, losses = _train_model(config, splits["train"], train_targets, device)

    audit_kwargs = {
        "dx": dx,
        "dt": dt,
        "horizon": config.horizon,
        "mc_samples": config.mc_samples,
        "device": device,
        "fine_reference_factor": config.fine_reference_factor,
    }
    calibration = _audit_batch(
        model, splits["calibration"], seed=config.seed + 40, **audit_kwargs
    )
    id_audit = _audit_batch(
        model, splits["test_id"], seed=config.seed + 41, **audit_kwargs
    )
    ood_audit = _audit_batch(
        model, splits["test_ood"], seed=config.seed + 42, **audit_kwargs
    )

    calibration_scales = {}
    for name in CHEAP_VERIFIERS:
        field = SCORE_FIELD[name]
        scale, floor = _empirical_scale(
            calibration[field], calibration["eta"], config.empirical_coverage
        )
        calibration_scales[name] = {
            "scale": scale,
            "floor": floor,
            "target_coverage": config.empirical_coverage,
        }

    selected_thresholds = {
        name: float(
            np.quantile(calibration[field], config.gate_accept_quantile)
        )
        for name, field in SCORE_FIELD.items()
    }
    unsafe_eta_threshold = float(
        np.quantile(calibration["eta"], config.unsafe_eta_quantile)
    )
    shock_threshold = float(np.quantile(calibration["shock_strength"], 0.75))
    threshold_grids = {
        name: _threshold_grid(calibration[field], config.sweep_points)
        for name, field in SCORE_FIELD.items()
    }

    rollout_sets = {
        "id": splits["rollout_id"],
        "ood": splits["rollout_ood"],
    }
    baselines = {}
    sweeps = {}
    representative = {}
    for split, states in rollout_sets.items():
        baselines[split] = {}
        diagnostic_cache = {}
        for policy in POLICIES:
            threshold = selected_thresholds.get(policy)
            rollouts = _run_rollout_set(
                model,
                states,
                dx=dx,
                dt=dt,
                config=config,
                policy=policy,
                threshold=threshold,
            )
            diagnostic_cache[policy] = rollouts
            baselines[split][policy] = _aggregate_rollouts(rollouts)

        solver_runtime = benchmark_policy_runtime(
            model,
            states,
            dx=dx,
            dt=dt,
            horizon=config.horizon,
            macro_steps=config.macro_steps,
            policy="always_solver",
            repeats=config.runtime_repeats,
            mc_samples=config.mc_samples,
            seed=config.seed + 70,
            max_abs=config.max_abs_global,
        )
        for policy in POLICIES:
            threshold = selected_thresholds.get(policy)
            runtime = (
                solver_runtime
                if policy == "always_solver"
                else benchmark_policy_runtime(
                    model,
                    states,
                    dx=dx,
                    dt=dt,
                    horizon=config.horizon,
                    macro_steps=config.macro_steps,
                    policy=policy,
                    threshold=threshold,
                    repeats=config.runtime_repeats,
                    mc_samples=config.mc_samples,
                    seed=config.seed + 70,
                    max_abs=config.max_abs_global,
                )
            )
            baselines[split][policy]["actual_runtime_sec"] = float(runtime)
            baselines[split][policy]["speedup_vs_solver"] = float(
                solver_runtime / max(runtime, 1e-15)
            )

        sweeps[split] = {
            policy: _policy_sweep(
                model,
                states,
                threshold_grids[policy],
                dx=dx,
                dt=dt,
                config=config,
                policy=policy,
            )
            for policy in SCORE_FIELD
        }
        representative[split] = diagnostic_cache["residual"][0]

    primitives = benchmark_primitives(
        model,
        splits["test_id"][0],
        dx=dx,
        dt=dt,
        horizon=config.horizon,
        mc_samples=config.mc_samples,
        repeats=config.primitive_repeats,
        seed=config.seed + 80,
        max_abs=config.max_abs_global,
    )
    stability = stability_sweep(pairs=128, seed=config.seed + 90)
    validation = godunov_validation_suite(
        n_cells=max(64, config.n_cells), cfl=config.cfl, seed=config.seed + 91
    )

    probe_current = splits["test_id"][0]
    probe_reference = _reference_batch(
        probe_current[None], dx, dt, config.horizon
    )[0]
    counterexamples = audit_verifier_counterexamples(
        probe_current,
        probe_reference,
        dx=dx,
        elapsed_time=dt * config.horizon,
        thresholds={
            "conservation": selected_thresholds["conservation"],
            "residual": selected_thresholds["residual"],
        },
    )

    plot_paths = []
    if config.make_plots:
        plot_paths = _plot_results(
            out_path,
            id_audit,
            ood_audit,
            shock_threshold,
            sweeps,
            baselines,
            representative,
        )

    summary_kwargs = {
        "shock_threshold": shock_threshold,
        "selected_thresholds": selected_thresholds,
        "unsafe_eta_threshold": unsafe_eta_threshold,
    }
    payload = {
        "schema_version": "2.0",
        "config": asdict(config),
        "comparison_contract": {
            "domain": [0.0, 1.0],
            "boundary": "periodic",
            "dx": float(dx),
            "dt_per_reference_step": float(dt),
            "declared_state_envelope_max_abs": float(config.max_abs_global),
            "nominal_cfl_at_state_envelope": float(config.cfl),
            "observed_initial_state_max_abs_by_split": {
                name: float(np.max(np.abs(values)))
                for name, values in splits.items()
            },
            "observed_initial_cfl_by_split": {
                name: float(dt * np.max(np.abs(values)) / dx)
                for name, values in splits.items()
            },
            "horizon_reference_steps_per_neural_call": int(config.horizon),
            "macro_steps": int(config.macro_steps),
            "total_reference_steps": int(config.reference_steps),
            "final_physical_time": float(dt * config.reference_steps),
            "same_final_time_across_horizons_rule": (
                "macro_steps=reference_steps/horizon; non-divisible horizons are rejected"
            ),
            "threshold_source": "calibration split only",
            "unsafe_definition": (
                f"eta above calibration quantile {config.unsafe_eta_quantile}"
            ),
            "runtime_protocol": {
                "same_initial_states_and_final_time": True,
                "median_repeats": int(config.runtime_repeats),
                "diagnostic_oracle_solver_calls_excluded_for_deployable_gates": True,
            },
        },
        "data_split_audit": split_manifest,
        "software_environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "device": str(device),
        "training": {
            "surrogate_kind": config.surrogate_kind,
            "architecture": model.architecture_summary(),
            "first_epoch_mse": float(losses[0]),
            "final_epoch_mse": float(losses[-1]),
        },
        "error_definitions": {
            "oracle_local_advice_error_eta": (
                "L1(P_H(v_t), S_h^H(v_t)) on the same algorithm state and grid"
            ),
            "hybrid_deviation_same_grid": (
                "L1(v_t, u_t^S_h), used by the discrete contraction-bound audit"
            ),
            "solver_discretization_proxy": (
                "L1(S_h^H(v), R S_{h/r}^{Hr}(prolong(v))); a refined-Godunov proxy, "
                "not exact PDE error"
            ),
            "advice_vs_refined_proxy": (
                "L1(P_H(v), R S_{h/r}^{Hr}(prolong(v)))"
            ),
            "triangle_decomposition": (
                "advice-vs-refined <= eta + solver-discretization-proxy"
            ),
        },
        "one_step_audit": {
            "meaning": (
                "One policy decision / one surrogate call; this spans H="
                f"{config.horizon} trusted reference steps."
            ),
            "horizon_reference_steps": int(config.horizon),
            "calibration": _audit_summary(
                calibration,
                **summary_kwargs,
                calibration_scales=calibration_scales,
            ),
            "id": _audit_summary(
                id_audit,
                **summary_kwargs,
                calibration_scales=calibration_scales,
            ),
            "ood": _audit_summary(
                ood_audit,
                **summary_kwargs,
                calibration_scales=calibration_scales,
            ),
            "empirical_scales": calibration_scales,
            "unsafe_eta_threshold": unsafe_eta_threshold,
        },
        "selected_gate_thresholds": selected_thresholds,
        "baselines": baselines,
        "threshold_sweeps": sweeps,
        "controlled_verifier_failure_probes": counterexamples,
        "primitive_wall_clock": primitives,
        "godunov_validation": validation,
        "godunov_l1_stability_sweep": stability,
        "plots": plot_paths,
        "claim_boundaries": [
            "eta is an oracle diagnostic because it evaluates S_h^H at the algorithm state.",
            "Conservation, weak-residual, and MC-dropout scores are empirical proxies, not proven certificates.",
            "The calibrated multiplicative envelope measures held-out coverage only; it is not a theorem.",
            "The contraction audit bounds deviation from the declared same-grid Godunov trajectory, not error to the exact entropy solution.",
            "The contraction theorem is applicable only on rollouts whose declared fixed-step CFL assumptions remain satisfied; this is reported separately from the empirical inequality check.",
            "The refined-grid comparison is a discretization-error proxy and is reported separately from advice error.",
            "Oracle Gate is non-deployable because revealing eta requires the trusted solver call.",
            "Controlled corruptions demonstrate score blind spots but do not estimate trained-model failure frequencies.",
        ],
    }
    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_cells", type=int, default=128)
    parser.add_argument("--train_samples", type=int, default=768)
    parser.add_argument("--calib_samples", type=int, default=192)
    parser.add_argument("--test_samples", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--cfl", type=float, default=0.8)
    parser.add_argument("--max_abs_global", type=float, default=4.0)
    parser.add_argument("--surrogate", choices=("state", "flux"), default="state")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mc_samples", type=int, default=8)
    parser.add_argument("--rollout_cases", type=int, default=8)
    parser.add_argument(
        "--reference_steps",
        "--rollout_steps",
        dest="reference_steps",
        type=int,
        default=24,
        help=(
            "Total fine Godunov steps to the final time. --rollout_steps is a "
            "deprecated alias with the new fair-comparison semantics."
        ),
    )
    parser.add_argument("--fine_reference_factor", type=int, default=4)
    parser.add_argument("--sweep_points", type=int, default=9)
    parser.add_argument("--gate_accept_quantile", type=float, default=0.8)
    parser.add_argument("--unsafe_eta_quantile", type=float, default=0.9)
    parser.add_argument("--empirical_coverage", type=float, default=0.95)
    parser.add_argument("--runtime_repeats", type=int, default=3)
    parser.add_argument("--primitive_repeats", type=int, default=50)
    parser.add_argument("--out", default="outputs/certified_burgers_poc.json")
    parser.add_argument("--no_plots", action="store_true")
    arguments = parser.parse_args()
    config = ExperimentConfig(
        n_cells=arguments.n_cells,
        train_samples=arguments.train_samples,
        calib_samples=arguments.calib_samples,
        test_samples=arguments.test_samples,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        lr=arguments.lr,
        horizon=arguments.horizon,
        cfl=arguments.cfl,
        max_abs_global=arguments.max_abs_global,
        surrogate_kind=arguments.surrogate,
        seed=arguments.seed,
        mc_samples=arguments.mc_samples,
        rollout_cases=arguments.rollout_cases,
        reference_steps=arguments.reference_steps,
        fine_reference_factor=arguments.fine_reference_factor,
        sweep_points=arguments.sweep_points,
        gate_accept_quantile=arguments.gate_accept_quantile,
        unsafe_eta_quantile=arguments.unsafe_eta_quantile,
        empirical_coverage=arguments.empirical_coverage,
        runtime_repeats=arguments.runtime_repeats,
        primitive_repeats=arguments.primitive_repeats,
        make_plots=not arguments.no_plots,
    )
    print(json.dumps(run(config, arguments.out), indent=2))


if __name__ == "__main__":
    main()
