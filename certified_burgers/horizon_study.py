"""Fair one-step versus multi-step surrogate comparison.

Every horizon sees identical role-separated initial states and is evaluated at
the same physical final time.  A horizon H uses ``reference_steps / H`` neural
decisions, so non-divisible choices are rejected rather than silently rounded.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from .experiment import ExperimentConfig, run


def _compact_policy_row(payload: dict, split: str, policy: str) -> dict:
    row = payload["baselines"][split][policy]
    return {
        "final_error_vs_same_grid_solver": row[
            "final_error_vs_same_grid_solver"
        ],
        "final_error_vs_refined_solver_proxy": row[
            "final_error_vs_refined_solver_proxy"
        ],
        "fallback_rate": row["fallback_rate"],
        "hard_rejection_rate": row["hard_rejection_rate"],
        "theorem_assumptions_hold_all": row["theorem_assumptions_hold_all"],
        "max_algorithm_cfl_number": row["max_algorithm_cfl_number"],
        "actual_runtime_sec": row["actual_runtime_sec"],
        "speedup_vs_solver": row["speedup_vs_solver"],
    }


def run_horizon_study(
    base_config: ExperimentConfig,
    *,
    horizons=(1, 4),
    out_dir="outputs/certified_burgers_horizons",
) -> dict:
    horizons = tuple(int(value) for value in horizons)
    if not horizons or any(value < 1 for value in horizons):
        raise ValueError("Provide at least one positive horizon.")
    if len(set(horizons)) != len(horizons):
        raise ValueError("Horizons must be unique.")
    incompatible = [
        value for value in horizons if base_config.reference_steps % value
    ]
    if incompatible:
        raise ValueError(
            f"reference_steps={base_config.reference_steps} is not divisible by "
            f"horizons {incompatible}."
        )

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    payloads = {}
    rows = []
    split_hashes = None
    for horizon in horizons:
        config = replace(base_config, horizon=horizon)
        output_path = destination / f"horizon_{horizon}.json"
        payload = run(config, output_path)
        payloads[horizon] = payload

        current_hashes = {
            name: row["sha256"]
            for name, row in payload["data_split_audit"]["roles"].items()
        }
        if split_hashes is None:
            split_hashes = current_hashes
        elif current_hashes != split_hashes:
            raise RuntimeError("Horizon runs did not use identical data splits.")

        rows.append(
            {
                "horizon": horizon,
                "macro_steps": payload["comparison_contract"]["macro_steps"],
                "reference_steps": payload["comparison_contract"][
                    "total_reference_steps"
                ],
                "final_physical_time": payload["comparison_contract"][
                    "final_physical_time"
                ],
                "training_final_mse": payload["training"]["final_epoch_mse"],
                "id": {
                    policy: _compact_policy_row(payload, "id", policy)
                    for policy in payload["baselines"]["id"]
                },
                "ood": {
                    policy: _compact_policy_row(payload, "ood", policy)
                    for policy in payload["baselines"]["ood"]
                },
            }
        )

    final_times = {round(row["final_physical_time"], 15) for row in rows}
    reference_counts = {row["reference_steps"] for row in rows}
    summary = {
        "schema_version": "1.0",
        "purpose": (
            "Fair one-step/multi-step comparison at fixed data roles, grid, "
            "reference-step count, and final physical time."
        ),
        "base_config": asdict(base_config),
        "horizons": list(horizons),
        "fairness_checks": {
            "identical_split_hashes": True,
            "identical_reference_step_count": len(reference_counts) == 1,
            "identical_final_physical_time": len(final_times) == 1,
            "final_physical_time_values": sorted(final_times),
        },
        "rows": rows,
        "individual_results": {
            str(horizon): str(destination / f"horizon_{horizon}.json")
            for horizon in horizons
        },
    }
    summary_path = destination / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", type=int, nargs="+", default=(1, 4))
    parser.add_argument("--reference_steps", type=int, default=24)
    parser.add_argument("--surrogate", choices=("state", "flux"), default="flux")
    parser.add_argument("--n_cells", type=int, default=128)
    parser.add_argument("--train_samples", type=int, default=768)
    parser.add_argument("--calib_samples", type=int, default=192)
    parser.add_argument("--test_samples", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--rollout_cases", type=int, default=8)
    parser.add_argument("--fine_reference_factor", type=int, default=4)
    parser.add_argument("--sweep_points", type=int, default=9)
    parser.add_argument("--runtime_repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default="outputs/certified_burgers_horizons")
    parser.add_argument("--no_plots", action="store_true")
    arguments = parser.parse_args()
    config = ExperimentConfig(
        n_cells=arguments.n_cells,
        train_samples=arguments.train_samples,
        calib_samples=arguments.calib_samples,
        test_samples=arguments.test_samples,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        rollout_cases=arguments.rollout_cases,
        reference_steps=arguments.reference_steps,
        surrogate_kind=arguments.surrogate,
        fine_reference_factor=arguments.fine_reference_factor,
        sweep_points=arguments.sweep_points,
        runtime_repeats=arguments.runtime_repeats,
        seed=arguments.seed,
        make_plots=not arguments.no_plots,
    )
    print(
        json.dumps(
            run_horizon_study(
                config, horizons=arguments.horizons, out_dir=arguments.out_dir
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
