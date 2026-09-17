#!/usr/bin/env python3
"""Compare pure-surrogate and RCCP-controlled long rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from calibration.residual_calibrator import EmpiricalRiskCalibrator
from eval_transformer_hybrid import load_checkpoint
from evaluation.rollout_policy_eval import RolloutCosts, aggregate_case_metrics, rollout_case
from hybrid_temporal_dataset import HybridTemporalDataset
from models.model_hybrid_temporal_spatial import make_model
from physics.residual_projection import ResidualProjectionConfig
from policies.clamp_policy import ResidualCalibratedClampPolicy
from solvers.weno_adapter import WENOSolverAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--meta_csv", default="data/meta.csv")
    parser.add_argument("--u_val", default="data/u_val.npz")
    parser.add_argument("--u_test_profile", default="data/u_test_profile_ood.npz")
    parser.add_argument("--u_test_mismatch", default="data/u_test_mismatch_ood.npz")
    parser.add_argument("--splits", nargs="+", default=["val", "test_profile_ood", "test_mismatch_ood"])
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--mc_samples", type=int, default=8)
    parser.add_argument("--threshold_scale", type=float, default=1.0)
    parser.add_argument("--hard_guard_multiplier", type=float, default=2.0)
    parser.add_argument(
        "--group_strategy",
        choices=["global", "predicted", "soft_conservative", "worst_case"],
        default="soft_conservative",
    )
    parser.add_argument("--group_safety_blend", type=float, default=0.25)
    parser.add_argument("--projection_steps", type=int, default=4)
    parser.add_argument("--projection_lr", type=float, default=2e-2)
    parser.add_argument("--surrogate_cost", type=float, default=1.0)
    parser.add_argument("--correction_cost", type=float, default=4.0)
    parser.add_argument("--fallback_cost", type=float, default=50.0)
    parser.add_argument("--out", default="outputs/policy_rollout_metrics.json")
    parser.add_argument("--save_cases", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(Path(args.checkpoint), device)
    saved_args = checkpoint.get("args", {})
    arch = saved_args.get("arch", "transformer_hybrid")
    history = int(saved_args.get("H", 5))
    causal = saved_args.get("mode", "full") != "no_causal"
    model = make_model(
        arch, n_classes=3, causal=causal, history=history,
        **checkpoint.get("model_kwargs", {}),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    calibrator = EmpiricalRiskCalibrator.load(args.calibration)
    policy = ResidualCalibratedClampPolicy(
        calibrator,
        threshold_scale=args.threshold_scale,
        hard_guard_multiplier=args.hard_guard_multiplier,
        group_strategy=args.group_strategy,
        group_safety_blend=args.group_safety_blend,
    )
    solver = WENOSolverAdapter()
    projection = ResidualProjectionConfig(
        steps=args.projection_steps, step_size=args.projection_lr,
    )
    costs = RolloutCosts(
        surrogate=args.surrogate_cost,
        correction=args.correction_cost,
        fallback=args.fallback_cost,
    )
    split_paths = {
        "val": args.u_val,
        "test_profile_ood": args.u_test_profile,
        "test_mismatch_ood": args.u_test_mismatch,
    }
    results = {
        "checkpoint": args.checkpoint,
        "calibration": args.calibration,
        "arch": arch,
        "history": history,
        "mc_samples": args.mc_samples,
        "threshold_scale": args.threshold_scale,
        "group_strategy": args.group_strategy,
        "splits": {},
    }
    for split in args.splits:
        if split not in split_paths:
            raise ValueError(f"Unknown split: {split}")
        dataset = HybridTemporalDataset(
            args.meta_csv, split_paths[split], split=split, H=history,
        )
        n_cases = dataset.Ncases if args.max_cases is None else min(dataset.Ncases, args.max_cases)
        pure_cases = []
        policy_cases = []
        for case_index in range(n_cases):
            trajectory = dataset.case_trajectory(case_index)
            params = dataset.case_params(case_index)
            pure_cases.append(
                rollout_case(
                    model, trajectory, dataset.x, params, history, device,
                    policy=None, costs=costs, mc_samples=1,
                )
            )
            policy_cases.append(
                rollout_case(
                    model, trajectory, dataset.x, params, history, device,
                    policy=policy, solver=solver,
                    projection_config=projection, costs=costs,
                    mc_samples=args.mc_samples,
                )
            )
        split_result = {
            "surrogate_only": aggregate_case_metrics(pure_cases),
            "rccp": aggregate_case_metrics(policy_cases),
        }
        if args.save_cases:
            split_result["case_metrics"] = {
                "surrogate_only": pure_cases,
                "rccp": policy_cases,
            }
        results["splits"][split] = split_result

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Wrote rollout metrics to {output}")


if __name__ == "__main__":
    main()
