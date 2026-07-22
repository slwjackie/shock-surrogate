#!/usr/bin/env python3
"""Fit empirical residual-risk distributions on the validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from calibration.residual_calibrator import (
    COMPONENT_NAMES,
    EmpiricalRiskCalibrator,
    ParameterOODScorer,
)
from eval_transformer_hybrid import load_checkpoint
from hybrid_temporal_dataset import HybridTemporalDataset
from models.model_hybrid_temporal_spatial import make_model
from physics.discrete_residual import risk_components_torch
from train_transformer_hybrid import move_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--meta_csv", default="data/meta.csv")
    parser.add_argument("--u_train", default="data/u_train.npz")
    parser.add_argument("--u_val", default="data/u_val.npz")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--q_low", type=float, default=0.80)
    parser.add_argument("--q_high", type=float, default=0.95)
    parser.add_argument("--min_group_size", type=int, default=20)
    parser.add_argument("--group_source", choices=["predicted", "true", "none"], default="predicted")
    parser.add_argument("--out", default="calibration/residual_quantiles.json")
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
        arch,
        n_classes=3,
        causal=causal,
        history=history,
        **checkpoint.get("model_kwargs", {}),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    train_dataset = HybridTemporalDataset(args.meta_csv, args.u_train, "train", H=history)
    val_dataset = HybridTemporalDataset(args.meta_csv, args.u_val, "val", H=history)
    ood_scorer = ParameterOODScorer.fit(train_dataset.parameter_matrix())
    loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    collected = {name: [] for name in COMPONENT_NAMES}
    groups: list[str] = []
    with torch.no_grad():
        for x, state_history, _, u_last, regime, params, _ in loader:
            x = x.to(device)
            state_history = state_history.to(device)
            u_last = u_last.to(device)
            regime = regime.to(device)
            params_device = move_params(params, device)
            prediction, logits = model(x, state_history, params_device)

            coeff_ood_np = ood_scorer.score_mapping(params)
            coeff_ood = torch.as_tensor(
                coeff_ood_np, device=device, dtype=prediction.dtype
            )
            components = risk_components_torch(
                prediction,
                u_last,
                x,
                params_device,
                coeff_ood=coeff_ood,
            )
            for name in COMPONENT_NAMES:
                collected[name].append(components[name].detach().cpu().numpy())

            if args.group_source == "predicted" and logits is not None:
                groups.extend(str(int(v)) for v in logits.argmax(dim=1).cpu().tolist())
            elif args.group_source == "true":
                groups.extend(str(int(v)) for v in regime.cpu().tolist())

    arrays = {name: np.concatenate(values) for name, values in collected.items()}
    calibrator = EmpiricalRiskCalibrator(q_low=args.q_low, q_high=args.q_high)
    calibrator.ood_scorer = ood_scorer
    calibrator.fit(
        arrays,
        groups=groups if args.group_source != "none" else None,
        min_group_size=args.min_group_size,
    )
    calibrator.save(args.out)

    summary = {
        "checkpoint": args.checkpoint,
        "arch": arch,
        "history": history,
        "n_calibration_windows": int(len(arrays["residual"])),
        "global_thresholds": calibrator.global_thresholds,
        "group_thresholds": calibrator.group_thresholds,
        "output": args.out,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
