#!/usr/bin/env python3
"""One-step evaluation for Transformer or FNO advice backbones."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from hybrid_temporal_dataset import HybridTemporalDataset
from models.model_hybrid_temporal_spatial import make_model
from physics.discrete_residual import burgers_reaction_residual_fd, centered_abs_gradient, physical_grid
from train_transformer_hybrid import move_params
from uncertainty.predictive import predict_distribution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default=None)
    parser.add_argument("--mode", default="full", choices=["full", "no_causal", "no_phys", "data_only"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--H", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--mc_samples", type=int, default=1)
    parser.add_argument("--meta_csv", default="data/meta.csv")
    parser.add_argument("--u_val", default="data/u_val.npz")
    parser.add_argument("--u_test_profile", default="data/u_test_profile_ood.npz")
    parser.add_argument("--u_test_mismatch", default="data/u_test_mismatch_ood.npz")
    parser.add_argument("--ckpt_dir", default="ckpt")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--save_metrics", action="store_true")
    parser.add_argument("--out_metrics", default=None)
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return 0.0
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.std() < 1e-12 or ya.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(xa, ya)[0, 1])


@torch.no_grad()
def evaluate_split(model, loader, device: torch.device, mc_samples: int = 1) -> dict[str, float]:
    squared_error = 0.0
    absolute_error = 0.0
    element_count = 0
    correct = 0
    sample_count = 0
    residual_sum = 0.0
    shock_error_sum = 0.0
    sample_errors: list[float] = []
    uncertainties: list[float] = []

    for x, history, target, last, regime, params, _ in loader:
        x = x.to(device)
        history = history.to(device)
        target = target.to(device)
        last = last.to(device)
        regime = regime.to(device)
        params_device = move_params(params, device)
        distribution = predict_distribution(
            model, x, history, params_device, mc_samples=mc_samples,
        )
        prediction, logits = distribution.mean, distribution.logits
        diff = prediction - target
        squared_error += float(diff.square().sum().item())
        absolute_error += float(diff.abs().sum().item())
        element_count += diff.numel()
        sample_count += x.shape[0]
        sample_errors.extend(diff.square().mean(dim=1).cpu().tolist())
        uncertainties.extend(distribution.uncertainty.cpu().tolist())
        if logits is not None:
            correct += int((logits.argmax(dim=1) == regime).sum().item())

        residual = burgers_reaction_residual_fd(prediction, last, x, params_device)
        residual_sum += float(residual.abs().mean(dim=1).sum().item())
        x_physical = physical_grid(x, params_device.get("L_mm"))
        pred_peak = centered_abs_gradient(prediction, x_physical).max(dim=1).values
        target_peak = centered_abs_gradient(target, x_physical).max(dim=1).values
        shock_error_sum += float((pred_peak - target_peak).abs().sum().item())

    mse = squared_error / max(element_count, 1)
    return {
        "mse": mse,
        "rmse": mse**0.5,
        "mae": absolute_error / max(element_count, 1),
        "accuracy": correct / max(sample_count, 1),
        "physics_residual_mae": residual_sum / max(sample_count, 1),
        "peak_gradient_error": shock_error_sum / max(sample_count, 1),
        "predictive_uncertainty_mean": float(np.mean(uncertainties)) if uncertainties else 0.0,
        "uncertainty_error_pearson": _pearson(uncertainties, sample_errors),
        "mc_samples": int(mc_samples),
        "n_samples": sample_count,
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    guessed_arch = args.arch or "transformer_hybrid"
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else Path(args.ckpt_dir) / f"best_{guessed_arch}_{args.mode}_seed{args.seed}_H{args.H}.pt"
    )
    checkpoint = load_checkpoint(checkpoint_path, device)
    saved_args = checkpoint.get("args", {})
    arch = args.arch or saved_args.get("arch", guessed_arch)
    history = int(saved_args.get("H", args.H))
    causal = saved_args.get("mode", args.mode) != "no_causal"
    model_kwargs = checkpoint.get("model_kwargs", {})
    model = make_model(arch, n_classes=3, causal=causal, history=history, **model_kwargs).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    splits = [
        ("val", "val", args.u_val),
        ("test_profile_ood", "test_profile_ood", args.u_test_profile),
        ("test_mismatch_ood", "test_mismatch_ood", args.u_test_mismatch),
    ]
    results = {}
    for name, split, path in splits:
        dataset = HybridTemporalDataset(args.meta_csv, path, split=split, H=history, stride=1)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        results[name] = evaluate_split(model, loader, device, mc_samples=args.mc_samples)

    if args.save_metrics:
        output = (
            Path(args.out_metrics)
            if args.out_metrics
            else Path("outputs") / f"metrics_{arch}_{args.mode}_seed{args.seed}_H{history}.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2))
        print(f"Wrote metrics to {output}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
