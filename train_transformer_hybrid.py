#!/usr/bin/env python3
"""Train an interchangeable parameter-conditioned shock-surrogate backbone."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from backbones.conditioning import fit_parameter_stats
from hybrid_temporal_dataset import HybridTemporalDataset
from models.model_hybrid_temporal_spatial import make_model, physics_residual_hybrid
from physics.discrete_residual import shock_aware_residual_loss, total_variation


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.register_buffer(
            "alpha",
            None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32),
        )
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, reduction="none")
        focal = (1.0 - torch.exp(-ce)).pow(self.gamma) * ce
        if self.alpha is not None:
            focal = self.alpha[target] * focal
        if self.reduction == "sum":
            return focal.sum()
        if self.reduction == "none":
            return focal
        return focal.mean()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_params(params, device: torch.device) -> dict[str, torch.Tensor]:
    moved = {}
    for key, value in params.items():
        moved[key] = value.to(device=device, dtype=torch.float32) if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=device, dtype=torch.float32)
    return moved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="transformer_hybrid")
    parser.add_argument("--mode", default="full", choices=["full", "no_causal", "no_phys", "data_only"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--H", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--meta_csv", default="data/meta.csv")
    parser.add_argument("--u_train", default="data/u_train.npz")
    parser.add_argument("--u_val", default="data/u_val.npz")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--save_dir", default="ckpt")
    parser.add_argument("--physics_residual", choices=["discrete", "autograd"], default="discrete")
    parser.add_argument("--shock_beta", type=float, default=2.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--w_data", type=float, default=1.0)
    parser.add_argument("--w_cls", type=float, default=1.0)
    parser.add_argument("--w_phys", type=float, default=None)
    parser.add_argument("--w_tv", type=float, default=None)

    # Shared/Transformer parameters.
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--mlp_hidden", type=int, default=128)
    # FNO-specific parameters.
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--modes", type=int, default=24)
    parser.add_argument("--depth", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    default_w_tv = 0.0 if args.mode == "data_only" else 1e-3
    default_w_phys = 0.0 if args.mode in {"no_phys", "data_only"} else 5e-3
    w_tv = default_w_tv if args.w_tv is None else float(args.w_tv)
    w_phys = default_w_phys if args.w_phys is None else float(args.w_phys)
    causal = args.mode != "no_causal"

    train_ds = HybridTemporalDataset(args.meta_csv, args.u_train, "train", H=args.H, stride=args.stride)
    val_ds = HybridTemporalDataset(args.meta_csv, args.u_val, "val", H=args.H, stride=args.stride)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model_kwargs = {
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
        "mlp_hidden": args.mlp_hidden,
        "width": args.width,
        "modes": args.modes,
        "depth": args.depth,
    }
    model = make_model(
        args.arch,
        n_classes=3,
        causal=causal,
        history=args.H,
        **model_kwargs,
    ).to(device)
    parameter_mean, parameter_std = fit_parameter_stats(train_ds.parameter_matrix())
    if hasattr(model, "set_parameter_stats"):
        model.set_parameter_stats(parameter_mean, parameter_std)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05
    )
    mse = nn.MSELoss()
    classification_loss = FocalLoss(alpha=[1.0, 1.0, 1.2], gamma=2.0).to(device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / f"best_{args.arch}_{args.mode}_seed{args.seed}_H{args.H}.pt"
    best_val = float("inf")

    def run_epoch(training: bool) -> dict[str, float]:
        model.train(training)
        totals = {name: 0.0 for name in ["loss", "data", "phys", "tv", "cls", "mse", "acc"]}
        totals["n"] = 0.0
        loader = train_loader if training else val_loader
        needs_autograd_residual = w_phys > 0 and args.physics_residual == "autograd"
        grad_context = nullcontext() if training or needs_autograd_residual else torch.no_grad()

        with grad_context:
            for x, history, target, last, regime, params, _ in loader:
                x = x.to(device)
                history = history.to(device)
                target = target.to(device)
                last = last.to(device)
                regime = regime.to(device)
                params_device = move_params(params, device)

                if needs_autograd_residual:
                    x = x.clone().detach().requires_grad_(True)

                prediction, logits = model(x, history, params_device)
                data_loss = mse(prediction, target)
                cls_loss = (
                    classification_loss(logits, regime)
                    if logits is not None
                    else torch.zeros((), device=device)
                )
                accuracy = (
                    (logits.argmax(dim=1) == regime).float().mean()
                    if logits is not None
                    else torch.zeros((), device=device)
                )

                # Match target total variation instead of suppressing shocks.
                tv_loss = (total_variation(prediction) - total_variation(target)).abs().mean()
                physics_loss = torch.zeros((), device=device)
                if w_phys > 0:
                    if args.physics_residual == "discrete":
                        physics_loss, _ = shock_aware_residual_loss(
                            prediction,
                            last,
                            x,
                            params_device,
                            shock_beta=args.shock_beta,
                        )
                    else:
                        residual = physics_residual_hybrid(
                            prediction,
                            last,
                            x,
                            dt=params_device["dt"],
                            nu=params_device["nu"],
                            k=params_device["k"],
                            E=params_device["E"],
                            dTdx=params_device["dTdx"],
                            b_quad=params_device.get("b_quad"),
                        )
                        physics_loss = residual.square().mean()

                loss = (
                    args.w_data * data_loss
                    + args.w_cls * cls_loss
                    + w_tv * tv_loss
                    + w_phys * physics_loss
                )

                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()

                batch = x.shape[0]
                values = {
                    "loss": loss,
                    "data": data_loss,
                    "phys": physics_loss,
                    "tv": tv_loss,
                    "cls": cls_loss,
                    "mse": data_loss,
                    "acc": accuracy,
                }
                for name, value in values.items():
                    totals[name] += float(value.detach().item()) * batch
                totals["n"] += batch

        count = max(totals["n"], 1.0)
        return {name: totals[name] / count for name in totals if name != "n"}

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(True)
        val_metrics = run_epoch(False)
        scheduler.step()
        print(
            f"[{args.arch}/{args.mode}] ep {epoch:4d} | "
            f"train {train_metrics['loss']:.3e} "
            f"(data {train_metrics['data']:.2e} phys {train_metrics['phys']:.2e} "
            f"tv {train_metrics['tv']:.2e} cls {train_metrics['cls']:.2e}) | "
            f"val mse {val_metrics['mse']:.3e} acc {val_metrics['acc']:.2f}"
        )
        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "model_kwargs": model_kwargs,
                    "parameter_mean": parameter_mean,
                    "parameter_std": parameter_std,
                    "best_val_mse": best_val,
                },
                best_path,
            )

    print(f"Saved best checkpoint: {best_path} (val mse={best_val:.3e})")


if __name__ == "__main__":
    main()
