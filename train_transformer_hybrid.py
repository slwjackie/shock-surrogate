#!/usr/bin/env python3
"""
Train Hybrid Temporal+Spatial Transformer:
  - temporal causal attention over history H (FD time derivative)
  - spatial autograd derivatives via continuous x input
  - multitask: field regression + regime classification (+ optional physics residual)

Usage example:
  python train_transformer_hybrid.py --arch transformer_hybrid --mode full --seed 0 --H 5

Modes:
  - full: data + phys + tv + cls
  - no_causal: same as full but causal=False
  - no_phys: data + tv + cls (phys weight=0)
  - data_only: data + cls only (no phys, no tv)
"""

import argparse
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from hybrid_temporal_dataset import HybridTemporalDataset
from models.model_hybrid_temporal_spatial import make_model, physics_residual_hybrid

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def tv_1d(u: torch.Tensor):
    # u: (B, Nx)
    return (u[:, 1:] - u[:, :-1]).abs().mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce)  # prob of true class

        focal = (1.0 - pt) ** self.gamma * ce

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            focal = alpha[target] * focal

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        else:
            return focal

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="transformer_hybrid")
    ap.add_argument("--mode", default="full", choices=["full","no_causal","no_phys","data_only"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--H", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--meta_csv", default="meta.csv")
    ap.add_argument("--u_train", default="u_train.npz")
    ap.add_argument("--u_val", default="u_val.npz")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--save_dir", default="ckpt")
    # ap.add_argument("--cls_tail_frac", type=float, default=0.3,
    #             help="Use classification loss only on the last frac of windows. 0.3 means last 30%.")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # weights by mode
    w_data = 1.0
    w_cls  = 1.0
    w_tv   = 0.0 if args.mode == "data_only" else 1e-3
    w_phys = 0.0 if args.mode in ["no_phys","data_only"] else 5e-3
    causal = False if args.mode == "no_causal" else True

    # datasets
    ds_train = HybridTemporalDataset(args.meta_csv, args.u_train, split="train", H=args.H, stride=args.stride)
    ds_val   = HybridTemporalDataset(args.meta_csv, args.u_val,   split="val",   H=args.H, stride=args.stride)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    dl_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    n_classes = 3
    model = make_model(args.arch, n_classes=n_classes, causal=causal).to(device)

    # opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # mse = nn.MSELoss()
    # ce  = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    ce  = FocalLoss(alpha=[1.0, 1.0, 1.2], gamma=2.0, reduction="mean")

    save_dir = Path(args.save_dir); save_dir.mkdir(exist_ok=True)
    best_path = save_dir / f"best_{args.arch}_{args.mode}_seed{args.seed}_H{args.H}.pt"
    best_val = float("inf")

    def run_epoch(train: bool):
        model.train(train)
        total = {"loss":0.0,"data":0.0,"phys":0.0,"tv":0.0,"cls":0.0,"mse":0.0,"acc":0.0,"n":0}
        loader = dl_train if train else dl_val
        for batch in loader:
            x, u_hist, u_next, u_last, regime_id, params, _ = batch
            # x: (B,Nx), u_hist: (B,Nx,H), u_next: (B,Nx)
            x = x.to(device)
            u_hist = u_hist.to(device)
            u_next = u_next.to(device)
            u_last = u_last.to(device)
            y_cls = regime_id.to(device)

            # enable autograd on x for spatial derivatives
            x = x.clone().detach().requires_grad_(w_phys > 0)

            u_pred, logits = model(x, u_hist)

            data_loss = mse(u_pred, u_next)

            # cls_loss = torch.tensor(0.0, device=device)
            # acc = torch.tensor(0.0, device=device)
            # if logits is not None:
            #     cls_loss = ce(logits, y_cls)
            #     acc = (logits.argmax(dim=1) == y_cls).float().mean()
            # cls_loss = torch.tensor(0.0, device=device)
            # acc = torch.tensor(0.0, device=device)
            # acc_tail = torch.tensor(0.0, device=device)

            # if logits is not None:
            #     # overall accuracy (전체 샘플 기준, 모니터링용)
            #     acc = (logits.argmax(dim=1) == y_cls).float().mean()

            #     # t0 기반으로 후반 window만 classification loss 적용
            #     t0 = t0.to(device)
            #     tail_frac = float(args.cls_tail_frac)
            #     tail_frac = min(max(tail_frac, 0.0), 1.0)

                # Nt-H 개의 가능한 시작점 중 마지막 tail_frac 부분만 사용
                #n_t0_total = max(ds_train.Nt - args.H, 1)
                # n_t0_total = max(loader.dataset.Nt - args.H, 1)
                # t_cut = int((1.0 - tail_frac) * n_t0_total)

                # mask = t0 >= t_cut  # 후반 window만 True

                # if mask.any():
                #     cls_loss = ce(logits[mask], y_cls[mask])
                #     acc_tail = (logits[mask].argmax(dim=1) == y_cls[mask]).float().mean()
                # else:
                #     cls_loss = torch.tensor(0.0, device=device)
                #     acc_tail = torch.tensor(0.0, device=device)
                # n_tail_batch = 0
                # if mask.any():
                #     cls_loss = ce(logits[mask], y_cls[mask])
                #     acc_tail = (logits[mask].argmax(dim=1) == y_cls[mask]).float().mean()
                #     n_tail_batch = int(mask.sum().item())
                # else:
                #     cls_loss = torch.tensor(0.0, device=device)
                #     acc_tail = torch.tensor(0.0, device=device)
            cls_loss = torch.tensor(0.0, device=device)
            acc = torch.tensor(0.0, device=device)

            if logits is not None:
                cls_loss = ce(logits, y_cls)
                acc = (logits.argmax(dim=1) == y_cls).float().mean()
                                
                    
            tv_loss = tv_1d(u_pred)

            phys_loss = torch.tensor(0.0, device=device)
            if w_phys > 0:
                # params is a dict of tensors (DataLoader collates dict values)
                dt = params["dt"].to(device).float() if isinstance(params["dt"], torch.Tensor) else torch.tensor(params["dt"], device=device).float()
                nu = params["nu"].to(device).float()
                k  = params["k"].to(device).float() if "k" in params else torch.zeros_like(nu)
                E  = params["E"].to(device).float() if "E" in params else torch.zeros_like(nu)
                dTdx = params["dTdx"].to(device).float()
                bq = params["b_quad"].to(device).float() if "b_quad" in params else None
                r = physics_residual_hybrid(u_pred, u_last, x, dt=dt, nu=nu, k=k, E=E, dTdx=dTdx, b_quad=bq)
                phys_loss = (r**2).mean()

            loss = w_data*data_loss + w_cls*cls_loss + w_tv*tv_loss + w_phys*phys_loss

            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            bs = x.shape[0]
            total["loss"] += loss.item()*bs
            total["data"] += data_loss.item()*bs
            total["phys"] += phys_loss.item()*bs
            total["tv"]   += tv_loss.item()*bs
            total["cls"]  += cls_loss.item()*bs
            total["mse"]  += data_loss.item()*bs
            total["acc"]  += acc.item()*bs
            # if n_tail_batch > 0:
            #     total["acc_tail"] += acc_tail.item() * n_tail_batch
            #     total["n_tail"] += n_tail_batch
            total["n"]    += bs
        # for k in ["loss","data","phys","tv","cls","mse","acc"]:
        #     total[k] /= max(total["n"],1)

        # if total["n_tail"] > 0:
        #     total["acc_tail"] /= total["n_tail"]
        # else:
        #     total["acc_tail"] = 0.0
        # return total
        for k in ["loss","data","phys","tv","cls","mse","acc"]:
            total[k] /= max(total["n"],1)
        return total

    for ep in range(1, args.epochs+1):
        tr = run_epoch(train=True)
        va = run_epoch(train=False)
        print(f"[{args.arch}/{args.mode}] ep {ep:4d} | train loss {tr['loss']:.3e} (data {tr['data']:.2e} phys {tr['phys']:.2e} tv {tr['tv']:.2e} cls {tr['cls']:.2e}) | "
              f"val mse {va['mse']:.3e} acc {va['acc']:.2f}")

        if va["mse"] < best_val:
            best_val = va["mse"]
            torch.save({"model": model.state_dict(), "args": vars(args)}, best_path)

    print(f"Saved best ckpt: {best_path} (best val mse={best_val:.3e})")

if __name__ == "__main__":
    main()
