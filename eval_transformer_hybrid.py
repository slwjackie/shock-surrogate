#!/usr/bin/env python3
"""
Evaluate Hybrid Temporal+Spatial Transformer on:
  - val
  - test_profile_ood
  - test_mismatch_ood

Writes metrics JSON.
"""

import argparse, json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hybrid_temporal_dataset import HybridTemporalDataset
from models.model_hybrid_temporal_spatial import make_model

def mse_rmse(pred, y):
    mse = ((pred - y)**2).mean().item()
    rmse = mse**0.5
    return mse, rmse

# @torch.no_grad()
# def eval_split(model, dl, device):
#     ce = nn.CrossEntropyLoss()
#     total = {"mse":0.0,"rmse":0.0,"acc":0.0,"acc_tail":0.0,"cls_loss":0.0,"n":0, "n_tail": 0}
#     for batch in dl:
#         x, u_hist, u_next, u_last, regime_id, params, t0 = batch
#         x = x.to(device)
#         u_hist = u_hist.to(device)
#         u_next = u_next.to(device)
#         y_cls = regime_id.to(device)

#         u_pred, logits = model(x, u_hist)
#         mse_v, rmse_v = mse_rmse(u_pred, u_next)

#         # acc = 0.0
#         # cls_loss = 0.0
#         # if logits is not None:
#         #     cls_loss = ce(logits, y_cls).item()
#         #     acc = (logits.argmax(dim=1) == y_cls).float().mean().item()
#         cls_loss = 0.0
#         acc = 0.0
#         acc_tail = 0.0
#         n_tail_batch = 0

#         if logits is not None:
#             pred_cls = logits.argmax(dim=1)

#             # 전체 accuracy
#             cls_loss = ce(logits, y_cls).item()
#             acc = (pred_cls == y_cls).float().mean().item()

#             # tail accuracy
#             t0 = t0.to(device)
#             tail_frac = float(args.cls_tail_frac)
#             tail_frac = min(max(tail_frac, 0.0), 1.0)

#             n_t0_total = max(loader.dataset.Nt - args.H, 1)
#             t_cut = int((1.0 - tail_frac) * n_t0_total)

#             mask = t0 >= t_cut

#             if mask.any():
#                 acc_tail = (pred_cls[mask] == y_cls[mask]).float().mean().item()
#                 n_tail_batch = int(mask.sum().item())

#         bs = x.shape[0]
#         total["mse"] += mse_v*bs
#         total["rmse"] += rmse_v*bs
#         total["acc"] += acc*bs
#         total["cls_loss"] += cls_loss*bs
#         total["n"] += bs
#     for k in ["mse","rmse","acc","cls_loss"]:
#         total[k] /= max(total["n"],1)
#     return total

@torch.no_grad()
def eval_split(model, dl, device, H, cls_tail_frac):
    ce = nn.CrossEntropyLoss()
    total = {
        "mse": 0.0,
        "rmse": 0.0,
        "acc": 0.0,
        "acc_tail": 0.0,
        "cls_loss": 0.0,
        "n": 0,
        "n_tail": 0,
    }

    for batch in dl:
        x, u_hist, u_next, u_last, regime_id, params, t0 = batch
        x = x.to(device)
        u_hist = u_hist.to(device)
        u_next = u_next.to(device)
        y_cls = regime_id.to(device)

        u_pred, logits = model(x, u_hist)
        mse_v, rmse_v = mse_rmse(u_pred, u_next)

        cls_loss = 0.0
        acc = 0.0
        acc_tail = 0.0
        n_tail_batch = 0

        if logits is not None:
            pred_cls = logits.argmax(dim=1)

            # 전체 accuracy
            cls_loss = ce(logits, y_cls).item()
            acc = (pred_cls == y_cls).float().mean().item()

            # tail accuracy
            t0 = t0.to(device)
            tail_frac = float(cls_tail_frac)
            tail_frac = min(max(tail_frac, 0.0), 1.0)

            n_t0_total = max(dl.dataset.Nt - H, 1)
            t_cut = int((1.0 - tail_frac) * n_t0_total)

            mask = t0 >= t_cut

            if mask.any():
                acc_tail = (pred_cls[mask] == y_cls[mask]).float().mean().item()
                n_tail_batch = int(mask.sum().item())

        bs = x.shape[0]
        total["mse"] += mse_v * bs
        total["rmse"] += rmse_v * bs
        total["acc"] += acc * bs
        total["cls_loss"] += cls_loss * bs
        total["n"] += bs

        if n_tail_batch > 0:
            total["acc_tail"] += acc_tail * n_tail_batch
            total["n_tail"] += n_tail_batch

    for k in ["mse", "rmse", "acc", "cls_loss"]:
        total[k] /= max(total["n"], 1)

    #total["acc_tail"] /= max(total["n_tail"], 1)
    if total["n_tail"] > 0:
        total["acc_tail"] /= total["n_tail"]
    else:
        total["acc_tail"] = None
    
    return total



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="transformer_hybrid")
    ap.add_argument("--mode", default="full", choices=["full","no_causal","no_phys","data_only"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--H", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--meta_csv", default="meta.csv")
    ap.add_argument("--u_val", default="u_val.npz")
    ap.add_argument("--u_test_profile", default="u_test_profile_ood.npz")
    ap.add_argument("--u_test_mismatch", default="u_test_mismatch_ood.npz")
    ap.add_argument("--ckpt_dir", default="ckpt")
    ap.add_argument("--save_metrics", action="store_true")
    ap.add_argument("--out_metrics", default=None)
    ap.add_argument("--cls_tail_frac", type=float, default=0.3,
                help="Evaluate tail accuracy on the last frac of windows.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    causal = False if args.mode == "no_causal" else True
    n_classes = 3
    model = make_model(args.arch, n_classes=n_classes, causal=causal).to(device)

    ckpt_path = Path(args.ckpt_dir) / f"best_{args.arch}_{args.mode}_seed{args.seed}_H{args.H}.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    splits = [
        ("val", "val", args.u_val),
        ("test_profile_ood", "test_profile_ood", args.u_test_profile),
        ("test_mismatch_ood", "test_mismatch_ood", args.u_test_mismatch),
    ]
    results = {}
    for name, split, npz in splits:
        ds = HybridTemporalDataset(args.meta_csv, npz, split=split, H=args.H, stride=1)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        results[name] = eval_split(model, dl, device, args.H, args.cls_tail_frac)

    if args.save_metrics:
        out = Path(args.out_metrics) if args.out_metrics else Path("outputs") / f"metrics_{args.arch}_{args.mode}_seed{args.seed}_H{args.H}.json"
        out.parent.mkdir(exist_ok=True)
        with out.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote metrics to {out}")

    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
