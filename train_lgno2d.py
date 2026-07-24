#!/usr/bin/env python3
"""Train the 2-D local--global operator on consecutive compressible-flow snapshots."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from hypersonic.losses import lgno_loss
from hypersonic.models.lgno2d import LocalGlobalNeuralOperator2d


class SnapshotPairDataset(Dataset):
    def __init__(self, path: str):
        payload = np.load(path)
        self.u = payload["u"].astype(np.float32)
        self.dt = float(payload["sample_dt"])
        self.indices = [(i, t) for i in range(self.u.shape[0]) for t in range(self.u.shape[1] - 1)]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int):
        case, time = self.indices[index]
        return torch.from_numpy(self.u[case, time]), torch.from_numpy(self.u[case, time + 1])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/hypersonic_2d.npz")
    p.add_argument("--out", default="ckpt/lgno2d_hypersonic.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--modes_x", type=int, default=16)
    p.add_argument("--modes_y", type=int, default=16)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--boundary", choices=["periodic", "outflow", "outflow_learned"], default="outflow_learned")
    p.add_argument("--spectral_weight", type=float, default=1e-3)
    p.add_argument("--high_fraction", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SnapshotPairDataset(args.data)
    n_val = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = LocalGlobalNeuralOperator2d(state_channels=4, width=args.width, modes_x=args.modes_x, modes_y=args.modes_y, depth=args.depth, boundary=args.boundary).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best = float("inf")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total = 0.0
        train_n = 0
        for state, target in train_loader:
            state, target = state.to(device), target.to(device)
            pred, _ = model(state, dt=dataset.dt)
            loss, _ = lgno_loss(pred, target, args.spectral_weight, args.high_fraction)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_total += float(loss.item()) * state.shape[0]
            train_n += state.shape[0]

        model.eval()
        val_total = 0.0
        val_n = 0
        with torch.no_grad():
            for state, target in val_loader:
                state, target = state.to(device), target.to(device)
                pred, _ = model(state, dt=dataset.dt)
                loss, _ = lgno_loss(pred, target, args.spectral_weight, args.high_fraction)
                val_total += float(loss.item()) * state.shape[0]
                val_n += state.shape[0]
        train_mean = train_total / max(train_n, 1)
        val_mean = val_total / max(val_n, 1)
        print(f"epoch {epoch:4d} train={train_mean:.6e} val={val_mean:.6e}")
        if val_mean < best:
            best = val_mean
            torch.save({"model": model.state_dict(), "model_args": vars(args), "sample_dt": dataset.dt, "best_val": best}, out)
    print(f"Saved {out} with best validation loss {best:.6e}")


if __name__ == "__main__":
    main()
