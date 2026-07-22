#!/usr/bin/env python3
"""Autoregressive rollout evaluation for a trained 2-D LGNO checkpoint."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from hypersonic.models.lgno2d import LocalGlobalNeuralOperator2d
from hypersonic.state import pressure


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/hypersonic_2d.npz")
    p.add_argument("--checkpoint", default="ckpt/lgno2d_hypersonic.pt")
    p.add_argument("--case", type=int, default=0)
    p.add_argument("--steps", type=int, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["model_args"]
    model = LocalGlobalNeuralOperator2d(
        state_channels=4,
        width=cfg["width"],
        modes_x=cfg["modes_x"],
        modes_y=cfg["modes_y"],
        depth=cfg["depth"],
        boundary=cfg["boundary"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    payload = np.load(args.data)
    reference = torch.from_numpy(payload["u"][args.case]).to(device)
    dt = float(payload["sample_dt"])
    steps = min(args.steps or reference.shape[0] - 1, reference.shape[0] - 1)
    state = reference[0:1]
    predictions = [state.squeeze(0)]
    theta_values = []
    with torch.no_grad():
        for _ in range(steps):
            state, aux = model(state, dt=dt)
            predictions.append(state.squeeze(0))
            theta_values.append(float(aux["positivity_theta"].mean().item()))
    pred = torch.stack(predictions)
    ref = reference[: steps + 1]
    metrics = {
        "rollout_mse": float((pred - ref).square().mean().item()),
        "final_mse": float((pred[-1] - ref[-1]).square().mean().item()),
        "min_density": float(pred[:, 0].amin().item()),
        "min_pressure": float(pressure(pred, gamma=1.4).amin().item()),
        "mean_positivity_theta": float(np.mean(theta_values)) if theta_values else 1.0,
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
