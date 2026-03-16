#!/usr/bin/env python3
"""Quick OOD sanity check for generated Burgers-reaction datasets.

What this script checks
-----------------------
1. Parameter shift summary between train / profile OOD / mismatch OOD.
2. Field-level differences using multiple train cases, not just one example.
3. Initial-time and final-time distances, so solver-mismatch OOD is not judged
   only from a near-identical initial condition.
4. Simple trajectory diagnostics that are more meaningful for shock-like data:
   max(u), max(|du/dx|), and time-averaged max(|du/dx|).

Usage
-----
python verify_ood.py
python verify_ood.py --data_dir data --n_examples 3 --n_pairs 32
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


PARAM_COLS = ["dTdx", "b_quad", "nu", "k", "E"]
SPLITS = ["train", "test_profile_ood", "test_mismatch_ood"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data", help="Directory containing meta.csv and .npz files")
    p.add_argument("--n_examples", type=int, default=3, help="How many per-split examples to print")
    p.add_argument(
        "--n_pairs",
        type=int,
        default=32,
        help="How many train-vs-OOD pairs to evaluate for aggregate statistics",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_split(data_dir: Path, split: str) -> np.ndarray:
    name_map = {
        "train": "u_train.npz",
        "val": "u_val.npz",
        "test_profile_ood": "u_test_profile_ood.npz",
        "test_mismatch_ood": "u_test_mismatch_ood.npz",
    }
    path = data_dir / name_map[split]
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    arr = np.load(path)["u"]
    if arr.ndim != 3:
        raise ValueError(f"Expected array of shape [N,T,X] in {path}, got {arr.shape}")
    return arr


def require_cols(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"meta.csv is missing required columns: {missing}")


def nearest_by_params(train_params: np.ndarray, q: np.ndarray) -> int:
    # Normalize by train spread so each parameter contributes comparably.
    scale = np.std(train_params, axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    d2 = np.sum(((train_params - q[None, :]) / scale[None, :]) ** 2, axis=1)
    return int(np.argmin(d2))


def grad_abs_max(U: np.ndarray) -> float:
    # U shape [T, X]
    gx = np.gradient(U, axis=-1)
    return float(np.max(np.abs(gx)))


def grad_abs_timeavg(U: np.ndarray) -> float:
    gx = np.gradient(U, axis=-1)
    return float(np.mean(np.max(np.abs(gx), axis=-1)))


def traj_metrics(Ua: np.ndarray, Ub: np.ndarray) -> Dict[str, float]:
    diff = Ua - Ub
    return {
        "full_mse": float(np.mean(diff**2)),
        "t0_mse": float(np.mean((Ua[0] - Ub[0]) ** 2)),
        "tend_mse": float(np.mean((Ua[-1] - Ub[-1]) ** 2)),
        "max_u_a": float(np.max(Ua)),
        "max_u_b": float(np.max(Ub)),
        "peak_grad_a": grad_abs_max(Ua),
        "peak_grad_b": grad_abs_max(Ub),
        "avg_peak_grad_a": grad_abs_timeavg(Ua),
        "avg_peak_grad_b": grad_abs_timeavg(Ub),
    }


def summarize_params(df: pd.DataFrame, split: str) -> pd.DataFrame:
    sub = df[df["split"] == split]
    rows = []
    for c in PARAM_COLS:
        rows.append(
            {
                "param": c,
                "min": float(sub[c].min()),
                "max": float(sub[c].max()),
                "mean": float(sub[c].mean()),
                "std": float(sub[c].std(ddof=0)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    data_dir = Path(args.data_dir)

    meta_path = data_dir / "meta.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")

    df = pd.read_csv(meta_path)
    require_cols(df, ["split", "case_id", *PARAM_COLS])

    arrays = {split: load_split(data_dir, split) for split in SPLITS}

    print("=== Parameter ranges by split ===")
    for split in SPLITS:
        print(f"\n[{split}]")
        print(summarize_params(df, split).to_string(index=False))

    train_meta = df[df["split"] == "train"].sort_values("case_id").reset_index(drop=True)
    prof_meta = df[df["split"] == "test_profile_ood"].sort_values("case_id").reset_index(drop=True)
    mis_meta = df[df["split"] == "test_mismatch_ood"].sort_values("case_id").reset_index(drop=True)

    train_params = train_meta[PARAM_COLS].to_numpy(dtype=float)

    def report_examples(name: str, meta_sub: pd.DataFrame, U_sub: np.ndarray) -> None:
        n = min(args.n_examples, len(meta_sub))
        idxs = np.linspace(0, len(meta_sub) - 1, n, dtype=int)
        print(f"\n=== Example cases: {name} ===")
        for idx in idxs:
            row = meta_sub.iloc[idx]
            U = U_sub[int(row.case_id)]
            print(
                f"case_id={int(row.case_id)} | "
                f"dTdx={row.dTdx:.6g}, b={row.b_quad:.6g}, nu={row.nu:.6g}, k={row.k:.6g}, E={row.E:.6g} | "
                f"min/max={float(U.min()):.6g}/{float(U.max()):.6g} | "
                f"peak|ux|={grad_abs_max(U):.6g} | avg_t peak|ux|={grad_abs_timeavg(U):.6g}"
            )

    report_examples("train", train_meta, arrays["train"])
    report_examples("test_profile_ood", prof_meta, arrays["test_profile_ood"])
    report_examples("test_mismatch_ood", mis_meta, arrays["test_mismatch_ood"])

    def aggregate_against_train(name: str, meta_sub: pd.DataFrame, U_sub: np.ndarray) -> None:
        n = min(args.n_pairs, len(meta_sub))
        chosen = rng.choice(len(meta_sub), size=n, replace=False)
        rows = []
        for j in chosen:
            row = meta_sub.iloc[int(j)]
            q = row[PARAM_COLS].to_numpy(dtype=float)
            i = nearest_by_params(train_params, q)
            U_train = arrays["train"][int(train_meta.iloc[i].case_id)]
            U_ood = U_sub[int(row.case_id)]
            rows.append(traj_metrics(U_train, U_ood))
        out = pd.DataFrame(rows)
        print(f"\n=== Aggregate comparison: train vs {name} (nearest-parameter pairing, n={len(out)}) ===")
        summary = out.agg(["mean", "std", "min", "max"]).T
        print(summary.to_string(float_format=lambda x: f"{x:.6g}"))

    aggregate_against_train("test_profile_ood", prof_meta, arrays["test_profile_ood"])
    aggregate_against_train("test_mismatch_ood", mis_meta, arrays["test_mismatch_ood"])

    print("\nInterpretation tips:")
    print("- Profile OOD should usually show larger t0/final field differences because the input profile itself changes.")
    print("- Solver-mismatch OOD may have modest t0 difference but noticeably different final-time error or gradient diagnostics.")
    print("- If mismatch OOD only differs in parameters but not in trajectory diagnostics, consider increasing coefficient separation in build_dataset.py.")


if __name__ == "__main__":
    main()
