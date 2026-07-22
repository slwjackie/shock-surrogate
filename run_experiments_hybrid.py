#!/usr/bin/env python3
"""Run backbone, calibration, and RCCP rollout experiments across seeds/modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
import subprocess


def run(command: list[str], dry: bool = False) -> None:
    print("\n$ " + " ".join(command), flush=True)
    if not dry:
        subprocess.check_call(command)


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def dotted_keys(data, prefix=""):
    output = []
    if isinstance(data, dict):
        for key, value in data.items():
            full = f"{prefix}{key}"
            if isinstance(value, dict):
                output.extend(dotted_keys(value, full + "."))
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                output.append(full)
    return output


def nested_get(data, key):
    current = data
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="python")
    parser.add_argument("--arch", default="transformer_hybrid")
    parser.add_argument("--modes", nargs="+", default=["full", "no_causal", "no_phys", "data_only"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--H", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--physics_residual", choices=["discrete", "autograd"], default="discrete")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--ckpt_dir", default="ckpt")
    parser.add_argument("--outputs_dir", default="outputs")
    parser.add_argument("--with_policy", action="store_true")
    parser.add_argument("--threshold_scales", nargs="+", type=float, default=[0.75, 1.0, 1.25])
    parser.add_argument("--max_policy_cases", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_if_exists", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.outputs_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = str(data_dir / "meta.csv")
    paths = {
        "train": str(data_dir / "u_train.npz"),
        "val": str(data_dir / "u_val.npz"),
        "profile": str(data_dir / "u_test_profile_ood.npz"),
        "mismatch": str(data_dir / "u_test_mismatch_ood.npz"),
    }
    summary = {
        "arch": args.arch,
        "H": args.H,
        "modes": args.modes,
        "seeds": args.seeds,
        "runs": [],
        "aggregates": {},
    }

    for mode in args.modes:
        for seed in args.seeds:
            metrics_path = output_dir / f"metrics_{args.arch}_{mode}_seed{seed}_H{args.H}.json"
            checkpoint = Path(args.ckpt_dir) / f"best_{args.arch}_{mode}_seed{seed}_H{args.H}.pt"
            record = {"mode": mode, "seed": seed, "metrics_path": str(metrics_path)}
            try:
                if not (args.skip_if_exists and metrics_path.exists()):
                    train_command = [
                        args.python,
                        "train_transformer_hybrid.py",
                        "--arch", args.arch,
                        "--mode", mode,
                        "--seed", str(seed),
                        "--H", str(args.H),
                        "--epochs", str(args.epochs),
                        "--batch_size", str(args.batch_size),
                        "--lr", str(args.lr),
                        "--stride", str(args.stride),
                        "--num_workers", str(args.num_workers),
                        "--physics_residual", args.physics_residual,
                        "--meta_csv", meta,
                        "--u_train", paths["train"],
                        "--u_val", paths["val"],
                        "--save_dir", args.ckpt_dir,
                    ]
                    eval_command = [
                        args.python,
                        "eval_transformer_hybrid.py",
                        "--arch", args.arch,
                        "--mode", mode,
                        "--seed", str(seed),
                        "--H", str(args.H),
                        "--meta_csv", meta,
                        "--u_val", paths["val"],
                        "--u_test_profile", paths["profile"],
                        "--u_test_mismatch", paths["mismatch"],
                        "--ckpt_dir", args.ckpt_dir,
                        "--save_metrics",
                        "--out_metrics", str(metrics_path),
                    ]
                    run(train_command, args.dry_run)
                    run(eval_command, args.dry_run)
                record["metrics"] = load_json(metrics_path)

                if args.with_policy:
                    calibration_path = output_dir / f"calibration_{args.arch}_{mode}_seed{seed}_H{args.H}.json"
                    calibration_command = [
                        args.python,
                        "calibrate_residual_policy.py",
                        "--checkpoint", str(checkpoint),
                        "--meta_csv", meta,
                        "--u_train", paths["train"],
                        "--u_val", paths["val"],
                        "--out", str(calibration_path),
                    ]
                    run(calibration_command, args.dry_run)
                    policy_outputs = {}
                    for scale in args.threshold_scales:
                        policy_path = output_dir / f"policy_{args.arch}_{mode}_seed{seed}_H{args.H}_scale{scale:g}.json"
                        policy_command = [
                            args.python,
                            "eval_policy.py",
                            "--checkpoint", str(checkpoint),
                            "--calibration", str(calibration_path),
                            "--meta_csv", meta,
                            "--u_val", paths["val"],
                            "--u_test_profile", paths["profile"],
                            "--u_test_mismatch", paths["mismatch"],
                            "--threshold_scale", str(scale),
                            "--out", str(policy_path),
                        ]
                        if args.max_policy_cases is not None:
                            policy_command += ["--max_cases", str(args.max_policy_cases)]
                        run(policy_command, args.dry_run)
                        policy_outputs[str(scale)] = load_json(policy_path)
                    record["policy"] = policy_outputs
                summary["runs"].append(record)
            except subprocess.CalledProcessError as error:
                record["error"] = str(error)
                summary["runs"].append(record)
                if not args.continue_on_error:
                    raise

        runs = [r for r in summary["runs"] if r["mode"] == mode and r.get("metrics")]
        aggregate = {}
        keys = set()
        for record in runs:
            keys.update(dotted_keys(record["metrics"]))
        for key in sorted(keys):
            values = [nested_get(record["metrics"], key) for record in runs]
            values = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if values:
                aggregate[key] = {
                    "mean": mean(values),
                    "std": pstdev(values) if len(values) > 1 else 0.0,
                    "n": len(values),
                }
        summary["aggregates"][mode] = aggregate

    summary_path = output_dir / "summary_hybrid_all.json"
    if not args.dry_run:
        summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary to {summary_path}")


if __name__ == "__main__":
    main()
