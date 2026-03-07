#!/usr/bin/env python3
"""
Run Hybrid Transformer experiments (train + eval) across modes/seeds.

Uses *_updated.py train/eval scripts and passes data paths under --data_dir.

Example:
  python run_experiments_hybrid_updated.py --data_dir data --epochs 1200 --modes full no_causal no_phys --seeds 0 1 2 3 4 --H 5 \
    --skip_if_exists --continue_on_error
"""

import argparse, json, subprocess
from pathlib import Path
from statistics import mean, pstdev


def run(cmd, dry=False):
    print("\n$ " + " ".join(cmd), flush=True)
    if dry:
        return
    subprocess.check_call(cmd)


def load_json(p: Path):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def dotted_keys(d, prefix=""):
    out = []
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict):
                out += dotted_keys(v, prefix + k + ".")
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                out.append(prefix + k)
    return out


def get(d, key):
    cur = d
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default="python")
    ap.add_argument("--arch", default="transformer_hybrid")
    ap.add_argument("--modes", nargs="+", default=["full", "no_causal", "no_phys"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    ap.add_argument("--H", type=int, default=5)

    ap.add_argument("--epochs", type=int, default=1200)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--cls_tail_frac", type=float, default=0.3,
                help="Use classification loss only on the last frac of windows.")

    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--ckpt_dir", default="ckpt")
    ap.add_argument("--outputs_dir", default="outputs")

    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--skip_if_exists", action="store_true")
    ap.add_argument("--continue_on_error", action="store_true")

    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    meta_csv = str(data_dir / "meta.csv")
    u_train = str(data_dir / "u_train.npz")
    u_val = str(data_dir / "u_val.npz")
    u_prof = str(data_dir / "u_test_profile_ood.npz")
    u_mis = str(data_dir / "u_test_mismatch_ood.npz")

    out_dir = Path(args.outputs_dir)
    out_dir.mkdir(exist_ok=True)

    summary = {"arch": args.arch, "H": args.H, "modes": args.modes, "seeds": args.seeds,
               "runs": [], "aggregates": {}}

    for mode in args.modes:
        for seed in args.seeds:
            metrics_path = out_dir / f"metrics_{args.arch}_{mode}_seed{seed}_H{args.H}.json"

            if args.skip_if_exists and metrics_path.exists():
                m = load_json(metrics_path)
                summary["runs"].append({"mode": mode, "seed": seed,
                                         "metrics_path": str(metrics_path), "metrics": m,
                                         "skipped": True})
                continue

            train_cmd = [
                args.python, "train_transformer_hybrid.py",
                "--arch", args.arch,
                "--mode", mode,
                "--seed", str(seed),
                "--H", str(args.H),
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--lr", str(args.lr),
                "--stride", str(args.stride),
                "--num_workers", str(args.num_workers),
                "--cls_tail_frac", str(args.cls_tail_frac),
                "--meta_csv", meta_csv,
                "--u_train", u_train,
                "--u_val", u_val,
                "--save_dir", args.ckpt_dir,
            ]

            eval_cmd = [
                args.python, "eval_transformer_hybrid.py",
                "--arch", args.arch,
                "--mode", mode,
                "--seed", str(seed),
                "--H", str(args.H),
                "--batch_size", "32",
                "--num_workers", str(args.num_workers),
                "--cls_tail_frac", str(args.cls_tail_frac),
                "--meta_csv", meta_csv,
                "--u_val", u_val,
                "--u_test_profile", u_prof,
                "--u_test_mismatch", u_mis,
                "--ckpt_dir", args.ckpt_dir,
                "--save_metrics",
                "--out_metrics", str(metrics_path),
            ]

            try:
                run(train_cmd, dry=args.dry_run)
                run(eval_cmd, dry=args.dry_run)
                m = load_json(metrics_path)
                summary["runs"].append({"mode": mode, "seed": seed,
                                         "metrics_path": str(metrics_path), "metrics": m})
            except subprocess.CalledProcessError as e:
                summary["runs"].append({"mode": mode, "seed": seed,
                                         "metrics_path": str(metrics_path),
                                         "metrics": None, "error": str(e)})
                if not args.continue_on_error:
                    raise

        runs = [r for r in summary["runs"] if r["mode"] == mode and r.get("metrics") is not None]
        agg = {}
        if runs:
            keys = set()
            for r in runs:
                keys |= set(dotted_keys(r["metrics"]))
            for k in sorted(keys):
                vals = []
                for r in runs:
                    v = get(r["metrics"], k)
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        vals.append(float(v))
                if vals:
                    agg[k] = {"mean": mean(vals), "std": pstdev(vals) if len(vals) > 1 else 0.0, "n": len(vals)}
        summary["aggregates"][mode] = agg

    out_path = out_dir / "summary_hybrid_all.json"
    if not args.dry_run:
        out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary to {out_path}")


if __name__ == "__main__":
    main()
    
