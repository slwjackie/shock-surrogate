"""Publication-oriented diagnostic plots for the PoC experiment."""
from __future__ import annotations
from pathlib import Path
import numpy as np


def _plt():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    return plt


def plot_verifier_scatter(out_dir, verifier_name, id_eta, id_score, ood_eta, ood_score,
                          id_shock=None, ood_shock=None):
    plt=_plt(); out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(6.4,4.8))
    ax.scatter(id_score,id_eta,alpha=0.5,label="ID")
    ax.scatter(ood_score,ood_eta,alpha=0.5,label="OOD")
    if id_shock is not None and np.any(id_shock): ax.scatter(np.asarray(id_score)[id_shock],np.asarray(id_eta)[id_shock],marker="x",label="ID shock-heavy")
    if ood_shock is not None and np.any(ood_shock): ax.scatter(np.asarray(ood_score)[ood_shock],np.asarray(ood_eta)[ood_shock],marker="+",label="OOD shock-heavy")
    ax.set_xlabel(f"{verifier_name} score q")
    ax.set_ylabel("oracle local error eta")
    ax.set_title(f"Verifier audit: {verifier_name}")
    ax.legend(); fig.tight_layout()
    path=out/f"scatter_{verifier_name}.png"; fig.savefig(path,dpi=180); plt.close(fig); return str(path)


def plot_error_fallback(out_dir, sweeps, tag="id"):
    plt=_plt(); out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(6.4,4.8))
    for name,rows in sweeps.items():
        x=[r["fallback_rate"] for r in rows]; y=[r["final_error"] for r in rows]
        ax.plot(x,y,marker="o",label=name)
    ax.set_xlabel("fallback rate")
    ax.set_ylabel("final L1 deviation from all-Godunov")
    ax.set_title(f"Accuracy vs fallback rate ({tag.upper()})")
    ax.legend(); fig.tight_layout(); path=out/f"error_vs_fallback_{tag}.png"; fig.savefig(path,dpi=180); plt.close(fig); return str(path)


def plot_error_speedup(out_dir, baselines, tag="id"):
    plt=_plt(); out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(6.4,4.8))
    for name,row in baselines.items():
        ax.scatter([row["speedup_vs_solver"]],[row["final_error"]],s=55)
        ax.annotate(name,(row["speedup_vs_solver"],row["final_error"]),xytext=(4,4),textcoords="offset points")
    ax.axvline(1.0,linewidth=1)
    ax.set_xlabel("actual wall-clock speedup vs Always Solver")
    ax.set_ylabel("final L1 deviation from all-Godunov")
    ax.set_title(f"Accuracy-runtime tradeoff ({tag.upper()})")
    fig.tight_layout(); path=out/f"error_vs_speedup_{tag}.png"; fig.savefig(path,dpi=180); plt.close(fig); return str(path)


def plot_theorem_bound(out_dir, rollout_result, name="representative"):
    plt=_plt(); out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(6.4,4.8))
    steps=np.arange(len(rollout_result.trajectory_deviation))
    ax.plot(steps,rollout_result.trajectory_deviation,label="observed deviation")
    ax.plot(steps,rollout_result.cumulative_accepted_eta,label="sum accepted eta")
    ax.set_xlabel("macro step")
    ax.set_ylabel("L1")
    ax.set_title(f"Theorem-1 audit: {name}")
    ax.legend(); fig.tight_layout(); path=out/f"theorem_bound_{name}.png"; fig.savefig(path,dpi=180); plt.close(fig); return str(path)
