"""Figures for Stage A2 (static PNGs for the report; numbers also in JSON/markdown tables)."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
MARKERS = ["o", "s", "^", "D", "v"]


def _style(plt):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 11, "axes.titlecolor": INK, "legend.frameon": False,
        "legend.labelcolor": INK2, "lines.linewidth": 2, "lines.markersize": 7})


def fig_resolution(e1, path, plt):
    kinds = [("smooth", "Smooth data"), ("shocked", "Evolved past shock formation"), ("riemann", "Riemann (piecewise constant)")]
    series = [("separable_b16", "Face-separable (original)", S1), ("table_b16", "Difference, table", S2),
              ("local", "Difference, local box", S3)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, (kind, title) in zip(axes, kinds):
        rows = [r for r in e1 if r["kind"] == kind]
        ns = sorted({r["n"] for r in rows})
        for (key, label, color), mk in zip(series, MARKERS):
            med = [np.median([r[key]/r["actual"] for r in rows if r["n"] == n]) for n in ns]
            ax.plot(ns, med, color=color, marker=mk, markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
            txt = f"{med[-1]:,.0f}×" if med[-1] >= 10 else f"{med[-1]:.2f}×"
            ax.annotate(txt, (ns[-1], med[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", color=INK2, fontsize=9)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("Cells N")
        ax.set_xlim(ns[0]/1.3, ns[-1]*2.2)
    axes[0].set_ylabel("Certified bound / exact error")
    axes[0].legend(loc="upper left")
    fig.suptitle("E1. One-step overestimation factor vs resolution (median over ICs and seeds)", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_frontier(e2f, path, plt):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    for ax, kind, title in zip(axes, ("shocked", "riemann"), ("Evolved past shock formation", "Riemann")):
        recs = [r for r in e2f if r["kind"] == kind]
        grid = np.linspace(0, 1, 101)
        for (mode, label, color), mk in zip((("separable", "Face-separable", S1), ("table", "Difference, table", S2),
                                             ("local", "Difference, local box", S3)), MARKERS):
            curves = []
            for r in recs:
                c = np.array(r[mode])
                n = len(c)-1
                frac = [np.max(np.flatnonzero(c <= max(g*c[-1], c[0])))/n for g in grid]
                curves.append(frac)
            med = np.median(np.array(curves), axis=0)
            ax.plot(grid, med, color=color, label=label, marker=mk, markevery=20,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Per-step budget / full-trust certified cost")
    axes[0].set_ylabel("Max fraction of neural faces")
    axes[0].legend(loc="lower right")
    fig.suptitle("E2. Exact trust frontier (median over ICs, N = 256); diagonal = all-or-nothing", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_trust_map(m, path, plt):
    from matplotlib.colors import ListedColormap
    trust = np.array(m["trust"])
    states = np.array(m["states"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    ax = axes[0]
    im = ax.imshow(states, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-1, vmax=1,
                   extent=[0, trust.shape[1], 0, trust.shape[0]])
    ax.set_title("State v(x, t)")
    ax.set_xlabel("Cell")
    ax.set_ylabel("Time step")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    ax = axes[1]
    ax.imshow(trust, aspect="auto", origin="lower", cmap=ListedColormap(["#e1e0d9", S1]),
              extent=[0, trust.shape[1], 0, trust.shape[0]], interpolation="nearest")
    ax.set_title("Router: neural face (blue) / Godunov face (gray)")
    ax.set_xlabel("Face")
    ax.grid(False)
    axes[0].grid(False)
    fig.suptitle(f"E2. Space-time trust map, per-step allowance = {m['budget_fraction']:.0%} of full-trust cost, local certificate",
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_policies(e3, path, plt):
    fracs = sorted({r["fraction"] for r in e3})
    fig, axes = plt.subplots(len(fracs), 1, figsize=(8.5, 3.3*len(fracs)))
    for ax, f in zip(np.atleast_1d(axes), fracs):
        agg = collections.defaultdict(list)
        for r in e3:
            if r["fraction"] == f:
                agg[r["policy"]].append(r["empirical_ratio"])
        names = sorted(agg, key=lambda k: np.median(agg[k]))
        med = [np.median(agg[k]) for k in names]
        y = np.arange(len(names))
        ax.barh(y, [m-1 for m in med], left=1, height=0.55, color=S1)
        for yi, mv in zip(y, med):
            ax.text(mv+0.015, yi, f"{mv:.2f}", va="center", color=INK2, fontsize=9)
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_title(f"Total budget = {f:.0%} of the full-trust certified cost", loc="left")
        ax.set_xlim(1, max(med)*1.1)
    np.atleast_1d(axes)[-1].set_xlabel("Hindsight LP bound / neural faces used (1 = optimal)")
    fig.suptitle("E3. Online budget policies on certified Burgers rollouts (median over test ICs)", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_tradeoff(e3, e4, path, plt, theta=1000, fraction=0.3):
    worst = collections.defaultdict(float)
    for r in e4:
        if r["theta"] == theta:
            worst[r["policy"]] = max(worst[r["policy"]], r["ratio"])
    typical = collections.defaultdict(list)
    for r in e3:
        if r["fraction"] == fraction:
            typical[r["policy"]].append(r["empirical_ratio"])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    names = [k for k in worst if k in typical]
    for k in names:
        x, y = worst[k], np.median(typical[k])
        ax.plot([x], [y], marker="o", color=S1, markeredgecolor=SURFACE, markeredgewidth=2, markersize=9, linestyle="none")
        ax.annotate(k, (x, y), xytext=(7, 4), textcoords="offset points", color=INK2, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel(f"Worst ratio on adversarial menus (θ = {theta})")
    ax.set_ylabel("Median LP / ALG on Burgers rollouts")
    ax.set_title("E4. Consistency–robustness trade-off of budget policies", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_timing(e5, path, plt):
    row = e5[0]
    keys = [("godunov_step", "Godunov step (FP64)"), ("neural_step", "Neural step (frozen NN)"),
            ("cert_separable", "Certificate: face-separable"), ("cert_table", "Certificate: difference, table"),
            ("cert_local", "Certificate: difference, local box"), ("frontier_dp", "Trust frontier DP O(N²)"),
            ("mixed_step_with_fp_bound", "Mixed step + FP bound")]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    vals = [row[k]*1e3 for k, _ in keys]
    y = np.arange(len(keys))
    ax.barh(y, vals, height=0.55, color=S1)
    for yi, v in zip(y, vals):
        ax.text(v*1.08, yi, f"{v:.3g} ms", va="center", color=INK2, fontsize=9)
    ax.set_yticks(y, [label for _, label in keys])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(min(vals)/2, max(vals)*6)
    ax.set_xlabel("Median wall time per step [ms], N = %d, one CPU thread" % row["n"])
    fig.suptitle("E5. Cost per step (Python/NumPy prototype; no speedup claimed)", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def export(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    d = Path(directory)
    r = json.loads((d/"results.json").read_text())
    files = []
    for name, fn, args in (("fig_e1_resolution.png", fig_resolution, (r["e1"],)),
                           ("fig_e2_frontier.png", fig_frontier, (r["e2_frontier"],)),
                           ("fig_e2_trust_map.png", fig_trust_map, (r["e2_map"],)),
                           ("fig_e3_policies.png", fig_policies, (r["e3"],)),
                           ("fig_e4_tradeoff.png", fig_tradeoff, (r["e3"], r["e4"])),
                           ("fig_e5_timing.png", fig_timing, (r["e5"],))):
        fn(*args, d/name, plt)
        files.append(str(d/name))
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory")
    print(json.dumps(export(ap.parse_args().directory), indent=1))


if __name__ == "__main__":
    main()
