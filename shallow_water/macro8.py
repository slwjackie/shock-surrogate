"""Advance a 256-cell wet shallow-water field with the frozen eight-step model."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .macro8_benchmark import OUT, apply, build, read_thresholds
from .macro8_data import DURATION, N, ReferenceSolver, analytical_dam
from .rollout_study import initial


def advance_blocks(lib, state, blocks, method, thresholds, replay=None):
    """Advance from the supplied (h, hu) field; return its endpoint and route log."""
    q = np.ascontiguousarray(state, dtype=np.float64)
    if q.shape != (N, 2) or not np.isfinite(q).all() or np.any(q[:, 0] <= 0):
        raise ValueError(f"Expected {N} rows of finite (depth, discharge) with positive depth")
    if not isinstance(blocks, (int, np.integer)) or blocks < 1:
        raise ValueError("Use a positive integer block count")
    routes = []
    for _ in range(blocks):
        if replay is not None and method in ("hard", "routed_hard"):
            from .macro8_replay_native import replay_step
            replay_lib, arm = replay
            cutoff = thresholds[arm] if method == "routed_hard" else None
            q, route = replay_step(replay_lib, q, arm, cutoff)
        else:
            q, route = apply(lib, method, q, thresholds)
        if not np.isfinite(q).all() or np.any(q[:, 0] <= 0):
            raise ArithmeticError(f"{method} produced an invalid field in block {len(routes)+1}")
        routes.append(route)
    return q, routes


def plot_example(path, initial_field, blocks, lib, thresholds, replay=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        reference = analytical_dam(initial_field, blocks)
        label = "Analytical cell averages"
    except ValueError:
        with ReferenceSolver() as solver:
            reference = solver.refined(initial_field, refinement=4, blocks=blocks)
        label = "4× grid Godunov"

    classical, _ = advance_blocks(lib, initial_field, blocks, "teacher8", thresholds)
    routed, routes = advance_blocks(lib, initial_field, blocks, "routed_hard", thresholds, replay)
    try:
        bare, _ = advance_blocks(lib, initial_field, blocks, "hard", thresholds, replay)
    except ArithmeticError:
        bare = None
    x = (np.arange(N) + 0.5) / N
    # Show the entire region that has evolved, plus a margin of six cells.
    changed = np.flatnonzero(np.any(np.abs(reference-initial_field) > 1e-7, axis=1))
    xrange = (0, 1) if not len(changed) else (
        max(0., x[changed[0]]-6/N), min(1., x[changed[-1]]+6/N))
    fig, axes = plt.subplots(2, 1, figsize=(8.1, 5.6), sharex=True)
    curves = ((reference, label, "#121923", "-"), (classical, "8-step Godunov", "#007a72", "-"),
              (bare, "Learned, unchecked", "#cd7118", ":"), (routed, "Learned + guard", "#5340a8", "--"))
    for column, (ax, quantity) in enumerate(zip(axes, ("Depth (m)", "Discharge (m²/s)"))):
        for field, name, color, style in curves:
            if field is not None:
                ax.plot(x, field[:, column], label=name, color=color, linestyle=style, linewidth=1.8)
        ax.set_ylabel(quantity)
        ax.grid(True, color="#dde2e5", lw=0.6)
        ax.spines[["right", "top"]].set_visible(False)
    axes[-1].set(xlim=xrange, xlabel="Position (m)")
    fig.suptitle(f"Shallow water after {8*blocks} reference steps", x=.10, ha="left",
                 fontweight="bold", fontsize=16)
    status = f"Guard accepted {routes.count(1)} of {blocks} proposals"
    if bare is None:
        status += "   ·   Unchecked solver lost physical validity"
    fig.text(.10,.915,f"t = {blocks*DURATION:.6f} s   ·   {status}",
             color="#43515c", fontsize=10)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
    fig.subplots_adjust(left=.12, right=.98, top=.85, bottom=.19, hspace=.15)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Optional .npy array of (h, hu) rows, defaults to a 4 m / 1 m dam break")
    parser.add_argument("--out", type=Path, help="Save the final (h, hu) grid as .npy")
    parser.add_argument("--blocks", type=int, default=1, help="Each block advances 8 reference steps")
    methods = {"guarded": "routed_hard", "neural": "hard", "godunov": "teacher8",
               "hlle": "hlle_cfl", "rusanov": "rusanov_cfl"}
    parser.add_argument("--method", choices=methods, default="guarded",
                        help="guarded applies the empirical check and classical fallback")
    parser.add_argument("--weights", choices=("offline-hard", "rollout-uniform", "rollout-hard"),
                        default="rollout-uniform", help="Learned profile used by the predictor")
    parser.add_argument("--plot", type=Path, help="Also run reference methods and save a comparison figure")
    args = parser.parse_args()
    state = np.load(args.input, allow_pickle=False) if args.input else initial((4., 0., 1., 0.), N)
    with TemporaryDirectory() as directory:
        learned = args.method in ("guarded", "neural")
        replay = None
        if args.weights != "offline-hard" and (learned or args.plot):
            from .macro8_replay_native import build_replay_native
            arm = 1 if args.weights == "rollout-uniform" else 2
            replay = build_replay_native(Path(directory)), arm
            thresholds = read_thresholds(OUT / "replay_guard.json")
        else:
            thresholds = read_thresholds()
        lib = build(Path(directory)) if replay is None or args.plot or not learned else None
        final, routes = advance_blocks(lib, state, args.blocks, methods[args.method], thresholds, replay)
        if args.out:
            np.save(args.out, final)
        print(f"{args.method}  |  {N} cells  |  {8*args.blocks} reference steps  |  t = {DURATION*args.blocks:.8f} s")
        if learned:
            print(f"Predictor weights: {args.weights}")
        print(f"Depth range {final[:, 0].min():.6g} to {final[:, 0].max():.6g} m")
        if args.method == "guarded":
            print(f"Learned blocks {routes.count(1)}  |  Godunov fallbacks {routes.count(0)}")
        if args.plot:
            plot_example(args.plot, state, args.blocks, lib, thresholds, replay)


if __name__ == "__main__":
    main()
