"""Compare certified middle-depth proposals at a single wet shallow-water interface."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .benchmark import build_solver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("states", nargs=4, metavar="VALUE", type=float,
                        help="Left depth, left velocity, right depth, right velocity, in m and m/s")
    parser.add_argument("--tol", type=float, default=0.01, help="absolute middle-depth tolerance in m")
    parser.add_argument("--gravity", type=float, default=9.81, help="acceleration in m/s^2")
    args = parser.parse_args()
    states = np.array([args.states], dtype=np.float64)
    depths = np.empty(1, dtype=np.float64)
    counts = np.empty((1, 4), dtype=np.int32)
    routes = ("analytic", "accepted proposal", "corrected proposal")
    modes = ("arithmetic mean", "two-shock", "first network", "hard-case network",
             "uniform-replay network", "two-rarefaction", "filtered Newton")
    with TemporaryDirectory() as tmp:
        lib = build_solver(Path(tmp))
        for mode, name in enumerate(modes):
            failed = lib.sw_solve_batch(1, states, args.gravity, args.tol, mode, depths, counts)
            if failed:
                parser.exit(2, f"{name}: could not certify this input (status {counts[0, 0]})\n")
            route, newton, bisection, checks = counts[0]
            print(f"{name:23} {depths[0]:.12g} m   {routes[route]:19}  "
                  f"Newton {newton}  bisection {bisection}  checks {checks}")


if __name__ == "__main__":
    main()
