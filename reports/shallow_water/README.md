# Certified proposals for the middle depth of a shallow-water Riemann problem

This experiment solves a local nonlinear equation at each cell interface. It compares classical guesses and a small neural network under the same absolute error tolerance in metres. The hybrid uses a deterministic check and corrects rejected guesses. Offline training and reference computations are separate from inference. The existing Burgers certificates apply to their original scalar problem.

## What is certified

The inputs are two depths and two velocities on a flat bed, with positive gravity. Both sides must be wet and the middle state must be wet. For side depth k, the wave curve at trial depth x is

```text
wave(x,k) = 2 (sqrt(g x) - sqrt(g k))                 for 0 <= x <= k
          = (x-k) sqrt((g/2) (1/x + 1/k))            for x > k
phi(x)    = wave(x,hL) + wave(x,hR) + uR - uL.
```

Each branch is continuous and strictly increasing for positive x. If phi(0) < 0 the wet root h* is unique. For a proposed positive depth z and tolerance e, strict negative and positive signs at the respective ends of [max(0,z−e), z+e] suffice to establish |z−h*| ≤ e. Equal input states admit an exact identity check. Every successful route meets the same accuracy condition, whether the initial guess came from a formula or a network. This is a bound on the middle water depth for the represented FP64 inputs. The cell-interface flux and the full solution trajectory have different errors.

The native code rounds test endpoints inward and first checks whether a forward roundoff error bound fixes the sign of the residual. Uncertain signs are evaluated with outward-rounded bounds on elementary operations. An ambiguous sign cannot certify a proposal. The argument assumes IEEE binary64 arithmetic, round-to-nearest, gradual underflow, correctly rounded basic operations and square root, and compilation without reassociation, fast math or fused contraction. The numerical audits provide an independent check of the implementation under these assumptions. See [certificate.h](../../shallow_water/certificate.h).

## Methods

The solver shares its analytical treatment of equal states and two-rarefaction inputs across methods. It tries a proposal on the other faces, then uses safeguarded Newton corrections if necessary. The arithmetic mean, two-shock and two-rarefaction estimates are classical controls. An extra classical control uses an inexpensive Newton-step estimate to decide when a certificate is worth evaluating. The numerical certificate always makes the acceptance decision.

The network takes three dimensionless features constructed from the two water depths and the velocity difference. A hidden layer with 12 ReLU units predicts a correction to the two-rarefaction estimate. The model returns one depth for the whole interface; the correction vanishes for the analytic two-rarefaction branch. Initial, uniformly retrained and hard-case retrained versions use the same architecture. Training labels come from independent bisection on wet states. The hard-example replay borrows the idea of exposing errors in one round and emphasizing them in the next from bootstrap multigrid. No multigrid cycle runs inside the Riemann solver.

The positive-depth root and classical initial estimates follow [Muñoz Moncayo, Quezada de Luna and Ketcheson](https://arxiv.org/abs/2209.12235). For the wave patterns and dam-break problem, see [Clawpack's shallow-water chapter](https://www.clawpack.org/riemann_book/html/Shallow_water.html). [Zhang et al.](https://arxiv.org/abs/2603.30007) study learned Riemann solutions with physical constraints. Their hardware and workloads differ from this experiment.

## Results on one CPU core

At a 0.01 m tolerance, hard-case retraining saved 440 corrections relative to uniform retraining on 8,192 held-out interfaces with a large depth contrast. The two models started from the same weights and received 4,096 additional training rows and 3,000 optimizer steps each. Of the 32,768 mining candidates, the first network failed its certificate on 1,428. Uniform sampling selected 183 of those failures. The hard-case policy selected all 1,428, followed by 620 cases ranked by offline depth error and 2,048 random cases. Both policies had access to the same candidate data. This comparison uses one training seed.

Each row below uses 8,192 held-out faces. Corrections count faces on which the initial result required further work. The two timing columns show the first run's medians, including proposal, certificate and all corrections. The classical column picks the fastest certified classical method for each row. A second timing run used the same inputs and changed the timing order.

| Held-out interfaces | Initial NN corrections | Uniform NN corrections | Hard-case NN corrections | Hard-case NN, ns/face | Best certified classical, ns/face |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training-like wet states | 2 | 0 | 0 | 96.09 | 106.18 |
| Large depth contrast (shifted distribution) | 5,203 | 3,822 | 3,382 | 134.95 | 136.91 |
| Smooth neighbouring states | 0 | 0 | 0 | 93.32 | 68.30 |

For the large-contrast set, the two timings differed by only about 1.5%; their paired timing interquartile range in the first run included equal performance. With all three sets weighted equally, the speedup relative to the fastest single certified classical method was 1.035× in the first run and 0.999× in the repeat. At a tighter 0.001 m tolerance, the hard-case model took longer on both the large-contrast and smooth sets. Across seven certified methods, two tolerances and 24,576 distinct inputs, all 344,064 returned depths met tolerance against bisection. A further 1,386 comparisons using high-precision decimal arithmetic also met tolerance. These are repeated method/input comparisons. [Full measurements](benchmark.json)

A separate timing compared the hybrid to an engineering Newton solver that stopped when the proposed Newton step was at most 0.001 m and returned the corrected depth. With the same equal split weighting, that Newton method took 62.44 ns/face and the certified neural method took 109.81 ns/face in this run. The Newton method passed the empirical 0.01 m accuracy target on all 24,576 inputs. Its stopping rule supplies a different guarantee from the endpoint certificate. [Engineering Newton comparison](engineering_control/results.json)

## End-to-end simulations

The first-order finite-volume solver ran to 0.045 s on a flat one-metre domain with transmissive boundaries. For each case, method and mesh, the terminal mean absolute errors in both depth and discharge per unit width were measured against cell averages of the analytical Riemann solution. The two error targets for each case were 1.05 times the respective errors of a 256-cell classical run with a 1e−8 m local depth tolerance. Each entry gives the quickest eligible mesh among 128, 256, 512 and 1,024 cells, with a seven-run median wall time. The Godunov methods use a 0.01 m local depth tolerance. Rusanov and HLLE compute the flux directly.

| Initial condition | Hard-case NN | Best certified classical | HLLE | Rusanov |
| --- | ---: | ---: | ---: | ---: |
| Dam break, 4 m and 1 m at rest | 1.524 ms (256) | 1.218 ms (256) | 1.288 ms (512) | 0.707 ms (512) |
| Colliding streams | 0.749 ms (256) | 0.597 ms (256) | 0.165 ms (256) | 0.435 ms (512) |
| Separating streams | 0.670 ms (256) | 0.536 ms (256) | 0.163 ms (256) | 0.426 ms (512) |

Across 74,715 interior-face events on the three 256-cell trajectories, the classical two-rarefaction guess needed only two Newton updates, both in the dam break. The hard-case network removed those two updates and evaluated the MLP 25,817 times. The rest of the face events either bypassed the middle-depth solve or used a shared analytic formula. This count was measured in a separate run, with identical terminal states for the instrumented and production code. The neural solver took about 25% longer than the best certified classical method in each case. These findings apply to the tested wet states, terminal error targets, meshes and single-core implementation. [Rollout timings and errors](rollout.json) · [Route counts](rollout_routes.json)

## Reproduction

Run from the repository root with Python, NumPy and a C11 compiler. The examples below use the saved model weights.

```bash
python -m shallow_water 4 0 1 0 --tol 0.01
python -m unittest discover -s tests/shallow_water -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m shallow_water.benchmark
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m shallow_water.benchmark --repeat-timing --rounds 31
python -m shallow_water.rollout_study
python -m shallow_water.rollout_routes
python reports/shallow_water/engineering_control/compare.py
```

To train anew and regenerate the corresponding C header, run `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m shallow_water.model` before the benchmark. Seeds, optimizer steps and development outcomes are recorded in [training.json](training.json). Repeated per-face timings, input/output hashes, hardware configuration, error checks and route counts are recorded in [benchmark.json](benchmark.json).
