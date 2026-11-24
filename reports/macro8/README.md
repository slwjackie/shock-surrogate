# Can one learned update beat eight Godunov steps?

On 30 held-out states, a width-16 flux MLP took 8.29 times as long as eight optimized Godunov steps. Its mean cell error against the PDE reference was 32.3% higher. A different model that keeps the Godunov flux and learns five integration coefficients did beat eight Godunov steps. A fixed-coefficient method achieved the same speedup, while two larger, CFL-stable Godunov steps gave the lowest mean PDE error and runtime among the tested methods.

## Equal-time speed and accuracy

Each input is a single periodic 128-cell FP64 Burgers state on a domain of length one. The original step has dt/dx = 0.2. Eight original steps reach t = 0.0125. All entries in both tables use the same 30 held-out initial states, with ten smooth, ten evolved shock states and ten Riemann states. The frozen models use separate training seeds.

| Eight-step equivalent | Single-core C time (µs) | Mean cell error vs entropy solution | Cases within 1.05 × original Godunov error | Mean cell difference from eight original Godunov steps |
|---|---:|---:|---:|---:|
| Godunov, eight steps at dt/dx = 0.2 | 0.311 | 0.0015440 | 30/30 | 0 |
| One learned MLP update | 2.578 | 0.0020423 | 13/30 | 0.0009644 |
| One three-stage update, five fitted weights | 0.189 | 0.0015079 | 30/30 | 0.0000825 |
| One three-stage update, fixed weights | 0.188 | 0.0014999 | 30/30 | 0.0001960 |
| Godunov, two steps at dt/dx = 0.8 | 0.084 | 0.0009841 | 30/30 | 0.0006131 |

| 64-step equivalent, t = 0.1 | Single-core C time (µs) | Mean cell error vs entropy solution | Cases within 1.05 × original Godunov error | Mean cell difference from 64 original steps |
|---|---:|---:|---:|---:|
| Godunov, 64 original steps | 2.426 | 0.0067299 | 30/30 | 0 |
| Eight learned MLP updates | 20.439 | 0.0096159 | 6/30 | 0.0052445 |
| Eight fitted three-stage updates | 1.433 | 0.0065348 | 30/30 | 0.0002381 |
| Eight fixed three-stage updates | 1.434 | 0.0067430 | 30/30 | 0.0000544 |
| Godunov, 16 larger steps | 0.607 | 0.0047338 | 30/30 | 0.0023465 |

At the first endpoint the MLP passed the 5% tolerance on 10/10 smooth, 0/10 evolved shock, and 3/10 Riemann cases. Models are applied recursively from their own outputs for longer runs. The 60-step comparison uses seven macro updates and four original Godunov steps, reaching t = 0.09375 exactly. The larger-step control uses 15 steps there. Results at all three horizons appear in the JSON files.

### Relation to the original certified-routing objective

The Stage A2 budgets measure distance to the original same-grid Godunov trajectory. On the nine earlier confirmation states all nine fitted three-stage runs and all nine fixed three-stage runs stayed within their existing total budgets at each available endpoint through step 60. The MLP did so on seven of nine. At step 60 alone the larger-step Godunov method met eight of nine same-grid budgets even though it gave lower PDE error than the original method in all nine cases. No intermediate states are emitted inside an eight-step update. These are observed endpoint discrepancies. A proof for an adaptive long-step solver would need bounds suitable for the long-step update and the error budget used by the routing policy.

## Models, references and measurement

- The MLP forms ten inputs from eight adjacent cell values at each face, uses a width-16 ReLU layer, and predicts a correction to the instantaneous Godunov flux. The corrected time-average flux updates all cells with one conservative divergence. Both dense layers were trained against eight-step mean-flux labels. Parameter count is 193 including biases. Training used seeds 78351/78352 and an independent development set used seed 51771.
- The fitted three-stage method evaluates the usual Burgers flux three times on intermediate states. Five trainable scalars combine the resulting flux differences. It was fitted to eight-step same-grid states with seeds 1100/1101. The fixed control uses the same three-stage implementation and determines its coefficients by matching terms through cubic order in the smooth expansion of the eight discrete updates. The match does not presume a smooth solution at shocks.
- Entropy-reference errors compare cell averages to the Hopf–Lax solution for the *same piecewise-constant initial state*, calculated in FP64. The reference is independent of the labels used for learning. The threshold is per case, with an absolute roundoff allowance of 1e-12. Case means and the count within tolerance describe different aspects of the results. Derivation of the reference formula appears in [Menon and Srinivasan, section 1.1](https://www.dam.brown.edu/people/menon/publications/ms1-jsp.pdf).
- Unit tests cover constants, a translating shock plus a rarefaction, periodicity, mass conservation, and convergence of independent 8× and 16× refined Godunov calculations. All original-timestep algebraic NumPy Godunov endpoints agree bit-for-bit with the public solver on these 39 inputs. The largest difference between C and NumPy across all methods, cases and horizons was 1.55e-13.
- Timings include a full 128-cell state update with periodic boundaries on an AMD EPYC 9V74, one pinned logical CPU, GCC 14.2, FP64, with contraction and fast-math disabled. The native benchmark warms each method and interleaves them in randomized order over 21 batches per input. The table reports the arithmetic mean of the individual case medians for seed 79023. The native JSON also summarizes all 39 cases together. The initial state and weights are hot in cache. Offline training, initial-condition generation, reference evaluation, verification and Python dispatch are outside the native timing.
- Implementation matters. On nine earlier confirmation cases, the NumPy median time for the eight-step MLP was about 60 µs, versus 333 µs for the public Godunov routine and 42 µs for its algebraically equivalent vectorized kernel. The native comparison uses compiled kernels on both sides. Target hardware, other grid sizes and other CFL regimes require their own measurements.

## Reproduction

Run from the repository root with Python, NumPy and GCC on Linux. The saved weights suffice for evaluation. Use an available core number in place of 6 for the native command.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest discover -s tests/stage_ab -p test_burgers_entropy_reference.py
python -m stage_ab.experiments_macro8
taskset -c 6 python reports/macro8/native/benchmark.py
```

Optional training commands are `python -m stage_ab.macro_network` and `python -m stage_ab.macro_flux`. Fitting the three-stage weights requires SciPy.

[Per-case errors, budget checks and NumPy timings](evaluation.json) · [Native timings, model hashes and parity checks](native/timing.json)
