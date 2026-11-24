# Selective local certification on Burgers

This experiment fixes the number of expensive local checks at ten per step and
changes which cells receive them. It uses the frozen Stage A2 flux network,
128 periodic cells, 60 steps, a 16-bin table, and the original E3 pacing policy.
For each initial condition, all six policies share an error budget equal to
30% of the accumulated certificate in an all-neural, full-local reference run.
The error budget measures deviation from the exact-arithmetic Godunov solution
on the same grid.

## Results

Neural use is the fraction of all 128 face decisions across all 60 steps that
the router certified for the neural flux. Each split has nine initial
conditions, with three smooth waves, three evolved waves with shocks, and
three Riemann states.

| Method | Local cells per step | E3 neural use | E3 time | Fresh neural use | Fresh time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Table only | 0 | 1.35% | 0.270 s | 1.72% | 0.272 s |
| Largest table bounds | 10 | 14.12% | 0.736 s | 16.90% | 0.737 s |
| Fixed contiguous-run rule | 10 | 33.08% | 0.765 s | 33.30% | 0.772 s |
| Learned, first round | 10 | 25.93% | 0.777 s | 26.76% | 0.791 s |
| Learned, second round | 10 | 27.53% | 0.788 s | 26.79% | 0.787 s |
| Full local | 128 | 68.87% | 1.031 s | 68.27% | 1.046 s |

The second training round added 1.60 percentage points on E3 and 0.03 points
on the fresh set. Both learned models outperformed the simple top-ten rule
in accepted faces. The fixed contiguous-run rule outperformed both trained
models in mean acceptance on both splits. On the fresh Riemann states it
accepted 75.4% of faces, compared with 56.0% for round two. On the other two
fresh families the differences were under one percentage point.

The second-round selector took 24.8% less time than checking all 128 cells on
the fresh set and 6.7% more time than the simple top-ten rule. Plain Godunov
took 0.00276 s per E3 rollout and 0.00290 s per fresh rollout. Offline training,
including training reference runs and label generation, took 26.3 s on this
machine. The timing measurements and the setup cost refer to this Python/NumPy
implementation.

All 108 distinct policy/initial-condition combinations respected their budgets.
Across six separately audited second-round rollouts, all 360 one-step checks
passed against exact one-step errors. All 360 cumulative checks passed against
the floating-point Godunov reference with its own roundoff allowance included.

## How the selector learns

1. On 12 training initial conditions (seed 1100), save eight Godunov states
   from each trajectory. At each state, evaluate the table and local
   certificates on the entire grid. Fit histograms to the log ratio of the
   tightened bound to the table bound, using local state, variation, and table
   costs. This produces labels for 12,288 cells in round one.
2. Run the first selector on the same 12 training initial conditions and collect
   eight states per trajectory again. Use the actual full-grid local costs to
   find which cells the candidate-run search would have selected with hindsight.
   Give the missed cells five times the usual weight and update the histograms,
   retaining round-one observations.
3. At evaluation time, compute all table bounds and use the fitted histograms
   to rank refinements. The selector tries candidate contiguous runs of neural
   faces on the periodic grid. It considers separate bounds for the start,
   interior, and end of each run. Its target allowance is the initial average
   per-step budget. The downstream router uses its actual remaining budget and
   may choose any face pattern. Only the ten chosen cells receive local checks.

The fixed block rule uses the same candidate runs and target allowance as the
trained selector. It scores a possible check by assuming it could remove the
whole table cost at the selected row. The simple top-ten rule ranks only the
table cost for using the network on both faces of a cell. In every selective
policy, the router receives the actual certified bounds on the chosen cells
and table bounds elsewhere. The learned estimates only influence cell selection.

## Reproduction

Run from the repository root with the Stage AB dependencies installed.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m stage_ab.experiments_refinement --out outputs/selective_refinement --repeats 2
python -m stage_ab.refinement_controls --out outputs/selective_refinement/oracle_first_steps.json
python -m pytest -q
```

The evaluation JSON contains the two seeds, each initial condition, its error
budget, accepted face count, time samples, and audit outcomes. Each reported
certified runtime is the mean over initial conditions of the median of two
warmed-up 60-step CPU rollouts. Those timings include selection, local checks,
routing, and updates. The separate plain Godunov timing takes the median of
nine runs for each of the same initial conditions. Table construction, offline
training, reference budget calibration, and audits are outside the online
timers. Both test sets use the training generator with distinct seeds. The
results characterize those three wave families at this one resolution.
