# Certified Burgers research workflow

This is the execution checklist for the next research discussion. Tasks 1--5
produce code and data; Task 6 controls what can honestly be claimed as novel.

| Professor task | Implementation | Primary evidence | Decision it supports |
|---|---|---|---|
| 1. Godunov solver and stability | `godunov.py`, `validation.py`, `stability.py` | mass, maximum principle, TVD, shock speed, rarefaction error, sampled L1 non-expansion | Is the trusted baseline credible on the declared grid/CFL? |
| 2. One-/multi-step surrogate | `surrogate.py`, `horizon_study.py` | fixed-time H=1 versus H>1 summaries | Does skipping H solver calls create enough speed opportunity? |
| 3. Oracle local error | `experiment.py`, `rollout.py` | oracle threshold sweeps and cumulative-eta audit | Does any useful accuracy/acceptance frontier exist at all? |
| 4. Three cheap scores | `components.py`, `verifiers/` | conservation, weak residual, MC-dropout | Which observable signal is worth developing? |
| 5. Shock/OOD audit | `splits.py`, risk metrics, `counterexamples.py` | ID/OOD false accepts, shock coverage, controlled blind spots | Is a score robust enough for a theorem or selective calibration? |
| 6. Novelty table | `certified_burgers_novelty_table.md` | paper-by-paper claim matrix | What is actually missing from prior work? |

## Added comparison rules

1. Train, calibration, one-step test, and rollout test roles are disjoint and
   hashed.
2. Thresholds and empirical multipliers use calibration data only.
3. Every policy sees the same initial states, grid, fixed `dt`, and final time.
4. Every horizon uses the same `reference_steps`; only the number of neural
   decisions changes.
5. Runtime excludes research-only oracle calls for deployable gates.
6. Results retain both positive and negative outcomes, including cases where
   verification costs more than the trusted solver.

## Added error rules

- `eta` means neural advice versus the same-grid trusted macro step.
- The contraction audit means hybrid versus the same-grid all-Godunov path.
- Coarse versus refined Godunov is a discretization proxy, never exact PDE
  error.
- Reactive Euler may later reuse the empirical policy, but the scalar Burgers
  contraction theorem must not be transferred to that system.

## Go/no-go gates

| Gate | Continue when | If it fails |
|---|---|---|
| Trusted solver | no invariant/stability failure in the declared regime | fix numerics before training |
| Oracle opportunity | oracle sweep has a nontrivial low-error/high-accept region | change horizon/data/model target; do not tune verifiers |
| Cheap signal | acceptable OOD unsafe-case miss rate and verifier cost below the available solver saving | combine or replace signals |
| Claim strength | held-out empirical envelope remains stable across seeds and OOD families | call it a heuristic gate, not a certificate |
| Extension readiness | Burgers interfaces and conclusions survive at least two seeds/horizons | then add 0-D H2 or reactive Euler as an explicitly empirical extension |

The default threshold values are experimental operating points, not universal
pass/fail constants. Record the whole frontier and discuss the application
budget with the professor before fixing a target.
