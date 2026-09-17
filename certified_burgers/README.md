# Certified Burgers verifier study

This package is the scalar-conservation-law testbed for a learning-augmented
time stepper. It follows the professor's recommended order:

1. validate a 1-D Burgers Godunov solver;
2. train deliberately small one-step and multi-step surrogates;
3. measure the oracle local advice error;
4. implement conservation, weak-residual, and uncertainty scores;
5. audit prediction, empirical coverage, shocks, and OOD failures;
6. maintain a claim/novelty table before selecting the next theorem.

The reactive Burgers/WENO code elsewhere in the repository is not used by
this experiment.

## Mathematical object being tested

The trusted macro step is `S_h^H`, meaning `H` fixed-step, first-order exact
Godunov updates on grid `h`. The small CNN proposes

```text
P_H(v_t) ~= S_h^H(v_t).
```

The research-only oracle local error is

```text
eta_t = ||P_H(v_t) - S_h^H(v_t)||_(1,h).
```

For accepted neural steps, discrete L1 contraction gives the conditional
same-grid comparison

```text
||v_T - u_T^(S_h)||_(1,h) <= sum(accepted t) eta_t,
```

provided the declared fixed-step CFL assumptions continue to hold. The code
reports the CFL assumption check separately from the empirical inequality
check. This is a guarantee relative to the declared Godunov trajectory, not
the exact entropy solution.

## Verifier versus online decision policy

They are adjacent but not identical:

| Component | Role | Current implementation |
|---|---|---|
| Verifier | Maps a proposed state to observable evidence or a score | conservation defect, weak residual, MC-dropout, research-only oracle |
| Decision policy | Converts the evidence into accept or fallback | calibrated scalar threshold plus hard guards |
| Trusted stepper | Supplies the fallback update | `GodunovMacroStepper` |
| Advice model | Supplies the fast candidate | `TorchSurrogateAdapter(TinyConvSurrogate)` |

The verifier is therefore the trust signal; the online decision policy is the
rule that acts on it. The contracts live in `interfaces.py`, so a future 0-D
chemistry or reactive-Euler application can replace the physics-specific
components without rewriting the policy layer.

## Fair one-step/multi-step comparison

`reference_steps` fixes both the final physical time and the amount of trusted
fine-step evolution. Horizon `H` uses

```text
macro_steps = reference_steps / H.
```

Non-divisible horizons are rejected. For example, with `reference_steps=24`,
`H=1` makes 24 decisions while `H=4` makes 6 decisions, and both stop at the
same final time. All horizons regenerate identical train/calibration/test and
rollout splits from the same seeds, then verify their SHA-256 fingerprints.

Run one-step and multi-step training together:

```bash
python -m certified_burgers.horizon_study \
  --horizons 1 4 \
  --reference_steps 24 \
  --epochs 60 \
  --out_dir outputs/certified_burgers_horizons
```

## Error separation

The JSON result does not combine model error and numerical discretization
error into one ambiguous number.

| Output | Definition | Interpretation |
|---|---|---|
| `eta` | `||P_H(v)-S_h^H(v)||_1` | local surrogate/advice error on the same grid |
| hybrid deviation | `||v_t-u_t^(S_h)||_1` | quantity in the contraction-bound audit |
| solver discretization proxy | `||S_h^H(v)-R S_(h/r)^(Hr)(I v)||_1` | coarse versus restricted refined Godunov; not exact PDE error |
| advice versus refined proxy | `||P_H(v)-R S_(h/r)^(Hr)(I v)||_1` | combined practical discrepancy |

Here `I` is piecewise-constant prolongation and `R` is cell-average
restriction. The code also checks the corresponding triangle inequality.

## Cheap-score and failure audit

Thresholds and empirical multipliers are fitted only on the calibration split.
ID and deliberately shifted OOD test sets report:

- Pearson and Spearman association with `eta`;
- top-10% high-error capture;
- raw and calibrated empirical upper coverage;
- coverage split into shock-heavy and non-shock states;
- false acceptance among accepted proposals;
- fraction of all cases that are both unsafe and accepted;
- unsafe-case miss rate.

The experiment also applies controlled corruptions: a mass-preserving shock
shift, over-smoothing, a local oscillation, and a global bias. A separate
shared-bias ensemble probe shows why zero ensemble spread need not imply zero
error. These probes expose blind spots; they do not estimate how often the
trained model fails.

All three cheap scores are empirical proxies. A calibration relation such as
`eta <= C q` is reported as held-out coverage and is not called a certificate.

## Baselines and runtime

Every ID/OOD rollout evaluates the same six policies:

1. Always Solver;
2. Always Neural;
3. Oracle Gate;
4. Conservation Gate;
5. Residual Gate;
6. Uncertainty Gate.

Runtime measurements exclude research-only Godunov calls used to reveal
`eta` for deployable gates. Oracle Gate includes that call and is explicitly
non-deployable. Primitive timings report `c_S`, `c_N`, and each `c_V`, plus the
minimum acceptance fraction required by

```text
c_N + c_V + (1-a)c_S < c_S.
```

This catches a crucial negative result: a gate cannot provide speedup on a
given workload if verification plus neural inference already costs at least a
trusted step.

## Run a single horizon

Install the compact experiment dependencies:

```bash
python -m pip install -r requirements-certified-burgers.txt
```

Then run:

```bash
python -m certified_burgers.experiment \
  --horizon 1 \
  --reference_steps 24 \
  --epochs 60 \
  --out outputs/certified_burgers_h1.json
```

Use `--no_plots` for a faster smoke run. `--rollout_steps` remains a deprecated
CLI alias for `--reference_steps`; it now has fixed-final-time semantics.

## Output and go/no-go decision

The JSON contains:

- the exact comparison contract and software versions;
- data-role hashes and cross-split overlap audit;
- Godunov mass, maximum-principle, TVD, shock, rarefaction, and L1 stability checks;
- local ID/OOD verifier audits;
- selected thresholds and full threshold sweeps;
- autoregressive errors, fallback rates, guards, theorem audit, and runtime;
- controlled verifier failure probes;
- explicit claim boundaries.

The next research decision should be based on the oracle curve first. If the
Oracle Gate itself has no useful error-versus-acceptance region, verifier
engineering cannot create it. If the oracle curve is useful but cheap scores
have high OOD unsafe-accept rates, improve the trust signal or use selective
calibration. Only after a cheap signal survives that audit should it be taken
to a 0-D chemistry or reactive-Euler empirical extension.

## Tests

```bash
python -m pytest -q
```

The claim/novelty audit template is in
[`docs/certified_burgers_novelty_table.md`](../docs/certified_burgers_novelty_table.md).
