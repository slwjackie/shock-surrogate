# Certified Burgers PoC

Minimal theory testbed for:

> Can a cheap observable score predict or certify the local advice error
> `eta = ||P_theta(u) - S(u)||_1` without executing trusted solver `S`?

It is intentionally separate from the reactive Burgers/WENO pipeline.

## Reference problem
- PDE: `u_t + (u^2/2)_x = 0`
- trusted `S`: first-order exact Godunov finite-volume step
- default boundary: periodic
- fixed CFL-safe time step

## Advice model
`TinyConvSurrogate` is deliberately small. Use `--horizon H` to train one
neural call against `H` Godunov steps.

## Oracle and candidate verifier scores
The experiment computes the oracle diagnostic
`eta_t = ||P_theta(u_t) - S^H(u_t)||_1`.
It is not deployable because it requires the reference solver.

Three cheap scores are audited:
1. conservation defect,
2. smooth-test-function weak residual proxy,
3. MC-dropout uncertainty.

These are **proxies, not certificates** until a theorem is proved.

## Run
```bash
python -m certified_burgers.experiment --epochs 60 --horizon 1 \
  --out outputs/certified_burgers_poc.json
```

Multi-step leap:
```bash
python -m certified_burgers.experiment --epochs 60 --horizon 4 \
  --out outputs/certified_burgers_h4.json
```

Inspect ID and stronger-shock OOD results for:
- mean/p95 oracle error,
- oracle-gate fallback rate and hybrid error,
- Pearson correlation of each score with eta,
- top-10%-error capture.

If the oracle gate gives no useful tradeoff, do not over-invest in a verifier.
If a cheap score tracks eta, the next step is a restricted deterministic or
high-probability certificate theorem.
