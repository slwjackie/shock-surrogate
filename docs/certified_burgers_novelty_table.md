# One-page novelty and claim audit

This table is intentionally a claim-control artifact, not evidence that novelty
has already been established. Add the exact papers from the professor's reading
list, with page/theorem references, before writing a novelty sentence.

| Work | Prediction object | Trust signal | Fallback mechanism | Theorem / guarantee | PDE class | What remains missing relative to this project |
|---|---|---|---|---|---|---|
| **This branch: current Burgers PoC** | full next state `P_H(v_t)` after H fine steps | conservation defect, smooth weak residual, MC-dropout; oracle `eta_t` only for evaluation | global per-macro-step fallback to first-order finite-volume Godunov with the exact Burgers Riemann flux | conditional accumulated local-error bound from discrete L1 contraction; no cheap-score certificate | scalar inviscid Burgers | prove or reject a cheap bound; multi-seed evidence; stronger OOD families; useful wall-clock regime |
| **Planned 0-D H2 extension** | reactor state after a chemical integration interval | positivity, element/mass/energy defects, stiffness/uncertainty signals | trusted stiff ODE integrator | no Burgers L1-contraction transfer | stiff chemical ODE | define a system-appropriate stability/error argument; detailed-chemistry validation |
| **Planned reactive-Euler extension** | conservative reacting-flow state after a macro step | positivity, conservation, entropy/RH/reaction residuals, uncertainty | trusted shock-capturing reactive solver | empirical only unless a new system-specific result is proved | reacting hyperbolic balance-law system | no global scalar L1 contraction; conservative local mixing; shock/ignition validation |
| Professor paper 1: **TODO exact citation** | TODO | TODO | TODO | TODO theorem number and assumptions | TODO | TODO |
| Professor paper 2: **TODO exact citation** | TODO | TODO | TODO | TODO theorem number and assumptions | TODO | TODO |
| Professor paper 3: **TODO exact citation** | TODO | TODO | TODO | TODO theorem number and assumptions | TODO | TODO |
| Professor paper 4: **TODO exact citation** | TODO | TODO | TODO | TODO theorem number and assumptions | TODO | TODO |

## Candidate novelty to test, not yet claim

The strongest candidate is not “using uncertainty.” It is the conjunction of:

1. an online learning-augmented time stepper for shock-forming dynamics;
2. a trusted fallback with a scalar-conservation-law accumulated error budget;
3. a systematic oracle-opportunity test before verifier design;
4. shock- and OOD-conditioned audits of cheap physics/statistical trust signals;
5. an end-to-end error/fallback/runtime frontier that preserves negative
   results and verifier counterexamples.

The contribution becomes weaker if prior work already contains the same
prediction object, trust signal, fallback granularity, guarantee, and PDE
setting. It becomes stronger if the implemented cheap signal can be shown to
upper-bound local advice error under explicit assumptions, or if the study
establishes a clear impossibility/failure result for common cheap signals.

## Evidence required before using a novelty sentence

- exact citation and stable link for every closest work;
- theorem number, norm, probability statement, and assumptions;
- whether the reference solver is actually skipped on accepted steps;
- whether evaluation is autoregressive and includes shocks plus OOD;
- whether “uncertainty” predicts error, upper-bounds it, or only correlates;
- whether runtime includes the verifier and fallback overhead;
- a final row stating one precise gap no reviewed paper fills.
