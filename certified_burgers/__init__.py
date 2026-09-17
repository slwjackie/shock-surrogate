"""Learning-augmented trust-or-fallback testbed for 1-D inviscid Burgers."""
from .godunov import burgers_flux, godunov_flux, godunov_step, advance, stable_dt
from .interfaces import Proposal, VerifierResult, Decision, ThresholdPolicy
from .oracle_error import l1_error, oracle_gate

__all__ = [
    "burgers_flux", "godunov_flux", "godunov_step", "advance", "stable_dt",
    "l1_error", "oracle_gate", "Proposal", "VerifierResult", "Decision",
    "ThresholdPolicy",
]
