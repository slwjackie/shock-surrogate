"""Problem-independent contracts for trust-or-fallback time stepping.

The Burgers experiment uses these contracts directly.  A later chemistry or
reactive-flow experiment can provide different stepper, advice, and verifier
implementations without changing the decision layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import numpy as np


@dataclass(frozen=True)
class Proposal:
    """One untrusted proposal together with information available to a verifier."""

    state: np.ndarray
    candidate: np.ndarray
    horizon: int
    elapsed_time: float
    reference_candidate: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerifierResult:
    """Observable verifier output; ``score`` is not automatically a certificate."""

    name: str
    score: float
    hard_failure: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    accept: bool
    reason: str
    score: float | None


class TrustedStepper(Protocol):
    def step(self, state: np.ndarray) -> np.ndarray:
        """Advance the trusted method by one declared macro step."""


class AdviceModel(Protocol):
    def predict(self, state: np.ndarray) -> np.ndarray:
        """Propose the state after the same macro-step duration."""


class Verifier(Protocol):
    name: str

    def evaluate(self, proposal: Proposal) -> VerifierResult:
        """Return an observable score without changing the state."""


class DecisionPolicy(Protocol):
    def decide(self, result: VerifierResult) -> Decision:
        """Turn verifier information into an accept/fallback action."""


@dataclass(frozen=True)
class ThresholdPolicy:
    """The deliberately simple policy used in the professor's first workflow."""

    threshold: float

    def decide(self, result: VerifierResult) -> Decision:
        if result.hard_failure:
            return Decision(False, "hard_verifier_failure", result.score)
        if not np.isfinite(result.score):
            return Decision(False, "nonfinite_verifier_score", result.score)
        accept = bool(result.score <= float(self.threshold))
        return Decision(accept, "score_below_threshold" if accept else "score_above_threshold", result.score)


@dataclass(frozen=True)
class AlwaysAcceptPolicy:
    def decide(self, result: VerifierResult) -> Decision:
        return Decision(True, "always_accept", result.score)


@dataclass(frozen=True)
class AlwaysFallbackPolicy:
    def decide(self, result: VerifierResult) -> Decision:
        return Decision(False, "always_fallback", result.score)


def apply_accept_mask(candidate, fallback, accept_mask):
    """Select candidate/fallback values with a scalar or per-cell mask.

    Current Burgers policies still return one global boolean. This helper fixes the
    interface for later cell-wise fallback experiments without changing today's
    decision semantics. A 1-D mask of length N broadcasts over leading axes.
    """
    candidate = np.asarray(candidate)
    fallback = np.asarray(fallback)
    if candidate.shape != fallback.shape:
        raise ValueError("candidate and fallback must have identical shapes.")
    mask = np.asarray(accept_mask, dtype=bool)
    if mask.ndim == 0:
        return candidate.copy() if bool(mask) else fallback.copy()
    if mask.shape == candidate.shape:
        return np.where(mask, candidate, fallback)
    if mask.ndim == 1 and candidate.shape[-1] == mask.shape[0]:
        reshape = (1,) * (candidate.ndim - 1) + (mask.shape[0],)
        return np.where(mask.reshape(reshape), candidate, fallback)
    raise ValueError("accept_mask must be scalar, full-shape, or a 1-D cell mask.")
