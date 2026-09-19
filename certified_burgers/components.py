"""Burgers implementations of the generic trust-or-fallback contracts."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .godunov import advance
from .interfaces import Proposal, VerifierResult
from .oracle_error import l1_error
from .surrogate import mc_dropout_predictions
from .verifiers import conservation_defect, weak_residual_score


@dataclass(frozen=True)
class GodunovMacroStepper:
    dx: float
    dt: float
    horizon: int = 1
    boundary: str = "periodic"

    @property
    def elapsed_time(self) -> float:
        return float(self.dt) * int(self.horizon)

    def step(self, state: np.ndarray) -> np.ndarray:
        out, _ = advance(
            state,
            self.dx,
            int(self.horizon),
            dt=self.dt,
            boundary=self.boundary,
        )
        return out


class TorchSurrogateAdapter:
    """Expose a PyTorch state predictor through the problem-independent API."""

    def __init__(self, model: torch.nn.Module):
        self.model = model

    @property
    def device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def predict(self, state: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(np.asarray(state)[None], dtype=torch.float32, device=self.device)
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            candidate = self.model(tensor)[0]
        self.model.train(was_training)
        return candidate.detach().cpu().numpy().astype(np.float64, copy=False)


@dataclass(frozen=True)
class ConservationVerifier:
    dx: float
    name: str = "conservation"

    def evaluate(self, proposal: Proposal) -> VerifierResult:
        score = conservation_defect(proposal.state[None], proposal.candidate[None], self.dx)[0]
        return VerifierResult(self.name, float(score))


@dataclass(frozen=True)
class WeakResidualVerifier:
    dx: float
    name: str = "weak_residual"

    def evaluate(self, proposal: Proposal) -> VerifierResult:
        score = weak_residual_score(
            proposal.state[None],
            proposal.candidate[None],
            dx=self.dx,
            dt=proposal.elapsed_time,
        )[0]
        return VerifierResult(self.name, float(score))


class MCDropoutVerifier:
    name = "uncertainty"

    def __init__(self, model: torch.nn.Module, samples: int = 8):
        if samples < 2:
            raise ValueError("MC-dropout requires at least two samples.")
        self.model = model
        self.samples = int(samples)

    def evaluate(self, proposal: Proposal) -> VerifierResult:
        seed = proposal.metadata.get("seed")
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        state = torch.as_tensor(proposal.state[None], dtype=torch.float32, device=device)
        draws = mc_dropout_predictions(self.model, state, samples=self.samples)
        spread = draws.std(dim=0, unbiased=True)
        score = spread.flatten(start_dim=1).mean(dim=1)[0]
        return VerifierResult(self.name, float(score.detach().cpu()))


@dataclass(frozen=True)
class OracleVerifier:
    dx: float
    name: str = "oracle"

    def evaluate(self, proposal: Proposal) -> VerifierResult:
        if proposal.reference_candidate is None:
            raise ValueError("OracleVerifier requires the trusted reference candidate.")
        score = l1_error(
            proposal.candidate[None],
            proposal.reference_candidate[None],
            self.dx,
        )[0]
        return VerifierResult(self.name, float(score), details={"deployable": False})


class GuardedVerifier:
    """Reject non-finite or out-of-envelope proposals before using a score."""

    def __init__(self, base, max_abs: float | None):
        self.base = base
        self.max_abs = None if max_abs is None else float(max_abs)
        self.name = base.name

    def evaluate(self, proposal: Proposal) -> VerifierResult:
        candidate = np.asarray(proposal.candidate)
        if not np.all(np.isfinite(candidate)):
            return VerifierResult(self.name, float("inf"), True, {"guard": "nonfinite_candidate"})
        observed = float(np.max(np.abs(candidate)))
        if self.max_abs is not None and observed > self.max_abs + 1e-12:
            return VerifierResult(
                self.name,
                float("inf"),
                True,
                {"guard": "state_envelope", "max_abs_candidate": observed, "max_abs_allowed": self.max_abs},
            )
        return self.base.evaluate(proposal)


def make_verifier(name: str, *, model, dx: float, mc_samples: int, max_abs: float | None = None):
    if name == "oracle":
        verifier = OracleVerifier(dx)
    elif name == "conservation":
        verifier = ConservationVerifier(dx)
    elif name == "residual":
        verifier = WeakResidualVerifier(dx)
    elif name == "uncertainty":
        verifier = MCDropoutVerifier(model, samples=mc_samples)
    else:
        raise ValueError(f"Unknown verifier {name!r}.")
    return GuardedVerifier(verifier, max_abs=max_abs)
