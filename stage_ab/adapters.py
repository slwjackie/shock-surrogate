"""Adapters connecting the new certificate to the original PoC contracts."""
from fractions import Fraction
import numpy as np
from certified_burgers.interfaces import VerifierResult, Decision
from .burgers import certified_proposal, neural_step
from .interval import upper_fraction


class CertifiedFluxAdvice:
    def __init__(self, model, lam):
        self.model, self.lam = model, float(lam)
    def predict(self, state):
        return neural_step(self.model, np.asarray(state,dtype=float), self.lam)[0]


class CertifiedFluxVerifier:
    """Recomputes the frozen proposal to bind a certificate to THIS candidate.

    It never computes a Godunov step. Repeated inference is deliberately included
    in adapter cost; the optimized stage_ab.burgers runner shares the proposal.
    """
    name = "certified_flux"
    def __init__(self, model, table, lam, h):
        self.model,self.table,self.lam,self.h = model,table,float(lam),float(h)
    def evaluate(self, proposal):
        if proposal.horizon != 1 or proposal.elapsed_time != self.lam*self.h:
            return VerifierResult(self.name,float("inf"),True,{"reason":"time_contract"})
        try:
            candidate,bound,rounding=certified_proposal(self.model,self.table,proposal.state,self.lam,self.h)
            if not np.array_equal(candidate,proposal.candidate):
                raise ValueError("Candidate is not the certified frozen model update")
            maximum=max(abs(self.table.edges[0]),abs(self.table.edges[-1]))
            if np.max(np.abs(candidate))>maximum or Fraction(self.lam)*Fraction(float(maximum))>1:
                raise ValueError("State envelope/CFL contract")
            return VerifierResult(self.name,upper_fraction(bound),False,
                                  {"bound_fraction":str(bound),"roundoff_fraction":str(rounding),
                                   "target":"same-grid exact-arithmetic Godunov"})
        except (ValueError,ArithmeticError,OverflowError) as exc:
            return VerifierResult(self.name,float("inf"),True,{"reason":str(exc)})


class ErrorBudgetPolicy:
    """An exact rational accepted-advice budget. Fallback defects are separate."""
    def __init__(self, step_tolerance, total_budget):
        if min(step_tolerance,total_budget)<0:
            raise ValueError("Nonnegative budgets required")
        self.step=Fraction(float(step_tolerance));self.total=Fraction(float(total_budget))
        self.spent=Fraction(0)
    def decide(self, result):
        if result.hard_failure or "bound_fraction" not in result.details:
            return Decision(False,"missing_or_invalid_certificate",result.score)
        bound=Fraction(result.details["bound_fraction"])
        if bound<0 or bound>self.step or self.spent+bound>self.total:
            return Decision(False,"error_budget",result.score)
        self.spent+=bound
        return Decision(True,"certified_accept",result.score)
