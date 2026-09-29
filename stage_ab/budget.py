"""Online allocation of a certified global error budget.

At step n the router offers a menu of options j = (value v_nj, weight w_nj):
v = number of neural (trusted) faces, w = certified excess error charged
against the global budget B.  Option 0 is always (0, 0) (all-Godunov step,
whose unavoidable roundoff is reserved separately).  Choosing one option per
step with sum of weights <= B is the online multiple-choice knapsack problem.

Policies below only see the current menu.  Guarantees are instance-wise, i.e.
relative to the hindsight optimum on the realized sequence of menus
(docs/stage_a2_theory.md, Section 5).
"""
from __future__ import annotations

import math
import numpy as np


class Policy:
    name = "policy"

    def reset(self, budget: float, steps: int):
        if budget < 0 or steps < 1:
            raise ValueError("Nonnegative budget and positive horizon required")
        self.budget, self.steps = float(budget), int(steps)
        self.used, self.t = 0.0, 0

    @property
    def remaining(self):
        return max(self.budget - self.used, 0.0)

    def feasible(self, weights):
        return weights <= self.remaining

    def propose(self, values, weights) -> int:
        raise NotImplementedError

    def commit(self, idx, weight):
        self.used += float(weight)
        self.t += 1


def _argmax_value(values, mask):
    if not mask.any():
        return 0
    cand = np.flatnonzero(mask)
    return int(cand[np.argmax(values[cand])])


class GreedyPolicy(Policy):
    """Largest value that fits the remaining budget (optionally a per-step cap).

    With ``cap`` this is the Stage A rule 'accept if bound <= step tolerance and
    the cumulative budget allows it', generalized to menus.
    """

    def __init__(self, cap: float | None = None):
        self.cap = cap
        self.name = "greedy" if cap is None else "greedy_capped"

    def propose(self, values, weights):
        mask = self.feasible(weights)
        if self.cap is not None:
            mask &= weights <= self.cap
        return _argmax_value(values, mask)


class AllOrNothingPolicy(GreedyPolicy):
    """Original binary decision: all faces neural or none."""

    def __init__(self, cap: float | None = None):
        super().__init__(cap)
        self.name = "all_or_nothing"

    def propose(self, values, weights):
        last = len(values)-1
        ok = weights[last] <= self.remaining and (self.cap is None or weights[last] <= self.cap)
        return last if ok else 0


class PacingPolicy(Policy):
    """Adaptive uniform pacing: allowance = remaining / steps left."""
    name = "pacing"

    def propose(self, values, weights):
        left = max(self.steps - self.t, 1)
        return _argmax_value(values, weights <= self.remaining/left)


class ThresholdPolicy(Policy):
    """Pseudo-utility threshold policy for online multiple-choice knapsack.

    Psi(z) = (L/e) (theta e)^z on the utilization z = used/B in [0, 1], theta=U/L.
    Chooses the feasible option maximizing v - B * int_z^{z+w/B} Psi(u) du.
    Theorem K1: LP_OPT <= e^{alpha w_hat} (alpha ALG + B L / e), alpha = 1+ln(theta),
    provided every option density v/w is <= U.  L > 0 is a free design parameter.
    """
    name = "threshold"

    def __init__(self, L: float, U: float):
        if not (0 < L <= U and math.isfinite(U)):
            raise ValueError("Need 0 < L <= U < inf")
        self.L, self.U = float(L), float(U)
        self.alpha = 1.0 + math.log(self.U/self.L)

    def psi(self, z):
        return (self.L/math.e)*np.exp(self.alpha*np.asarray(z, dtype=np.float64))

    def integral(self, z0, z1):
        """int_{z0}^{z1} Psi = (Psi(z1) - Psi(z0)) / alpha."""
        return (self.psi(z1) - self.psi(z0))/self.alpha

    def propose(self, values, weights):
        if self.budget == 0:
            return 0 if weights[0] > 0 else _argmax_value(values, weights == 0)
        z = self.used/self.budget
        z1 = z + weights/self.budget
        ok = z1 <= 1.0
        score = np.where(ok, values - self.budget*self.integral(z, np.minimum(z1, 1.0)), -np.inf)
        idx = int(np.argmax(score))
        return idx if score[idx] >= 0 else 0


class RobustifiedPolicy(Policy):
    """Run a robust threshold policy on gamma*B and any advice policy on (1-gamma)*B.

    Execute whichever proposal has larger value; charge each sub-policy its own
    proposal.  Theorem K3: value >= max(advice value on (1-gamma)B,
    robust value on gamma*B), and total executed weight <= B.
    """

    def __init__(self, robust: ThresholdPolicy, advice: Policy, gamma: float):
        if not 0 < gamma < 1:
            raise ValueError("gamma must lie in (0,1)")
        self.robust, self.advice, self.gamma = robust, advice, float(gamma)
        self.name = f"robustified({advice.name},{gamma:g})"

    def reset(self, budget, steps):
        super().reset(budget, steps)
        self.robust.reset(self.gamma*budget, steps)
        self.advice.reset((1-self.gamma)*budget, steps)

    def propose(self, values, weights):
        self._ia = self.robust.propose(values, weights)
        self._ib = self.advice.propose(values, weights)
        self._w = weights
        return self._ia if values[self._ia] >= values[self._ib] else self._ib

    def commit(self, idx, weight):
        self.robust.commit(self._ia, self._w[self._ia])
        self.advice.commit(self._ib, self._w[self._ib])
        super().commit(idx, weight)


class DensityThresholdPolicy(Policy):
    """Advice-style policy: fixed price rho (e.g. learned from calibration runs).

    Picks the feasible option maximizing v - rho*w.  With rho equal to the LP's
    optimal dual price this is the offline-optimal rule; with a wrong rho it
    has no worst-case guarantee on its own (use RobustifiedPolicy).
    """

    def __init__(self, rho: float):
        self.rho = float(rho)
        self.name = "density_threshold"

    def propose(self, values, weights):
        score = np.where(self.feasible(weights), values - self.rho*weights, -np.inf)
        idx = int(np.argmax(score))
        return idx if score[idx] >= 0 else 0


def run_policy_on_menus(policy: Policy, menus, budget: float):
    """Replay a fixed sequence of menus (values, weights) through a policy."""
    policy.reset(budget, len(menus))
    total, used, choices = 0.0, 0.0, []
    for values, weights in menus:
        values, weights = np.asarray(values, float), np.asarray(weights, float)
        j = policy.propose(values, weights)
        policy.commit(j, weights[j])
        total += values[j]
        used += weights[j]
        choices.append(j)
    return {"value": total, "weight": used, "choices": choices}


def lp_dual_bound(menus, budget: float, iters: int = 200, return_price: bool = False):
    """Upper bound on the fractional (hence integral) hindsight optimum.

    g(beta) = beta*B + sum_n max_j (v_nj - beta w_nj)^+ is an upper bound of
    the LP optimum for every beta >= 0 (weak duality) and equals it at the
    minimizer.  Bisection on the subgradient B - sum_n w_n(j*(beta)).
    With ``return_price`` also returns the minimizing beta (an estimate of the
    optimal dual price, i.e. the critical value density).
    """
    menus = [(np.asarray(v, float), np.asarray(w, float)) for v, w in menus]

    def g(beta):
        total, spend = beta*budget, 0.0
        for v, w in menus:
            s = v - beta*w
            j = int(np.argmax(s))
            if s[j] > 0:
                total += s[j]
                spend += w[j]
        return total, spend

    hi = 1.0
    for v, w in menus:
        pos = w > 0
        if pos.any():
            hi = max(hi, float(np.max(v[pos]/w[pos])))
    lo = 0.0
    best, best_beta = g(0.0)[0], 0.0
    for _ in range(iters):
        mid = 0.5*(lo+hi)
        val, spend = g(mid)
        if val < best:
            best, best_beta = val, mid
        if spend > budget:
            lo = mid
        else:
            hi = mid
    for beta in (lo, hi):
        val = g(beta)[0]
        if val < best:
            best, best_beta = val, beta
    return (best, best_beta) if return_price else best


def lp_exact(menus, budget: float):
    """Exact fractional optimum via scipy HiGHS (small instances; tests only)."""
    from scipy.optimize import linprog
    menus = [(np.asarray(v, float), np.asarray(w, float)) for v, w in menus]
    sizes = [len(v) for v, _ in menus]
    n = sum(sizes)
    c = -np.concatenate([v for v, _ in menus])
    A = np.zeros((len(menus)+1, n))
    b = np.zeros(len(menus)+1)
    off = 0
    for i, (v, w) in enumerate(menus):
        A[i, off:off+len(v)] = 1.0
        b[i] = 1.0
        A[-1, off:off+len(v)] = w
        off += len(v)
    b[-1] = budget
    res = linprog(c, A_ub=A, b_ub=b, bounds=(0, None), method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    return -res.fun


class AllOrNothingWrapper(Policy):
    """Restrict any policy to the options {0, N} (no spatial selectivity).

    Used to separate the value of face-selective routing from the value of the
    temporal allocation rule."""

    def __init__(self, inner: Policy):
        self.inner = inner
        self.name = f"{inner.name}_all_or_nothing"

    def reset(self, budget, steps):
        super().reset(budget, steps)
        self.inner.reset(budget, steps)

    def propose(self, values, weights):
        w = np.full_like(weights, np.inf)
        w[0], w[-1] = weights[0], weights[-1]
        return self.inner.propose(values, w)

    def commit(self, idx, weight):
        self.inner.commit(idx, weight)
        super().commit(idx, weight)
