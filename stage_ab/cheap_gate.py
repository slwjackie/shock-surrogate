"""Offline bootstrap of a direct neural/Godunov router with one certificate per proposal.

Training labels and learned scores only propose a mask. The safety decision uses
the frozen model's anchor enclosure, the per-step spending limit and update
roundoff. The online path never evaluates a pointwise reference for verification.
"""
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from certified_burgers.godunov import godunov_flux, godunov_step
from .stage_a2 import FPConstants, exact_step_error, mixed_step, pairs, per_step_reserve
from .vinterval import rigorous_nonneg_sum, up, up_float


@dataclass
class DirectGate:
    edges: np.ndarray
    signed_error: np.ndarray
    levels: np.ndarray
    n_levels: int
    round: int

    def bins(self, state):
        i = np.clip(np.searchsorted(self.edges, state, side="right") - 1,
                    0, len(self.edges) - 2)
        return i * (len(self.edges) - 1) + np.roll(i, -1)

    def propose(self, state, max_raw_cost):
        """Choose the broadest affordable learned risk level in O(N+levels).

        As levels are activated, each cell contributes first the magnitude of
        the error on its earlier face, then the difference of the two estimated
        errors. All candidate sums follow from two weighted histograms.
        """
        bins = self.bins(state)
        level = self.levels[bins]
        error = self.signed_error[bins]
        prev_level = np.roll(level, 1)
        prev_error = np.roll(error, 1)
        first = np.minimum(prev_level, level)
        last = np.maximum(prev_level, level)
        first_cost = np.where(prev_level < level, np.abs(prev_error), np.abs(error))
        changes = np.bincount(first, weights=first_cost, minlength=self.n_levels + 1)
        changes += np.bincount(last, weights=np.abs(error - prev_error) - first_cost,
                               minlength=self.n_levels + 1)
        # The last level is reserved for unseen bins. A margin leaves room for
        # within-bin variation and the gap between true and certified errors.
        candidates = np.flatnonzero(np.cumsum(changes)[:-1] <= 0.5 * max_raw_cost)
        return level <= (candidates[-1] if len(candidates) else -1)


def train_gate(model, tables, anchor, cases, *, steps, lam):
    """Learn from training-only (initial_state, total_budget) pairs in two rounds."""
    anchor.check_contract(model, tables)
    cases = list(cases)
    if not cases or steps <= 0:
        raise ValueError("Positive training horizon and training cases required")
    size = anchor.bins**2
    counts, signed, absolute = (np.zeros(size) for _ in range(3))
    edges = anchor.edges
    n_levels = 16
    empty = DirectGate(edges, signed, np.full(size, n_levels), n_levels, 0)

    def observe(state, previous):
        x = pairs(state)
        observed = model.predict(x) - godunov_flux(x[:, 0], x[:, 1])
        bins = empty.bins(state)
        weights = np.ones(len(state))
        if previous is not None:
            learned = previous.signed_error[bins]
            # Greater weight to states where the first round missed substantial
            # signed residual, including unseen bins with large errors.
            missed = np.abs(observed - learned) > np.maximum(0.5 * np.abs(observed), 1e-12)
            weights[missed] = 5.
        counts[:] += np.bincount(bins, weights=weights, minlength=size)
        signed[:] += np.bincount(bins, weights=weights*observed, minlength=size)
        absolute[:] += np.bincount(bins, weights=weights*np.abs(observed), minlength=size)

    def snapshot(round_number):
        seen = counts > 0
        denom = np.maximum(counts, 1.)
        risk = absolute / denom
        # Equal-weight quantiles over observed bins allocate levels in the
        # empirical risk scale. Unseen face bins always use Godunov.
        cutoffs = np.quantile(risk[seen], np.arange(1, n_levels) / n_levels)
        levels = np.full(size, n_levels)
        levels[seen] = np.searchsorted(cutoffs, risk[seen])
        return DirectGate(edges, signed / denom, levels, n_levels, round_number)

    for initial, _ in cases:
        v = np.asarray(initial, dtype=np.float64)
        for _ in range(steps):
            observe(v, None)
            v, _ = godunov_step(v, 1., dt=lam)
    first = snapshot(1)
    for initial, budget in cases:
        gate_rollout(model, tables, anchor, first, initial, steps=steps, lam=lam,
                     total_budget=budget, _observe=lambda state: observe(state, first))
    return first, snapshot(2)


def _mask_cost(anchor, state, mask, constants):
    """Certified λ/N sum of adjacent signed flux-error differences for this mask."""
    right = np.roll(state, -1)
    errors = anchor.signed_errors(state[mask], right[mask])
    lo = np.full(len(state), -constants.eps_godunov)
    hi = np.full(len(state), constants.eps_godunov)
    lo[mask], hi[mask] = errors.lo, errors.hi
    difference = up(np.maximum(hi - np.roll(lo, 1), np.roll(hi, 1) - lo))
    # Identical face inputs evaluated by the same deterministic kernel have
    # exactly identical signed errors (also for +/-0 as exact real numbers).
    identical = (state == right) & (state == np.roll(state, 1)) & (mask == np.roll(mask, 1))
    difference[identical] = 0.
    return Fraction(constants.lam) * rigorous_nonneg_sum(difference) / len(state)


def gate_rollout(model, tables, anchor, gate, initial, *, steps, lam, total_budget,
                 audit=False, _observe=None, record_masks=False):
    """Return closed-loop counts and an upper bound on error relative to same-grid Godunov."""
    anchor.check_contract(model, tables)
    if steps <= 0 or not np.isfinite(lam) or lam <= 0 or Fraction(float(lam))*Fraction(tables.envelope) > 1:
        raise ValueError("Positive horizon and admissible CFL required")
    state = np.asarray(initial, dtype=np.float64).copy()
    if state.ndim != 1 or len(state) < 2:
        raise ValueError("Expected one periodic state with at least two cells")
    tables.bin_index(state)
    if not np.array_equal(gate.edges, anchor.edges):
        raise ValueError("Gate and anchor partitions differ")
    constants = FPConstants.build(lam, tables.envelope)
    base, update_reserve = per_step_reserve(tables, constants, len(state))
    fallback_cost = Fraction(base)
    total = Fraction(float(total_budget))
    if total < steps * (fallback_cost + update_reserve):
        raise ValueError("Error budget below all-Godunov reserve")
    spent = Fraction(0)
    accepted = proposed = rejected = audited_steps = 0
    reference = state.copy() if audit else None
    reference_spent = Fraction(0)
    masks = [] if record_masks else None
    for index in range(steps):
        if _observe is not None:
            _observe(state)
        remaining_steps = steps - index
        allowance = (total - spent) / remaining_steps - update_reserve
        mask = gate.propose(state, float(allowance)*len(state)/constants.lam)
        chosen = int(mask.sum())
        proposed += chosen
        cost = fallback_cost
        if chosen:
            cost = _mask_cost(anchor, state, mask, constants)
            if cost > allowance:
                rejected += 1
                mask[:] = False
                cost = fallback_cost
            else:
                accepted += chosen
        if record_masks:
            masks.append(mask.copy())
        new, update, _, _ = mixed_step(model, constants, state, mask)
        if update > update_reserve:
            raise ArithmeticError("Update defect exceeds its certified reserve")
        charge = cost + update
        spent += charge
        if spent > total - (remaining_steps - 1) * (fallback_cost + update_reserve):
            raise ArithmeticError("Error-budget invariant failed")
        if audit:
            if exact_step_error(state, new, lam) > charge:
                raise ArithmeticError("Exact one-step audit failed")
            reference, ref_update, _, _ = mixed_step(
                model, constants, reference, np.zeros(len(state), dtype=bool))
            reference_spent += fallback_cost + ref_update
            distance = sum(abs(Fraction(a)-Fraction(b)) for a, b in zip(new, reference)) / len(state)
            if distance > spent + reference_spent:
                raise ArithmeticError("Cumulative reference audit failed")
            audited_steps += 1
        state = new
    result = {"state": state, "spent": up_float(spent), "accepted": accepted, "proposed": proposed,
              "rejected_steps": rejected, "audited_steps": audited_steps}
    if record_masks:
        result["masks"] = masks
    return result
