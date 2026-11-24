"""Learn to prioritize local certificates using offline table/local comparisons.

The returned indices only request tighter *certified* bounds. Predicted cost
values are used for ranking and never sent to the verifier or its budget DP.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np


def _features(state, table):
    right = np.roll(state, -1)
    dl = state - np.roll(state, 1)
    dr = right - state
    variation = np.abs(dl) + np.abs(dr)
    # Single-face cost depends on the two ordered states (same face for 01/10).
    face = np.column_stack(((state + right) * 0.5, dr, table[:, 0, 1]))
    # Distinguish monotone slopes from a turning point in the three-cell input.
    shape = np.divide(dr - dl, variation, out=np.zeros_like(dr), where=variation > 0)
    both = np.column_stack((state, variation, shape, table[:, 1, 1]))
    return face, both


class _CostRatio:
    """Hierarchical histogram of log(local/table); shrink sparse bins to coarser estimates."""
    def __init__(self):
        self.levels = []
        self.total = 0.0
        self.weight = 0.0

    @staticmethod
    def _index(x, edges):
        coords = tuple(np.searchsorted(bounds, column) for bounds, column in zip(edges, x.T))
        return np.ravel_multi_index(coords, tuple(len(bounds)+1 for bounds in edges))

    def update(self, x, coarse, tight, w):
        # Identical floor in both terms maps two roundoff-sized bounds to a ratio near one.
        floor = 1e-14
        target = np.log(np.minimum(tight + floor, coarse + floor) / (coarse + floor))
        if not self.levels:
            for bins in (4, 8, 16):
                quantiles = np.linspace(0, 1, bins + 1)[1:-1]
                edges = [np.unique(np.quantile(col, quantiles)) for col in x.T]
                size = int(np.prod([len(bounds)+1 for bounds in edges]))
                self.levels.append((edges, np.zeros(size), np.zeros(size)))
        self.total += float(w @ target)
        self.weight += float(w.sum())
        for edges, totals, weights in self.levels:
            index = self._index(x, edges)
            totals += np.bincount(index, weights=w*target, minlength=len(totals))
            weights += np.bincount(index, weights=w, minlength=len(weights))

    def predict(self, x, coarse):
        if self.weight <= 0:
            raise ValueError("Selector must be fitted before predicting")
        estimate = np.full(len(x), self.total / self.weight)
        for edges, totals, weights in self.levels:
            index = self._index(x, edges)
            estimate = (totals[index] + 4.0*estimate) / (weights[index] + 4.0)
        return np.minimum(coarse, np.maximum(0, (coarse + 1e-14)*np.exp(estimate) - 1e-14))

    def dump(self):
        return {"total": self.total, "weight": self.weight,
                "levels": [([e.tolist() for e in edges], totals.tolist(), counts.tolist())
                           for edges, totals, counts in self.levels]}

    @classmethod
    def restore(cls, data):
        model = cls()
        model.total, model.weight = data["total"], data["weight"]
        model.levels = [([np.asarray(e) for e in edges], np.asarray(totals), np.asarray(counts))
                        for edges, totals, counts in data["levels"]]
        return model


@lru_cache(maxsize=32)
def _lengths(n, count):
    # Scan a geometric set of run lengths, including those near the inspection cap.
    lengths = {int(round(v)) for v in np.geomspace(1, n-1, 15)}
    lengths.update((count-1, count, count+1, n-1))
    return sorted((v for v in lengths if 0 < v < n), reverse=True)


def _choose_run(table, tight01, tight10, tight11, count, allowance):
    """Choose refinements on one proposed contiguous run. The final DP is unrestricted."""
    n = len(table)
    if count < 0 or count > n:
        raise ValueError("Requested local inspection count must be between 0 and N")
    if count == 0 or count == n:
        return np.arange(count, dtype=int)
    begin, end, middle = table[:, 0, 1], table[:, 1, 0], table[:, 1, 1]
    slack = np.maximum(middle - tight11, 0)
    best_all = np.argsort(-slack, kind="stable")[:count]
    if allowance is None:
        allowance = 0.30 * tight11.sum()
    if middle.sum() - slack[best_all].sum() <= allowance:
        return best_all
    starts = np.arange(n)
    zero = table[:, 0, 0]
    total_zero = float(zero.sum())
    prefix = np.r_[0, np.cumsum(np.tile(middle - zero, 2))]
    ring_slack = np.tile(slack, 2)

    for length in _lengths(n, count):
        stops = (starts + length) % n
        cost = (total_zero + begin - zero + end[stops] - zero[stops]
                + prefix[starts + length] - prefix[starts + 1])
        # Exact top-k reduction for this candidate run, given our predicted bounds.
        gains = np.empty((n, length+1))
        gains[:, 0] = np.maximum(begin - tight01, 0)
        gains[:, -1] = np.maximum(end[stops] - tight10[stops], 0)
        if length > 1:
            gains[:, 1:-1] = np.lib.stride_tricks.sliding_window_view(ring_slack[1:], length-1)[:n]
        take = min(count, length+1)
        if take == length+1:
            reduction = gains.sum(axis=1)
        else:
            reduction = np.partition(gains, length+1-take, axis=1)[:, -take:].sum(axis=1)
        estimated = cost - reduction
        start = int(np.argmin(estimated))
        if estimated[start] <= allowance or length == 1:
            positions = np.argsort(-gains[start], kind="stable")[:take]
            chosen = (start + positions) % n
            if len(chosen) < count:
                extra = slack.copy()
                extra[chosen] = -np.inf
                more = np.argsort(-extra, kind="stable")[:count-len(chosen)]
                chosen = np.r_[chosen, more]
            return chosen
    raise ArithmeticError("Run-length search did not reach unit length")


def select_table_block(v, table_costs, count, *, allowance=None):
    """Untrained geometric baseline: optimistically set inspected-row costs to zero."""
    zero = np.zeros(len(v))
    if allowance is None:
        allowance = 0.30 * table_costs[:, 1, 1].sum()
    return _choose_run(table_costs, zero, zero, zero, count, allowance)


class RefinementSelector:
    def __init__(self, training_count=10):
        self.face = _CostRatio()
        self.both = _CostRatio()
        self.training_count = int(training_count)
        self.rounds = 0

    def select(self, v, table_costs, count, *, allowance=None):
        x_face, x_both = _features(v, table_costs)
        right = self.face.predict(x_face, table_costs[:, 0, 1])
        middle = self.both.predict(x_both, table_costs[:, 1, 1])
        return _choose_run(table_costs, right, np.roll(right, 1), middle, count, allowance)

    def fit_round(self, states, *, refit=False):
        """In round two, upweight missed refinements identified by actual local checks.

        Records are (state, table_costs, local_costs[, allowance in unscaled cost
        units]). If absent, the training allowance is 30% of all-neural local cost.
        The teacher uses the same candidate runs as the selector and actual costs.
        """
        if refit != bool(self.rounds):
            raise ValueError("Use refit=False for the first round, refit=True for later rounds")
        features = [[], []]
        targets = [[], []]
        baselines = [[], []]
        weights = [[], []]
        for record in states:
            v, table, local = record[:3]
            tight = np.minimum(table, local)
            xs = _features(v, table)
            w_cell = np.ones(len(v))
            if refit:
                budget = float(record[3]) if len(record)>3 else 0.30 * tight[:, 1, 1].sum()
                count = min(self.training_count,len(v))
                teacher = _choose_run(table, tight[:,0,1], tight[:,1,0], tight[:,1,1], count, budget)
                predicted = self.select(v, table, count, allowance=budget)
                missed = np.setdiff1d(teacher, predicted)
                w_cell[missed] = 5.0
            # A selected cell also uses its left face, i.e. its predecessor's 01 record.
            w_face = np.maximum(w_cell, np.roll(w_cell, -1))
            for j, (left, w) in enumerate(((0, w_face), (1, w_cell))):
                features[j].append(xs[j])
                baselines[j].append(table[:,left,1])
                targets[j].append(tight[:,left,1])
                weights[j].append(w)
        if not weights[0]:
            raise ValueError("Training requires at least one snapshot")
        for j, model in enumerate((self.face, self.both)):
            model.update(np.concatenate(features[j]), np.concatenate(baselines[j]),
                         np.concatenate(targets[j]), np.concatenate(weights[j]))
        self.rounds += 1
        return self

    def save(self, path):
        data = {"rounds": self.rounds, "training_count": self.training_count,
                "face": self.face.dump(), "both": self.both.dump()}
        Path(path).write_text(json.dumps(data, separators=(",", ":")))

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        model = cls(data["training_count"])
        model.rounds = data["rounds"]
        model.face = _CostRatio.restore(data["face"])
        model.both = _CostRatio.restore(data["both"])
        return model
