"""Regression tests for the offline feedback and online choice of local checks."""
import numpy as np
import pytest

from stage_ab.refinement_learning import RefinementSelector, select_table_block


def test_feedback_reweights_missed_cells_and_their_two_faces():
    n = 8
    state = np.zeros(n)
    table = np.ones((n, 2, 2))
    table[:, 0, 0] = 0.0
    allowance = 100.0
    selector = RefinementSelector(training_count=2)
    selector.fit_round([(state, table, table, allowance)])

    # With no measured tightening, stable ranking chooses cells zero and one.
    before = selector.select(state, table, 2, allowance=allowance)
    np.testing.assert_array_equal(before, [0, 1])
    assert selector.face.weight == selector.both.weight == n

    # Actual local checks identify two missed cells on the all-neural route.
    tight = table.copy()
    tight[[5, 6], 1, 1] = 0.01
    selector.fit_round([(state, table, tight, allowance)], refit=True)
    assert selector.rounds == 2
    assert selector.both.weight == 2 * n + 4 * 2  # Two cells receive weight five.
    assert selector.face.weight == 2 * n + 4 * 3  # Their distinct faces are 4, 5, 6.


def test_selector_roundtrip_preserves_deterministic_inspection_count(tmp_path):
    rng = np.random.default_rng(73)
    n = 24
    records = []
    for _ in range(3):
        state = rng.normal(size=n)
        table = rng.uniform(0.05, 0.4, (n, 2, 2))
        table[:, 0, 0] = 1e-14
        table[:, 1, 0] = np.roll(table[:, 0, 1], 1)
        local = table * rng.uniform(0.03, 0.8, table.shape)
        local[:, 1, 0] = np.roll(local[:, 0, 1], 1)
        records.append((state, table, local, 0.2))

    selector = RefinementSelector(training_count=3).fit_round(records[:1])
    selector.fit_round(records[1:2], refit=True)
    path = tmp_path / "selector.json"
    selector.save(path)
    restored = RefinementSelector.load(path)
    state, table, _, allowance = records[-1]
    for count in (0, 1, 3, n-1, n):
        selected = selector.select(state, table, count, allowance=allowance)
        np.testing.assert_array_equal(restored.select(state, table, count, allowance=allowance),
                                      selected)
        baseline = select_table_block(state, table, count, allowance=allowance)
        for indices in (baseline, selected):
            assert len(indices) == len(set(indices)) == count
            assert np.all((0 <= indices) & (indices < n))
    for count in (-1, n+1):
        with pytest.raises(ValueError):
            restored.select(state, table, count, allowance=allowance)
