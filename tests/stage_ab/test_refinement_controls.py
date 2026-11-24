from itertools import combinations, product

import numpy as np

from stage_ab.refinement_controls import oracle_refinement_frontier


def test_oracle_matches_exhaustive_periodic_faces_and_checks():
    n = 5
    rng = np.random.default_rng(47)
    table = rng.uniform(0.1, 3.0, (n, 2, 2))
    local = rng.uniform(0.2, 2.0, (n, 2, 2))
    for count in (0, 1, 2, n):
        expected = np.full(n + 1, np.inf)
        for checked in combinations(range(n), count):
            costs = table.copy()
            for i in checked:
                costs[i] = np.minimum(table[i], local[i])
            for faces in product((0, 1), repeat=n):
                total = np.float64(0.0)
                for i in range(n):
                    total += costs[i, faces[i-1], faces[i]]
                k = sum(faces)
                expected[k] = min(expected[k], total)
        np.testing.assert_array_equal(oracle_refinement_frontier(table, local, count), expected)
