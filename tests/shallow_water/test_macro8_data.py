import unittest

import numpy as np

from shallow_water.macro8_data import (DURATION, FAMILIES, ReferenceSolver, SPLIT_SEEDS,
                                       analytical_dam, make_split)
from shallow_water.rollout_study import physical_flux


class Macro8DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = ReferenceSolver()
        cls.solver = cls.context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)

    def test_independent_splits_stable_prefixes_and_wet_labels(self):
        collected = set()
        for split in SPLIT_SEEDS:
            batch = make_split(split, 8, self.solver)
            self.assertEqual(batch.families, 2 * FAMILIES)
            self.assertTrue(np.isfinite(batch.target).all())
            self.assertGreater(batch.target[:, :, 0].min(), 0)
            self.assertFalse(collected.intersection(batch.ids))
            collected.update(batch.ids)
            shorter = make_split(split, 3, self.solver)
            np.testing.assert_array_equal(shorter.q, batch.q[:3])
            np.testing.assert_array_equal(shorter.target, batch.target[:3])
            # Constant edge buffers preserve the prescribed outflow flux.
            np.testing.assert_array_equal(batch.target[:, [0, -1]], batch.q[:, [0, -1]])
        self.assertEqual(len(SPLIT_SEEDS), len(set(SPLIT_SEEDS.values())))

    def test_refinement_time_and_dam_balance(self):
        data = make_split("test", 8, self.solver)
        for i in (1, 5):
            q = data.q[i]
            np.testing.assert_array_equal(self.solver.refined(q, 1), data.target[i])
            for blocks in (1, 4):
                end = self.solver.refined(q, 4, blocks)
                exact = analytical_dam(q, blocks)
                left, right = [physical_flux(h, m/h) for h, m in q[[0, -1]]]
                mean = q.mean(axis=0) + blocks*DURATION*(left-right)
                np.testing.assert_allclose(end.mean(axis=0), mean, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(exact.mean(axis=0), mean, rtol=1e-12, atol=1e-12)
                coarse = self.solver.advance(q, steps=8*blocks)
                self.assertTrue((np.abs(end-exact).mean(axis=0) <
                                 np.abs(coarse-exact).mean(axis=0)).all())

    def test_failed_reference_is_visible(self):
        q = np.ones((256, 2))
        q[8, 0] = 0
        with self.assertRaisesRegex(RuntimeError, r"rollout returned -2"):
            self.solver.advance(q)
        with self.assertRaises(ValueError):
            analytical_dam(np.ones_like(q))


if __name__ == "__main__":
    unittest.main()
