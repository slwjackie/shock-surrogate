"""Independent end-to-end tests for the certified wet-interface solvers."""
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from shallow_water.benchmark import METHODS, build_solver
from shallow_water.data import decimal_audit, reference, sample


MODES = range(len(METHODS))


class CertifiedDepthSolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.library = build_solver(Path(cls.temp.name))

    def solve(self, states, mode, *, g=9.81, tol=1e-7):
        inputs = np.ascontiguousarray(states, dtype=np.float64)
        outputs = np.full(len(inputs), np.nan)
        stats = np.full((len(inputs), 4), -10000, dtype=np.int32)
        failures = self.library.sw_solve_batch(
            len(inputs), inputs, g, tol, mode, outputs, stats,
        )
        return failures, outputs, stats

    def test_dam_break_and_two_rarefaction(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                fail, h, _ = self.solve([[4., 0., 1., 0.]], mode, tol=1e-8)
                self.assertEqual(fail, 0)
                self.assertAlmostEqual(h[0], 2.206987707674, delta=1.01e-8)

                # g=4: c*=2-(1-(-1))/4=1.5, so h*=c*²/g=0.5625.
                fail, h, stats = self.solve([[1., -1., 1., 1.], [2., 1., 2., 1.]],
                                            mode, g=4., tol=1e-8)
                self.assertEqual(fail, 0)
                np.testing.assert_allclose(h, [0.5625, 2.], rtol=0, atol=1e-8)
                np.testing.assert_array_equal(stats[:, 0], [0, 0])

    def test_bad_classical_guess_is_corrected(self):
        # The two shocks colliding head-on have a much deeper middle state.
        states = [[1., 3., 1., -3.]]
        fail, h, stats = self.solve(states, 0)
        self.assertEqual(fail, 0)
        self.assertGreater(h[0], 1.5)
        self.assertEqual(stats[0, 0], 2)
        self.assertGreater(stats[0, 1] + stats[0, 2], 0)
        self.assertTrue(decimal_audit(states[0], h[0], 1e-7))

    def test_mirror_and_velocity_boost_against_bisection(self):
        cases = np.asarray([[4., 0., 1., 0.], [0.4, 1.5, 3.0, -0.5],
                            [1.5, -0.25, 0.75, 1.25]])
        mirror = cases[:, [2, 3, 0, 1]].copy()
        mirror[:, [1, 3]] *= -1
        boost = cases.copy()
        boost[:, [1, 3]] += 2.5
        expected = np.tile(reference(cases), 3)
        for mode in MODES:
            fail, h, _ = self.solve(np.vstack([cases, mirror, boost]), mode)
            self.assertEqual(fail, 0)
            np.testing.assert_allclose(h, expected, rtol=0, atol=1e-7)

    def test_shifted_states_pass_decimal_audit(self):
        cases = sample(10, 61023, "shift")
        for mode in MODES:
            fail, h, _ = self.solve(cases, mode, tol=1e-3)
            self.assertEqual(fail, 0)
            for state, candidate in zip(cases, h):
                self.assertTrue(decimal_audit(state, candidate, 1e-3))

    def test_pure_or_generated_dry_and_invalid_data_fail_closed(self):
        # For these depths and g=4 the limiting velocity difference is
        # 2*(sqrt(g*hL)+sqrt(g*hR))=8. The root at zero is excluded.
        states = [[0., 0., 1., 0.], [1., -4., 1., 4.], [1., -5., 1., 5.],
                  [math.nan, 0., 1., 0.], [1., 0., math.inf, 0.]]
        for mode in MODES:
            fail, depths, stats = self.solve(states, mode, g=4.)
            self.assertEqual(fail, len(states))
            self.assertTrue(np.isnan(depths).all())
            self.assertTrue(np.all(stats[:, 0] < 0))
        for g, tol in [(0., 1e-4), (9.81, math.nan), (9.81, 0.)]:
            fail, _, stats = self.solve([[4., 0., 1., 0.]], 0, g=g, tol=tol)
            self.assertEqual(fail, 1)
            self.assertLess(stats[0, 0], 0)


if __name__ == "__main__":
    unittest.main()
