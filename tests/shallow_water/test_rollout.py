"""Test finite-volume interface sampling, conservation, and failure handling."""
from math import sqrt
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from shallow_water.rollout_study import compile_rollout, METHOD_NAMES

G = 9.81
TIGHT = 1e-8
CLASSICAL = 6


def state(h, u):
    return np.array([h, h*u])


class ShallowWaterRolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.advance = compile_rollout(Path(cls.directory.name))

    def test_tight_godunov_reflection(self):
        q = np.array([state(h, u) for h, u in
                      [(4., 0.), (4., 0.), (1.2, 0.3), (1., -0.7), (0.8, 0.)]])
        mirrored = np.ascontiguousarray(q[::-1] * [1., -1.])
        for current in (q, mirrored):
            self.assertEqual(self.advance(5, current, 5., .2, 4, TIGHT, CLASSICAL), 0)
        np.testing.assert_allclose(mirrored, q[::-1] * [1., -1.], rtol=0., atol=1e-12)

    def test_mass_and_positive_depth_with_undisturbed_outflows(self):
        # Interfaces at both boundaries stay stationary for these four steps.
        q0 = np.zeros((32, 2))
        q0[:, 0] = 1.
        q0[12:20, 0] = 3.
        for mode in range(len(METHOD_NAMES)):
            q = q0.copy()
            self.assertEqual(self.advance(32, q, 32., .16, 4, TIGHT, mode), 0)
            self.assertGreater(q[:, 0].min(), 0.)
            self.assertAlmostEqual(q[:, 0].sum(), q0[:, 0].sum(), places=12)

    def test_cfl_failure_leaves_input_intact_after_successful_first_step(self):
        q0 = np.stack([state(4., 0.), state(1., 0.)])
        dt = .94 / sqrt(4.*G)
        q = q0.copy()
        self.assertEqual(self.advance(2, q, 2., dt, 1, TIGHT, CLASSICAL), 0)
        self.assertFalse(np.array_equal(q, q0))
        self.assertGreater(float(np.max(abs(q[:, 1]/q[:, 0]) + np.sqrt(G*q[:, 0]))) * dt, 1.)
        q = q0.copy()
        self.assertEqual(self.advance(2, q, 2., 2.*dt, 2, TIGHT, CLASSICAL), -3)
        np.testing.assert_array_equal(q, q0)


if __name__ == "__main__":
    unittest.main()
