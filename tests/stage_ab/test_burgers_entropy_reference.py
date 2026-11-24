import unittest

import numpy as np

from certified_burgers.godunov import godunov_step
from stage_ab.burgers_entropy_reference import entropy_averages


class EntropyAveragesTests(unittest.TestCase):
    def test_constant_and_initial_time(self):
        for v in [-1.3, 0., 2.1]:
            state = np.full(13, v)
            np.testing.assert_array_equal(entropy_averages(state, 1.2), state)
        state = np.array([2., -0.5, 0., 0.5])
        np.testing.assert_array_equal(entropy_averages(state, 0), state)

    def test_shock_and_rarefaction_before_interaction(self):
        n, t = 32, 0.11
        state = np.r_[np.ones(n // 2), np.zeros(n // 2)]
        x = np.arange(n + 1) / n
        # 0->1 centered rarefaction at zero, 1->0 shock at 1/2+t/2.
        integrated = np.where(x < t, x**2 / (2 * t), np.minimum(x - t / 2, 0.5))
        expected = n * np.diff(integrated)
        np.testing.assert_allclose(entropy_averages(state, t), expected, atol=2e-14)
        for shift in [7, -11]:
            np.testing.assert_allclose(entropy_averages(np.roll(state, shift), t),
                                       np.roll(expected, shift), atol=2e-14)
        np.testing.assert_allclose(entropy_averages(-state[::-1], t),
                                   -expected[::-1], atol=2e-14)

    def test_conservation_and_bounds_through_periodic_crossings(self):
        state = np.random.default_rng(119).uniform(-0.3, 1.8, 31)
        for t in [0.0125, 0.09375, 0.1, 2.7]:
            evolved = entropy_averages(state, t)
            self.assertAlmostEqual(np.mean(evolved), np.mean(state), places=13)
            self.assertGreaterEqual(evolved.min(), state.min() - 1e-12)
            self.assertLessEqual(evolved.max(), state.max() + 1e-12)
        # Galilean boost of one period over the elapsed time.
        t = 2.7
        boost = -2 / t
        np.testing.assert_allclose(entropy_averages(state + boost, t),
                                   entropy_averages(state, t) + boost, atol=1e-13)

    def test_refined_godunov_convergence(self):
        n = 128
        x = (np.arange(n) + 0.5) / n
        state = 0.2 + 0.55 * np.sin(2 * np.pi * x) + 0.45 * (x < 0.47)
        times = [8, 60, 64]
        exact = {h: entropy_averages(state, h * 0.2 / n) for h in times}
        errors = {}
        for r in [8, 16]:
            fine = np.repeat(state, r)
            for k in range(1, max(times) * r + 1):
                fine, _ = godunov_step(fine, dx=1/(n*r), dt=0.2/(n*r))
                if k in [h * r for h in times]:
                    h = k // r
                    coarse = fine.reshape(n, r).mean(axis=1)
                    errors[r, h] = np.abs(coarse - exact[h]).mean()
        for h in times:
            self.assertLess(errors[16, h], errors[8, h])


if __name__ == "__main__":
    unittest.main()
