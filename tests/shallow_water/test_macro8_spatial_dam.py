import ctypes
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from shallow_water.macro8_data import LAMBDA, N, STEPS, ReferenceSolver
from shallow_water.macro8_model import OUT, predict
from shallow_water.macro8_spatial_dam import build_kernel, case_input, measure, summarize
from shallow_water.rollout_study import physical_flux


class SpatialDamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = TemporaryDirectory()
        cls.addClassCleanup(folder.cleanup)
        cls.run_dam = staticmethod(build_kernel(folder.name))
        with np.load(OUT/"replay.npz") as archive:
            cls.model = tuple(archive[k][0] for k in ("encoder", "bias", "kernel"))

    def test_shared_flux_and_partial_stencils_on_both_dam_orientations(self):
        with ReferenceSolver() as solver:
            for index, expected_left in [(321, True), (325, False)]:
                with self.subTest(index=index):
                    initial = case_input(index)
                    gate = int(np.flatnonzero(np.diff(initial[:, 0]))[0]+1)
                    self.assertEqual(bool(initial[0, 0] > initial[-1, 0]), expected_left)
                    q = solver.advance(initial, steps=5)
                    result = np.empty_like(q)
                    failed = ctypes.c_int()
                    code = self.run_dam(q, result, 2, gate, expected_left, 1, ctypes.byref(failed))
                    self.assertEqual((code, failed.value), (0, 0))
                    god = solver.advance(q)
                    nn = predict(q, self.model)
                    classical = slice(gate, N) if expected_left else slice(0, gate)
                    neural_interior = slice(0, gate-1) if expected_left else slice(gate+1, N)
                    np.testing.assert_allclose(result[classical], god[classical], rtol=0, atol=5e-13)
                    np.testing.assert_allclose(result[neural_interior], nn[neural_interior],
                                               rtol=0, atol=5e-13)
                    left_flux, right_flux = [physical_flux(h, m/h) for h, m in q[[0, -1]]]
                    np.testing.assert_allclose((result-q).sum(axis=0),
                                               STEPS*LAMBDA*(left_flux-right_flux), atol=3e-13)

    def test_baselines_and_new_case_id(self):
        row = measure(self.run_dam, 385, repeats=1)
        self.assertEqual(row["id"], "test-000385")
        self.assertEqual(summarize([row])["godunov"]["completed"], 1)
        initial = case_input(385)
        gate = int(np.flatnonzero(np.diff(initial[:, 0]))[0]+1)
        with ReferenceSolver() as solver:
            god = solver.advance(initial, steps=64)
        nn = initial.copy()
        for _ in range(8):
            nn = predict(nn, self.model)
        for method, expected in enumerate((god, nn)):
            result = np.empty_like(initial)
            failed = ctypes.c_int()
            code = self.run_dam(initial, result, method, gate, True, 8, ctypes.byref(failed))
            self.assertEqual(code, 0)
            np.testing.assert_allclose(result, expected, atol=8e-12, rtol=8e-12)


if __name__ == "__main__":
    unittest.main()
