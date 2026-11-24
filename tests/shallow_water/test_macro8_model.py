"""Invariants of the frozen, native whole-field eight-step neural operator."""
import ctypes
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from shallow_water.benchmark import FLAGS
from shallow_water.macro8_data import ReferenceSolver, make_split
from shallow_water.macro8_model import COURANT, HEADER, OUT, SCALES, predict


class Macro8ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        work = TemporaryDirectory()
        cls.addClassCleanup(work.cleanup)
        binary = Path(work.name) / "macro8.so"
        subprocess.run(["gcc", *FLAGS, str(HEADER.with_name("macro8_predict.c")),
                        "-lm", "-o", str(binary)], check=True)
        cls.lib = ctypes.CDLL(str(binary))
        cls.run_predict = cls.lib.sw_macro8_predict
        grid = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS")
        cls.run_predict.argtypes = [ctypes.c_int, grid, grid, ctypes.c_int]
        cls.run_predict.restype = ctypes.c_int
        with np.load(OUT/"weights.npz") as weights:
            cls.models = [tuple(weights[name][model] for name in ("encoder","bias","kernel"))
                          for model in range(3)]

    def test_native_and_python_predictions_match_for_all_modes_and_families(self):
        with ReferenceSolver() as ref:
            cases = make_split("dev", 8, ref)
        for mode, model in enumerate(self.models):
            for q in cases.q:
                out = np.empty_like(q)
                self.assertEqual(self.run_predict(len(q), q, out, mode), 0)
                np.testing.assert_allclose(out, predict(q, model), rtol=1e-12, atol=1e-12)

    def test_constant_moving_wet_state_is_preserved_exactly(self):
        for q in (np.tile([1.7,2.25], (256,1)), np.tile([2.5,-1.3], (2,1))):
            for mode in range(3):
                out = np.empty_like(q)
                self.assertEqual(self.run_predict(len(q), q, out, mode), 0)
                np.testing.assert_array_equal(out, q)

    def test_conservative_balance_matches_two_open_boundary_fluxes(self):
        # Deliberately put the jump close enough to the boundary to touch
        # the predictor's receptive field. Check the actual boundary rule.
        q = np.tile([1.6,.7], (32,1))
        q[3:21] = [.8,-.2]
        q[21:] = [1.2,.3]
        normalized = q[[0,-1]]/SCALES
        h,m = normalized.T
        flux = np.column_stack((m, m*m/h+.5*h*h))
        expected = COURANT*(flux[0]-flux[1])*SCALES
        for mode in range(3):
            out = np.empty_like(q)
            self.assertEqual(self.run_predict(len(q), q, out, mode), 0)
            np.testing.assert_allclose((out-q).sum(axis=0), expected, atol=2e-12)

    def test_invalid_state_and_in_place_call_rejected(self):
        for item in ([0.,0.],[-1.,0.],[np.nan,0.],[1.,np.inf]):
            q = np.tile([1.,0.], (32,1))
            q[9] = item
            out = np.full_like(q,123.)
            self.assertLess(self.run_predict(len(q),q,out,2), 0)
            np.testing.assert_array_equal(out, 123.)
        q = np.tile([1.,0.],(32,1))
        self.assertLess(self.run_predict(len(q),q,q,0), 0)
        self.assertLess(self.run_predict(len(q),q,np.empty_like(q),3), 0)
