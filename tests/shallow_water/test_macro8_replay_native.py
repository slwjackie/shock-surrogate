"""Frozen on-policy neural weights use the same native projection and guard ABI."""
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from shallow_water.macro8_data import N, ReferenceSolver, make_split
from shallow_water.macro8_model import OUT, predict
from shallow_water.macro8_replay_native import build_replay_native, replay_step


class ReplayDeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = TemporaryDirectory()
        cls.addClassCleanup(folder.cleanup)
        cls.lib = build_replay_native(folder.name)

    def test_native_three_arms_match_original_saved_numpy_arrays(self):
        with np.load(OUT/"weights.npz") as offline, np.load(OUT/"replay.npz") as replay:
            models = [tuple(archive[k][i] for k in ("encoder","bias","kernel"))
                      for archive,i in ((offline,2),(replay,0),(replay,1))]
        with ReferenceSolver() as solver:
            cases = make_split("dev",8,solver)
        for arm,model in enumerate(models):
            for q in cases.q:
                field, route = replay_step(self.lib,q,arm)
                self.assertEqual(route,-1)
                np.testing.assert_allclose(field,predict(q,model),rtol=1e-12,atol=1e-12)

    def test_fallback_and_acceptance_are_distinct_and_preserve_input(self):
        with ReferenceSolver() as solver:
            q = make_split("dev",2,solver).q[1]
            exact = solver.advance(q)
        before = q.copy()
        forced,route = replay_step(self.lib,q,2,-1.)
        self.assertEqual(route,0)
        np.testing.assert_array_equal(forced,exact)
        constant = np.tile([1.6,-.55],(N,1))
        accepted,route = replay_step(self.lib,constant,1,0.)
        self.assertEqual(route,1)
        np.testing.assert_array_equal(accepted,constant)
        np.testing.assert_array_equal(q,before)

    def test_input_validation(self):
        q = np.tile([1.,0.],(N,1))
        invalid = q.copy()
        invalid[3,0] = 0.
        for state,arm,cut in ((invalid,1,None),(q[:-1],1,None),(q,5,None),
                              (q,True,None),(q,1,np.nan)):
            with self.assertRaises(ValueError):
                replay_step(self.lib,state,arm,cut)
