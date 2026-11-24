import ctypes
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from shallow_water.macro8_data import ReferenceSolver
from shallow_water.macro8_guard import build_guard, fit_cutoff


class Macro8GuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        # A known proposal lets the guard and routing regressions run independently
        # of training. Experiment outputs always use the production learned model.
        stub = Path(cls.temp.name)/"identity.c"
        stub.write_text("""#include <string.h>
int sw_macro8_predict(int n,const double *q,double *out,int model) {
    (void)model; memcpy(out,q,2*(size_t)n*sizeof(double)); return 0;
}
""")
        cls.lib = build_guard(cls.temp.name, stub)

    def measure(self, q, pred, cut=np.inf):
        score = ctypes.c_double()
        ok = self.lib.sw_macro8_check(len(q), q, pred, cut, ctypes.byref(score))
        return ok, score.value

    def test_balance_positivity_and_open_boundaries(self):
        q = np.tile([1.,.2],(64,1))
        self.assertEqual(self.measure(q,q,0.0),(1,0.0))
        a = q.copy()
        a[24,0] += .1
        self.assertEqual(self.measure(q,a),(0,np.inf))
        a = q.copy()
        a[20,0], a[21,0] = -.1, 2.1
        self.assertEqual(self.measure(q,a),(0,np.inf))
        a = q.copy()
        a[2,0] += .01
        self.assertEqual(self.measure(a,a),(0,np.inf))
        q[32:]=[1.5,-.3]
        with ReferenceSolver() as reference:
            truth = reference.advance(q)
        self.assertEqual(self.measure(q,truth)[0],1)

    def test_conserved_displaced_pulse_has_a_finite_nonzero_score(self):
        q = np.tile([1.,0.],(64,1))
        q[24:33,0] = 1.6
        pred = np.roll(q,8,axis=0)
        with ReferenceSolver() as reference:
            target = reference.advance(q)
        self.assertGreater(np.abs(target-pred).mean(axis=0)[0],.01)
        np.testing.assert_allclose(pred.sum(axis=0),q.sum(axis=0))
        ok,score = self.measure(q,pred)
        self.assertEqual(ok,1)
        self.assertGreater(score,0.)
        self.assertEqual(self.measure(q,pred,score/2)[0],0)

    def test_weak_balance_alone_can_accept_an_inaccurate_profile(self):
        # Construct two adjacent patch changes with exactly cancelling weak
        # defects. The true solution from constant initial data is constant.
        # This synthetic probe tests the verifier independently of the model.
        q = np.tile([1.,0.],(48,1))
        pred = q.copy()
        j,d = 24,.5
        pred[j,0] += d
        pred[[j-1,j+1],0] -= .5*d
        impulse = .56*.25*9.81*((1+d)**2-1)
        pred[j-1,1],pred[j+1,1] = -impulse,impulse
        self.assertGreater(np.abs(q-pred).mean(axis=0)[0],.01)
        ok,score = self.measure(q,pred,1e-10)
        self.assertEqual(ok,1)
        self.assertLess(score,1e-10)

    def test_fallback_restarts_from_input_and_failed_result_is_atomic(self):
        q = np.tile([1.,0.],(64,1))
        q[:32,0] = 2
        with ReferenceSolver() as reference:
            target = reference.advance(q)
        original = q.copy()
        out = np.full_like(q,123)
        risk = ctypes.c_double()
        route = self.lib.sw_macro8_routed(len(q),q,out,0,-1,ctypes.byref(risk))
        self.assertEqual(route,0)
        np.testing.assert_array_equal(q,original)
        np.testing.assert_array_equal(out,target)
        aliased = original.copy()
        self.assertEqual(self.lib.sw_macro8_routed(len(q),aliased,aliased,0,-1,
                                                   ctypes.byref(risk)),0)
        np.testing.assert_array_equal(aliased,target)
        q[30,0]=-1
        self.assertLess(self.lib.sw_macro8_routed(len(q),q,out,0,-1,ctypes.byref(risk)),0)
        np.testing.assert_array_equal(out,target)

    def test_threshold_rejects_ties_and_extrapolation(self):
        self.assertEqual(fit_cutoff([0,.2,.2,9,np.inf],[True,True,False,True,False]),0.)
        self.assertEqual(fit_cutoff([.2,.3,np.inf],[False,True,True]),-1.)
        self.assertEqual(fit_cutoff([.2,.3,np.inf],[True,True,False]),.3)
