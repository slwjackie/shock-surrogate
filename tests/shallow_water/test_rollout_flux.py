"""FV flux sampling tests at stationary, transonic and supercritical interfaces."""
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from shallow_water.rollout_study import compile_rollout


class RolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temp = TemporaryDirectory()
        cls.addClassCleanup(temp.cleanup)
        cls.advance = compile_rollout(Path(temp.name))

    def test_constant_state_and_failure_rollback(self):
        for mode in range(9):
            q = np.tile([1.2, 1.7], (4, 1))
            copy = q.copy()
            self.assertEqual(self.advance(4,q,4.,.01,4,.01,mode), 0)
            np.testing.assert_array_equal(q, copy)
            self.assertLess(self.advance(4,q,4.,10.,1,.01,mode), 0)
            np.testing.assert_array_equal(q, copy)

    def test_sampled_riemann_flux(self):
        gravity = 9.81
        fan_velocity = 2*math.sqrt(gravity)/3
        for hl,ul,hr,ur,hf,uf in [
            (4.,0.,1.,0.,2.206987707674228,3.2223376340202954),
            (1.,0.,1.,5.,4./9.,fan_velocity),
            (1.,-5.,1.,0.,4./9.,-fan_velocity),
            (4.,25.,1.,25.,4.,25.),
            (4.,-25.,1.,-25.,1.,-25.)
        ]:
            with self.subTest(interface=(hl,ul,hr,ur)):
                dt = .001
                q = np.array([[hl,hl*ul],[hr,hr*ur]])
                first = q[0].copy()
                self.assertEqual(self.advance(2,q,2.,dt,1,1e-10,5),0)
                def flux(h,u):
                    return np.array([h*u,h*u*u+.5*gravity*h*h])
                expected = first+dt*(flux(hl,ul)-flux(hf,uf))
                np.testing.assert_allclose(q[0],expected,atol=2e-12,rtol=0)


if __name__ == "__main__":
    unittest.main()
