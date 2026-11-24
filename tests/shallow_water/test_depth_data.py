import math
import unittest

import numpy as np

from shallow_water.data import decimal_audit, reference, sample


class DepthDataTests(unittest.TestCase):
    def test_symmetric_cases_and_dam_break(self):
        states = np.array([[4.,0.,1.,0.],[1.,0.,4.,0.],[2.,1.,2.,1.],[1.,-1.,1.,1.]])
        truth = reference(states)
        expected = [2.206987707674228, 2.206987707674228, 2.,
                    (math.sqrt(9.81)-.5)**2/9.81]
        np.testing.assert_allclose(truth, expected, rtol=2e-14)
        for state, root in zip(states, truth):
            self.assertTrue(decimal_audit(state, root, 1e-12))
            self.assertFalse(decimal_audit(state, root+.011, .01))

    def test_valid_splits_and_invariances(self):
        for split in ("train", "iid", "smooth", "shift"):
            states = sample(64, 870, split)
            truth = reference(states)
            self.assertTrue((truth > 0).all())
            transformed = states[:, [2,3,0,1]].copy()
            transformed[:, [1,3]] = 2.-transformed[:, [1,3]]
            np.testing.assert_allclose(reference(transformed), truth, rtol=2e-13)
            self.assertTrue(all(decimal_audit(s,h,1e-11) for s,h in zip(states,truth)))

    def test_excluded_states(self):
        for state in ([0.,0.,1.,0.],[1.,-10.,1.,10.],[1.,math.inf,2.,1.]):
            with self.assertRaises(ValueError):
                reference([state])
            self.assertFalse(decimal_audit(state, 1., .01))


if __name__ == "__main__":
    unittest.main()
