import unittest
from fractions import Fraction

import numpy as np

from certified_burgers.godunov import godunov_flux
from stage_ab.anchor_refinement import AnchorTable
from stage_ab.cheap_gate import DirectGate, _mask_cost, gate_rollout, train_gate
from stage_ab.frozen_flux import FrozenFlux, exact_godunov
from stage_ab.stage_a2 import CertificateTables, CertifiedFlux, FPConstants, pairs


class TestCheapGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = CertifiedFlux(FrozenFlux([(np.array([[.3, -.2]]), np.array([.05]))]))
        cls.base = CertificateTables.build(cls.model, bins=2, triple=False)
        cls.anchor = AnchorTable.build(cls.model, cls.base, bins=8)
        cls.state = np.array([.6, .6, .6, -.45, .3, -.45, 0., -0.])
        cls.bins = cls.anchor.bins ** 2

    def test_signed_mask_cost_bounds_arbitrary_patterns(self):
        constants = FPConstants.build(.2, self.base.envelope)
        state = self.state
        faces = pairs(state)
        reference = [exact_godunov(Fraction(a), Fraction(b)) for a, b in faces]
        numerical = [godunov_flux(*faces.T), self.model.predict(faces)]
        masks = [np.ones(len(state), bool), np.zeros(len(state), bool)]
        masks += list(np.random.default_rng(39).random((12, len(state))) > .5)
        for mask in masks:
            error = [Fraction(numerical[int(trust)][i]) - reference[i] for i, trust in enumerate(mask)]
            exact = Fraction(.2) * sum(abs(v - error[i-1]) for i, v in enumerate(error)) / len(state)
            self.assertGreaterEqual(_mask_cost(self.anchor, state, mask, constants), exact)

    def test_bad_advice_is_rejected_and_exact_audit_passes(self):
        gate = DirectGate(self.anchor.edges, np.zeros(self.bins), np.zeros(self.bins, int), 16, 1)
        result = gate_rollout(self.model, self.base, self.anchor, gate, self.state,
                              steps=6, lam=.2, total_budget=1e-9, audit=True)
        self.assertEqual(result["proposed"], 6*len(self.state))
        self.assertEqual(result["accepted"], 0)
        self.assertEqual(result["rejected_steps"], 6)
        self.assertEqual(result["audited_steps"], 6)
        self.assertLessEqual(result["spent"], 1e-9)

    def test_bootstrap_returns_two_independent_tables_and_unseen_faces_fall_back(self):
        cases = [(self.state, .8), (self.state[::-1].copy(), .8)]
        first, second = train_gate(self.model, self.base, self.anchor, cases, steps=4, lam=.2)
        self.assertEqual((first.round, second.round), (1, 2))
        self.assertFalse(np.shares_memory(first.signed_error, second.signed_error))
        self.assertFalse(first.propose(np.full(8, -1.8), 1e4).any())
        result = gate_rollout(self.model, self.base, self.anchor, second, self.state, steps=4,
                              lam=.2, total_budget=.8, audit=True)
        self.assertGreater(result["accepted"], 0)
        self.assertEqual(result["audited_steps"], 4)
        self.assertLessEqual(result["spent"], .8)


if __name__ == "__main__":
    unittest.main()
