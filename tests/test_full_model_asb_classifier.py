import unittest

import numpy as np

from full_model.production.asb_classifier import (
    ASBCriteria, classify, matched_history, refinement_passes,
)


class StrictASBClassifierTest(unittest.TestCase):
    def criteria(self):
        return ASBCriteria(0.25, 50.0, 0.2, 2.0, 1e-6, 0.05)

    def history(self, temperature_excess=100.0):
        nt, nx, ny = 4, 32, 32
        rate = np.ones((nt, nx, ny))
        rate[:, 15:17, :] = 100.0
        control = np.full((nt, nx, ny), 1000.0)
        temperature = control.copy()
        temperature[:, 15:17, :] += temperature_excess
        stress = np.array([100.0, 90.0, 70.0, 60.0])
        time = np.array([1e-6, 2e-6, 3e-6, 4e-6])
        return matched_history(rate, temperature, control, stress, time, 1e-7, 1e-7)

    def test_all_simultaneous_fields_and_refinement_classify(self):
        decision = classify(self.history(), 1e-7, self.criteria(), True)
        self.assertTrue(decision.classified)
        self.assertAlmostEqual(decision.persistence_s, 1e-6)

    def test_temperature_variance_without_control_excess_is_not_asb(self):
        decision = classify(self.history(0.0), 1e-7, self.criteria(), True)
        self.assertFalse(decision.classified)
        self.assertIn("matched_temperature_excess", decision.failed_criteria)

    def test_refinement_is_conjunctive(self):
        decision = classify(self.history(), 1e-7, self.criteria(), False)
        self.assertFalse(decision.classified)
        self.assertIn("grid_and_timestep_refinement", decision.failed_criteria)
        self.assertTrue(refinement_passes(2.0, 2.05, 4.0, 4.1, 0.05))


if __name__ == "__main__":
    unittest.main()
