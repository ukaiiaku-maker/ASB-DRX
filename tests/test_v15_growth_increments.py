import unittest

import numpy as np

from full_model.analysis.audit_v15_growth_increments import linear_velocity


class V15IncrementMetricsTest(unittest.TestCase):
    def test_velocity_fit_recovers_slope_and_uncertainty(self):
        time = np.linspace(0.0, 4.0, 9)
        amplitude = 3.0e-7 + 2.5e-9*time
        fit = linear_velocity(time, amplitude)
        self.assertAlmostEqual(fit["velocity_m_s"], 2.5e-9, places=20)
        self.assertLess(fit["velocity_standard_error_m_s"], 1e-20)

    def test_velocity_fit_rejects_underspecified_window(self):
        with self.assertRaises(ValueError):
            linear_velocity([0.0, 1.0], [0.0, 1.0])
