import unittest

import numpy as np

from full_model.analysis.reweight_hazard_exposure import creation_rate


class HazardReweightTest(unittest.TestCase):
    def test_offline_rate_includes_production_patch_measure(self):
        cell_area = 4.0e-14
        radius = 1.0e-6
        state = {
            "dF_density": np.array([[1.0]]),
            "T": np.array([[1000.0]]),
            "site_factor": np.array([[1.0]]),
            "gate_AT": np.array([[1.0]]),
            "classical_barrier_J": np.array([[1.0]]),
            "classical_critical_R_m": np.array([[radius]]),
            "cell_area_m2": np.array(cell_area),
        }
        rate = creation_rate(
            state, attempt_s=2.0, h0_eV=0.0, critical_pa=1.0,
            exp_a=1.0, exp_n=1.0, floor=0.05, entropy_kB=0.0,
            rate_cap_s=10.0)
        expected = 2.0*cell_area/(np.pi*radius**2)
        self.assertAlmostEqual(float(rate[0, 0]), expected)

    def test_explicit_saved_patch_weight_is_authoritative(self):
        state = {
            "dF_density": np.array([[1.0]]),
            "T": np.array([[1000.0]]),
            "site_factor": np.array([[1.0]]),
            "gate_AT": np.array([[1.0]]),
            "classical_barrier_J": np.array([[1.0]]),
            "classical_critical_R_m": np.array([[1.0]]),
            "cell_area_m2": np.array(1.0),
            "patch_weight": np.array([[0.25]]),
        }
        rate = creation_rate(
            state, attempt_s=2.0, h0_eV=0.0, critical_pa=1.0,
            exp_a=1.0, exp_n=1.0, floor=0.05, entropy_kB=0.0,
            rate_cap_s=10.0)
        self.assertAlmostEqual(float(rate[0, 0]), 0.5)


if __name__ == "__main__":
    unittest.main()
