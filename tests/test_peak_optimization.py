from __future__ import annotations

import unittest

from asb_drx.analytical import ExpFloorLaw
from asb_drx.optimization import fit_peak_scale_parameters, synthetic_peak_observations


EV_J = 1.602176634e-19


class PeakOptimizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.truth = ExpFloorLaw(
            barrier_ref_J=1.7 * EV_J,
            stress_ref_Pa=9.0e8,
            reference_temperature_K=1000.0,
            floor_fraction=0.18,
            shape_a=1.8,
            shape_n=2.3,
            rate_prefactor_s_inv=4.0e11,
            density_exponent_p=4.2,
            burgers_m=2.48e-10,
            barrier_temperature_coefficient=0.35,
            stress_temperature_coefficient=0.12,
        )
        self.template = ExpFloorLaw(
            barrier_ref_J=1.2 * EV_J,
            stress_ref_Pa=1.4e9,
            reference_temperature_K=1000.0,
            floor_fraction=0.18,
            shape_a=1.8,
            shape_n=2.3,
            rate_prefactor_s_inv=1.0e12,
            density_exponent_p=4.2,
            burgers_m=2.48e-10,
            barrier_temperature_coefficient=0.05,
            stress_temperature_coefficient=0.4,
        )

    def test_strength_only_exposes_scale_compensation(self) -> None:
        observations = synthetic_peak_observations(
            self.truth, (800.0, 1000.0, 1200.0), (1.0e1, 1.0e3, 1.0e5), include_density=False
        )
        result = fit_peak_scale_parameters(self.template, observations)
        self.assertLess(result.rms_log_residual, 1.0e-8)
        self.assertFalse(result.identifiable)
        self.assertLess(result.jacobian_rank, result.parameter_count)

    def test_independent_peak_density_restores_scale_identifiability(self) -> None:
        observations = synthetic_peak_observations(
            self.truth, (800.0, 1000.0, 1200.0), (1.0e1, 1.0e3, 1.0e5), include_density=True
        )
        result = fit_peak_scale_parameters(self.template, observations)
        self.assertLess(result.rms_log_residual, 1.0e-8)
        self.assertTrue(result.identifiable)
        for name in (
            "barrier_ref_J",
            "stress_ref_Pa",
            "rate_prefactor_s_inv",
            "barrier_temperature_coefficient",
            "stress_temperature_coefficient",
        ):
            fitted = getattr(result.law, name)
            expected = getattr(self.truth, name)
            self.assertAlmostEqual(fitted / expected, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
