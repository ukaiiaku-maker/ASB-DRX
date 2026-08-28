from __future__ import annotations

import math
import unittest

from asb_drx.analytical import ExpFloorLaw


EV_J = 1.602176634e-19


class ExpFloorAnalyticalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.law = ExpFloorLaw(
            barrier_ref_J=1.5 * EV_J,
            stress_ref_Pa=1.2e9,
            reference_temperature_K=1000.0,
            floor_fraction=0.2,
            shape_a=2.0,
            shape_n=2.5,
            rate_prefactor_s_inv=1.0e12,
            density_exponent_p=4.0,
            burgers_m=2.5e-10,
            barrier_temperature_coefficient=0.3,
            stress_temperature_coefficient=0.1,
        )

    def test_barrier_endpoints_and_activation_volume(self) -> None:
        T = 1000.0
        G0 = self.law.barrier_scale_J(T)
        self.assertEqual(self.law.barrier_J(0.0, T), G0)
        high = self.law.barrier_J(100.0 * self.law.stress_scale_Pa(T), T)
        self.assertAlmostEqual(high / G0, self.law.floor_fraction, places=14)
        tau = 0.7 * self.law.stress_scale_Pa(T)
        step = tau * 1.0e-6
        numerical = -(
            self.law.barrier_J(tau + step, T) - self.law.barrier_J(tau - step, T)
        ) / (2.0 * step)
        analytical = self.law.activation_volume_m3(tau, T)
        self.assertAlmostEqual(numerical / analytical, 1.0, places=8)

    def test_peak_closes_rate_and_is_local_maximum(self) -> None:
        T = 1000.0
        rate = 1.0e3
        peak = self.law.peak(T, rate)
        recovered = self.law.shear_rate_s_inv(
            peak.local_activation_stress_Pa, peak.density_m2, T
        )
        self.assertAlmostEqual(recovered / rate, 1.0, places=11)
        delta = 1.0e-3
        for ratio in (1.0 - delta, 1.0 + delta):
            q = peak.taylor_ratio_q * ratio
            density = (q / (self.law.taylor_geometry_factor * self.law.burgers_m)) ** 2
            strength = self.law.macroscopic_strength_Pa(density, T, rate)
            self.assertLess(strength, peak.macroscopic_strength_Pa)

    def test_closed_form_rate_scaling(self) -> None:
        T = 900.0
        rate_1 = 1.0e2
        rate_2 = 1.0e6
        peak_1 = self.law.peak(T, rate_1)
        peak_2 = self.law.peak(T, rate_2)
        expected_strength_ratio = (rate_2 / rate_1) ** (1.0 / self.law.density_exponent_p)
        expected_density_ratio = expected_strength_ratio**2
        self.assertAlmostEqual(
            peak_2.macroscopic_strength_Pa / peak_1.macroscopic_strength_Pa,
            expected_strength_ratio,
            places=11,
        )
        self.assertAlmostEqual(
            peak_2.density_m2 / peak_1.density_m2,
            expected_density_ratio,
            places=11,
        )

    def test_peak_existence_condition_is_enforced(self) -> None:
        no_peak = ExpFloorLaw(
            barrier_ref_J=0.05 * EV_J,
            stress_ref_Pa=1.0e9,
            reference_temperature_K=1500.0,
            floor_fraction=0.8,
            shape_a=2.0,
            shape_n=1.0,
            rate_prefactor_s_inv=1.0e12,
            density_exponent_p=4.0,
            burgers_m=2.5e-10,
        )
        with self.assertRaisesRegex(ValueError, "no interior strength maximum"):
            no_peak.peak(1500.0, 1.0e3)

    def test_invalid_parameters_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "floor_fraction"):
            ExpFloorLaw(
                barrier_ref_J=EV_J,
                stress_ref_Pa=1.0e9,
                reference_temperature_K=1000.0,
                floor_fraction=1.0,
                shape_a=1.0,
                shape_n=1.0,
                rate_prefactor_s_inv=1.0e12,
                density_exponent_p=4.0,
                burgers_m=2.5e-10,
            )


if __name__ == "__main__":
    unittest.main()
