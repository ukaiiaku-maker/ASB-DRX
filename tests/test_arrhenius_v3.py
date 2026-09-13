from __future__ import annotations

from dataclasses import replace
import math
import unittest

from asb_drx.analytical import ExpFloorLaw
from asb_drx.arrhenius_v3 import (
    ArrheniusMechanism,
    BoundedActivationEntropy,
    ExpFloorEnthalpy,
    KB_J_PER_K,
    PARAMETER_CLASSIFICATION,
)


EV_J = 1.602176634e-19


class ArrheniusV3Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.enthalpy = ExpFloorEnthalpy(
            reference_enthalpy_J=1.5 * EV_J,
            reference_stress_Pa=1.2e9,
            reference_temperature_K=1000.0,
            floor_fraction=0.2,
            shape_a=2.0,
            shape_n=2.5,
            enthalpy_temperature_coefficient=0.3,
            stress_temperature_coefficient=0.1,
        )
        self.mechanism = ArrheniusMechanism(
            self.enthalpy, BoundedActivationEntropy(), 1.0e12,
            density_exponent=2.0, density_reference_m2=1.0e16,
            validity_temperature_K=(500.0, 1500.0),
            validity_stress_Pa=(0.0, 5.0e9),
        )

    def test_enthalpy_floor_and_activation_volume_sign(self) -> None:
        temperature = 1000.0
        scale = self.enthalpy.enthalpy_scale_J(temperature)
        self.assertEqual(self.enthalpy.enthalpy_J(0.0, temperature), scale)
        high = self.enthalpy.enthalpy_J(100.0e9, temperature)
        self.assertAlmostEqual(high / scale, 0.2, places=14)
        for stress in (0.0, 0.2e9, 1.0e9, 3.0e9):
            self.assertGreaterEqual(
                self.enthalpy.activation_volume_m3(stress, temperature), 0.0
            )

    def test_forward_reverse_symmetry_zero_flow_and_dissipation(self) -> None:
        for stress in (-1.0e9, -0.2e9, 0.0, 0.2e9, 1.0e9):
            rate = self.mechanism.net_rate_s_inv(stress, 900.0, 2.0e16)
            reverse = self.mechanism.net_rate_s_inv(-stress, 900.0, 2.0e16)
            self.assertAlmostEqual(rate, -reverse, places=14)
            self.assertGreaterEqual(stress * rate, 0.0)
        self.assertEqual(self.mechanism.net_rate_s_inv(0.0, 900.0, 2.0e16), 0.0)

    def test_signed_entropy_is_independent_of_enthalpy_shape(self) -> None:
        positive = replace(
            self.mechanism,
            entropy=BoundedActivationEntropy(reference_kB=2.0),
        )
        negative = replace(
            self.mechanism,
            entropy=BoundedActivationEntropy(reference_kB=-2.0),
        )
        stress = 0.7e9
        temperature = 900.0
        self.assertEqual(
            positive.enthalpy.enthalpy_J(stress, temperature),
            negative.enthalpy.enthalpy_J(stress, temperature),
        )
        ratio = positive.net_rate_s_inv(stress, temperature, 2.0e16) / negative.net_rate_s_inv(stress, temperature, 2.0e16)
        self.assertAlmostEqual(ratio, math.exp(4.0), places=11)

    def test_attempt_frequency_entropy_nonidentifiability(self) -> None:
        shifted = replace(
            self.mechanism,
            attempt_frequency_s_inv=self.mechanism.attempt_frequency_s_inv / math.exp(3.0),
            entropy=BoundedActivationEntropy(reference_kB=3.0),
        )
        for temperature in (700.0, 1000.0, 1300.0):
            self.assertAlmostEqual(
                shifted.effective_prefactor_s_inv(temperature),
                self.mechanism.effective_prefactor_s_inv(temperature),
                places=3,
            )
            reference = self.mechanism.net_rate_s_inv(
                0.8e9, temperature, 2.0e16
            )
            self.assertAlmostEqual(
                shifted.net_rate_s_inv(0.8e9, temperature, 2.0e16),
                reference, delta=abs(reference) * 3.0e-15,
            )

    def test_zero_entropy_recovers_verified_exp_floor_net_rate(self) -> None:
        old = ExpFloorLaw(
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
        density = 2.0e16
        geometry = old.taylor_ratio(density)
        new = replace(
            self.mechanism,
            density_exponent=4.0,
            density_reference_m2=density / geometry,
        )
        for stress in (-0.8e9, 0.0, 0.8e9):
            self.assertAlmostEqual(
                new.net_rate_s_inv(stress, 900.0, density),
                old.net_shear_rate_s_inv(stress, density, 900.0),
                delta=max(abs(old.net_shear_rate_s_inv(stress, density, 900.0)), 1.0) * 2.0e-14,
            )

    def test_bounded_temperature_entropy_and_free_barrier_stop(self) -> None:
        entropy = BoundedActivationEntropy(
            reference_kB=-1.0, temperature_amplitude_kB=2.0,
            temperature_width_K=100.0, reference_temperature_K=900.0,
        )
        values = [entropy.entropy_kB(t) for t in (500.0, 900.0, 1500.0)]
        self.assertTrue(all(-3.0 <= value <= 1.0 for value in values))
        invalid = replace(
            self.mechanism,
            entropy=BoundedActivationEntropy(reference_kB=100.0),
        )
        with self.assertRaisesRegex(ValueError, "barrierless/drag"):
            invalid.activation_free_energy_J(5.0e9, 1500.0)

    def test_validity_and_parameter_classification(self) -> None:
        with self.assertRaisesRegex(ValueError, "validity envelope"):
            self.mechanism.net_rate_s_inv(0.2e9, 1600.0, 2.0e16)
        self.assertEqual(PARAMETER_CLASSIFICATION["barrier_form"], "immutable_framework")
        self.assertEqual(
            PARAMETER_CLASSIFICATION["activation_entropy_kB"],
            "generic_development_parameter",
        )
        with self.assertRaisesRegex(ValueError, "zero-stress"):
            replace(self.enthalpy, shape_n=0.8)

    def test_event_frequency_and_glide_velocity_have_separate_scales(self) -> None:
        stress = 0.4e9
        event_frequency = self.mechanism.net_event_frequency_s_inv(
            stress, 900.0, 2.0e16
        )
        event_length_m = 3.0e-10
        velocity = self.mechanism.glide_velocity_m_s(
            stress, 900.0, event_length_m, 2.0e16
        )
        self.assertAlmostEqual(velocity / event_frequency, event_length_m)
        with self.assertRaisesRegex(ValueError, "event_length_m"):
            self.mechanism.glide_velocity_m_s(
                stress, 900.0, 0.0, 2.0e16
            )


if __name__ == "__main__":
    unittest.main()
