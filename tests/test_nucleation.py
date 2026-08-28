from __future__ import annotations

import math
import unittest

from asb_drx.nucleation import (
    BOLTZMANN_J_K,
    CylindricalNucleationParameters,
    evaluate_candidate,
)


class NucleationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = CylindricalNucleationParameters(0.1, 1.0e8, 1.0e-9, 1.0e20)
        self.common = {
            "candidate_radius_m": 1.5 * self.parameters.critical_radius_m,
            "minimum_resolved_radius_m": 0.8 * self.parameters.critical_radius_m,
            "candidate_orientation_rad": math.radians(12.0),
            "parent_orientation_rad": 0.0,
            "minimum_misorientation_rad": math.radians(5.0),
            "symmetry_order": 4,
            "temperature_K": 1000.0,
            "eligible_area_m2": 1.0e-12,
            "interval_s": 1.0e-3,
        }

    def test_critical_radius_is_stationary_barrier_and_escape_radius_is_zero(self) -> None:
        critical = self.parameters.critical_radius_m
        self.assertEqual(self.parameters.excess_energy_derivative_J_m(critical), 0.0)
        self.assertAlmostEqual(
            self.parameters.excess_energy_J(critical), self.parameters.barrier_J
        )
        self.assertAlmostEqual(
            self.parameters.excess_energy_J(self.parameters.escape_radius_m), 0.0
        )
        self.assertGreater(self.parameters.excess_energy_J(0.5 * critical), 0.0)
        self.assertLess(self.parameters.excess_energy_J(2.5 * critical), 0.0)

    def test_event_probability_matches_poisson_arrhenius_expression(self) -> None:
        probability = self.parameters.event_probability(1000.0, 1.0e-12, 1.0e-3)
        rate = self.parameters.areal_attempt_rate_m2_s * math.exp(
            -self.parameters.barrier_J / (BOLTZMANN_J_K * 1000.0)
        )
        expected = -math.expm1(-rate * 1.0e-12 * 1.0e-3)
        self.assertAlmostEqual(probability, expected)
        self.assertGreater(probability, 0.0)
        self.assertLess(probability, 1.0)

    def test_probability_increases_with_temperature_and_driving_energy(self) -> None:
        cold = self.parameters.event_probability(800.0, 1.0e-12, 1.0e-3)
        hot = self.parameters.event_probability(1200.0, 1.0e-12, 1.0e-3)
        stronger = CylindricalNucleationParameters(0.1, 2.0e8, 1.0e-9, 1.0e20)
        driven = stronger.event_probability(800.0, 1.0e-12, 1.0e-3)
        self.assertGreater(hot, cold)
        self.assertGreater(driven, cold)

    def test_candidate_acceptance_uses_external_draw_deterministically(self) -> None:
        probability = self.parameters.event_probability(1000.0, 1.0e-12, 1.0e-3)
        accepted = evaluate_candidate(
            self.parameters, uniform_draw=0.5 * probability, **self.common
        )
        rejected = evaluate_candidate(
            self.parameters,
            uniform_draw=min(0.999999999999, 1.5 * probability),
            **self.common,
        )
        self.assertTrue(accepted.accepted)
        self.assertEqual(accepted.reason, "accepted")
        self.assertFalse(rejected.accepted)
        self.assertEqual(rejected.reason, "thermal_draw_rejected")

    def test_resolution_subcritical_and_misorientation_rejections_are_distinct(self) -> None:
        unresolved = evaluate_candidate(
            self.parameters,
            uniform_draw=0.0,
            **{**self.common, "minimum_resolved_radius_m": 2.0 * self.parameters.critical_radius_m},
        )
        subcritical = evaluate_candidate(
            self.parameters,
            uniform_draw=0.0,
            **{**self.common, "candidate_radius_m": 0.9 * self.parameters.critical_radius_m},
        )
        equivalent = evaluate_candidate(
            self.parameters,
            uniform_draw=0.0,
            **{**self.common, "candidate_orientation_rad": 0.5 * math.pi + math.radians(2.0)},
        )
        self.assertEqual(unresolved.reason, "unresolved_support")
        self.assertEqual(subcritical.reason, "subcritical_radius")
        self.assertEqual(equivalent.reason, "insufficient_misorientation")

    def test_invalid_draw_and_zero_prefactor_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_candidate(self.parameters, uniform_draw=1.0, **self.common)
        with self.assertRaises(ValueError):
            CylindricalNucleationParameters(0.1, 1.0e8, 1.0e-9, 0.0)


if __name__ == "__main__":
    unittest.main()
