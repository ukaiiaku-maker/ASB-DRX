from __future__ import annotations

import unittest

import numpy as np

from asb_drx.antiplane import midpoint_work_ledger_J_m3, solve_periodic_antiplane


class AntiplaneEquilibriumTests(unittest.TestCase):
    def setUp(self) -> None:
        self.points = 32
        self.dx_m = 5.0e-7
        self.modulus_Pa = 8.0e10
        coordinate = np.linspace(0.0, 2.0 * np.pi, self.points, endpoint=False)
        self.x_pattern = np.sin(coordinate)[None, :] * np.ones((self.points, 1))
        self.y_pattern = np.sin(coordinate)[:, None] * np.ones((1, self.points))

    def test_uniform_state_recovers_common_stress_and_energy(self) -> None:
        plastic = np.full((self.points, self.points), 0.03)
        result = solve_periodic_antiplane(0.05, plastic, self.modulus_Pa, self.dx_m)
        expected = self.modulus_Pa * 0.02
        self.assertTrue(np.allclose(result.stress_x_Pa, expected, rtol=0.0, atol=1.0e-6))
        self.assertTrue(np.allclose(result.stress_y_Pa, 0.0, rtol=0.0, atol=1.0e-6))
        self.assertAlmostEqual(result.elastic_energy_J_m3, 0.5 * self.modulus_Pa * 0.02**2)

    def test_longitudinal_plastic_mode_is_relaxed_by_periodic_displacement(self) -> None:
        result = solve_periodic_antiplane(0.0, 0.01 * self.x_pattern, self.modulus_Pa, self.dx_m)
        self.assertLess(float(np.max(np.abs(result.stress_x_Pa))), 1.0e-5)
        self.assertLess(float(np.max(np.abs(result.stress_y_Pa))), 1.0e-5)

    def test_transverse_band_mode_creates_local_equilibrated_stress(self) -> None:
        plastic = 0.01 * self.y_pattern
        result = solve_periodic_antiplane(0.0, plastic, self.modulus_Pa, self.dx_m)
        self.assertTrue(np.allclose(result.stress_x_Pa, -self.modulus_Pa * plastic, rtol=2.0e-14, atol=1.0e-5))
        self.assertLess(result.equilibrium_residual_Pa_m_inv, 1.0)
        self.assertAlmostEqual(result.mean_stress_Pa, 0.0, delta=1.0e-7)

    def test_rotation_covariant_projection_swaps_stress_components(self) -> None:
        coordinate = np.linspace(0.0, 2.0 * np.pi, self.points, endpoint=False)
        diagonal = 0.01 * np.sin(coordinate[:, None] + coordinate[None, :])
        result = solve_periodic_antiplane(0.0, diagonal, self.modulus_Pa, self.dx_m)
        self.assertTrue(np.allclose(result.stress_x_Pa, -result.stress_y_Pa, rtol=2.0e-13, atol=1.0e-5))

    def test_midpoint_ledger_closes_exact_quadratic_energy_change(self) -> None:
        old_plastic = 0.002 * self.y_pattern
        increment = 0.001 * (self.x_pattern + 0.5 * self.y_pattern)
        old = solve_periodic_antiplane(0.01, old_plastic, self.modulus_Pa, self.dx_m)
        new = solve_periodic_antiplane(0.012, old_plastic + increment, self.modulus_Pa, self.dx_m)
        external, plastic, elastic, closure = midpoint_work_ledger_J_m3(
            old, new, 0.002, increment
        )
        scale = max(abs(external), abs(plastic), abs(elastic), 1.0)
        self.assertLess(abs(closure), 5.0e-14 * scale)

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            solve_periodic_antiplane(0.0, np.zeros(8), self.modulus_Pa, self.dx_m)
        state = solve_periodic_antiplane(0.0, np.zeros((8, 8)), self.modulus_Pa, self.dx_m)
        with self.assertRaises(ValueError):
            midpoint_work_ledger_J_m3(state, state, 0.0, np.zeros((4, 4)))


if __name__ == "__main__":
    unittest.main()
