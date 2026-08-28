from __future__ import annotations

import unittest

import numpy as np
from scipy.optimize import brentq

from asb_drx.fixtures import SingleGliderDDDParameterization
from asb_drx.implicit_flow import backward_euler_antiplane_flow


class ImplicitAntiplaneFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = SingleGliderDDDParameterization()
        self.law = self.fixture.law()
        self.G = self.fixture.shear_modulus_Pa
        self.points = 8
        self.dx = 2.0e-6
        self.plastic = np.zeros((self.points, self.points))
        self.temperature = np.full((self.points, self.points), 950.0)
        self.density = np.empty((2, self.points, self.points))
        self.density[0] = 1.0e17
        self.density[1] = 2.0e16
        self.weights = np.zeros_like(self.density)
        self.weights[0] = 1.0

    def test_uniform_field_matches_scalar_backward_euler_root(self) -> None:
        initial_stress = 8.0e8
        applied = initial_stress / self.G
        applied_increment = 2.0e-3
        dt = 2.0e-6
        result = backward_euler_antiplane_flow(
            applied, self.plastic, applied_increment, self.density,
            self.temperature, self.weights, dt, self.dx, self.G, self.law,
        )

        def residual(increment: float) -> float:
            stress = initial_stress + self.G * (applied_increment - increment)
            obstacle = abs(stress) / self.law.taylor_ratio(1.0e17)
            rate = self.law.net_shear_rate_s_inv(obstacle, 1.0e17, 950.0)
            return increment - dt * rate

        expected = brentq(residual, 0.0, applied_increment + initial_stress / self.G)
        self.assertTrue(
            np.allclose(result.plastic_increment, expected, rtol=2.0e-9, atol=1.0e-14)
        )
        self.assertLess(result.maximum_residual, 1.0e-10)

    def test_stiff_step_does_not_overshoot_stress_reversal(self) -> None:
        initial_stress = 1.2e9
        result = backward_euler_antiplane_flow(
            initial_stress / self.G,
            self.plastic,
            0.0,
            self.density,
            self.temperature,
            self.weights,
            1.0e-3,
            self.dx,
            self.G,
            self.law,
        )
        self.assertGreaterEqual(float(np.min(result.equilibrium.stress_x_Pa)), 0.0)
        self.assertLess(float(np.max(result.plastic_increment)), initial_stress / self.G)

    def test_invalid_timestep_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            backward_euler_antiplane_flow(
                0.01, self.plastic, 0.0, self.density, self.temperature,
                self.weights, 0.0, self.dx, self.G, self.law,
            )


if __name__ == "__main__":
    unittest.main()
