from __future__ import annotations

import unittest

import numpy as np

from asb_drx.coupled_stability import (
    HomogeneousCoupledState,
    coupled_mode_rhs,
    full_coupled_stability_mode,
    net_common_stress_rate_tangents,
)
from asb_drx.fixtures import SingleGliderDDDParameterization
from asb_drx.spatial_coupled import SpatialCoupledParameters


class FullCoupledStabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        fixture = SingleGliderDDDParameterization()
        self.law = fixture.law()
        self.parameters = fixture.spatial_parameters()
        peak = self.law.net_peak(950.0, 450.0)
        self.state = HomogeneousCoupledState(
            peak.macroscopic_strength_Pa,
            950.0,
            0.8 * peak.density_m2,
            0.55 * peak.density_m2,
            0.3,
        )
        self.kx = 2.0e5
        self.ky = 4.0e5

    def test_net_rate_tangents_match_centered_differences(self) -> None:
        a = net_common_stress_rate_tangents(
            self.law, self.state.macroscopic_stress_Pa,
            self.state.parent_density_m2, self.state.temperature_K,
        )
        sigma = self.state.macroscopic_stress_Pa
        rho = self.state.parent_density_m2
        T = self.state.temperature_K

        def rate(s: float, r: float, t: float) -> float:
            return self.law.net_shear_rate_s_inv(
                s / self.law.taylor_ratio(r), r, t
            )

        ds = sigma * 1.0e-6
        dr = rho * 1.0e-6
        dT = T * 1.0e-6
        numerical = (
            (rate(sigma + ds, rho, T) - rate(sigma - ds, rho, T)) / (2.0 * ds),
            (rate(sigma, rho, T + dT) - rate(sigma, rho, T - dT)) / (2.0 * dT),
            (rate(sigma, rho + dr, T) - rate(sigma, rho - dr, T)) / (2.0 * dr),
        )
        analytical = (
            a.macroscopic_stress_tangent_Pa_inv_s_inv,
            a.temperature_tangent_K_inv_s_inv,
            a.density_tangent_m2_s_inv,
        )
        self.assertTrue(np.allclose(analytical, numerical, rtol=2.0e-8, atol=0.0))

    def test_full_operator_matches_finite_difference_rhs(self) -> None:
        mode = full_coupled_stability_mode(
            self.law, self.state, self.kx, self.ky, self.parameters
        )
        scales = np.asarray((1.0e-8, 1.0e-3,
                             self.state.parent_density_m2 * 1.0e-7,
                             self.state.child_density_m2 * 1.0e-7, 1.0e-7))
        numerical = np.empty((5, 5))
        for column, step in enumerate(scales):
            delta = np.zeros(5)
            delta[column] = step
            plus = coupled_mode_rhs(
                delta, self.law, self.state, self.kx, self.ky, self.parameters
            )
            minus = coupled_mode_rhs(
                -delta, self.law, self.state, self.kx, self.ky, self.parameters
            )
            numerical[:, column] = (plus - minus) / (2.0 * step)
        self.assertTrue(np.allclose(mode.jacobian, numerical, rtol=2.0e-5, atol=1.0e-10))

    def test_antiplane_orientation_changes_only_mechanical_projection(self) -> None:
        longitudinal = full_coupled_stability_mode(
            self.law, self.state, 4.0e5, 0.0, self.parameters
        )
        transverse = full_coupled_stability_mode(
            self.law, self.state, 0.0, 4.0e5, self.parameters
        )
        self.assertEqual(longitudinal.antiplane_projection, 0.0)
        self.assertEqual(transverse.antiplane_projection, 1.0)
        self.assertTrue(np.array_equal(longitudinal.jacobian[:, 1:], transverse.jacobian[:, 1:]))
        self.assertFalse(np.array_equal(longitudinal.jacobian[:, 0], transverse.jacobian[:, 0]))

    def test_storage_cap_branch_and_kink_are_explicit(self) -> None:
        E = self.parameters.stored_line_energy_J_m
        capped_parameters = SpatialCoupledParameters(
            self.parameters.shear_modulus_Pa,
            self.parameters.volumetric_heat_capacity_J_m3_K,
            self.parameters.thermal_conductivity_W_m_K,
            E,
            2.0 * self.state.macroscopic_stress_Pa / E,
            self.parameters.pair_penalty_J_m3,
            self.parameters.gradient_coefficient_J_m,
            self.parameters.phase_mobility_m3_J_s,
        )
        capped = full_coupled_stability_mode(
            self.law, self.state, self.kx, self.ky, capped_parameters
        )
        self.assertEqual(capped.storage_branch, "capped")
        kink_parameters = SpatialCoupledParameters(
            self.parameters.shear_modulus_Pa,
            self.parameters.volumetric_heat_capacity_J_m3_K,
            self.parameters.thermal_conductivity_W_m_K,
            E,
            self.state.macroscopic_stress_Pa / E,
            self.parameters.pair_penalty_J_m3,
            self.parameters.gradient_coefficient_J_m,
            self.parameters.phase_mobility_m3_J_s,
        )
        with self.assertRaisesRegex(ValueError, "no unique Jacobian"):
            full_coupled_stability_mode(
                self.law, self.state, self.kx, self.ky, kink_parameters
            )


if __name__ == "__main__":
    unittest.main()
