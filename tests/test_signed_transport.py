from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from asb_drx.bertin_bcc import (
    BertinBCCParameters,
    BertinBCCState,
    bertin_bcc_step,
    initial_orientation,
)
from asb_drx.signed_transport import (
    SignedTransportParameters,
    SignedTransportState,
    advance_pattern,
    advect_periodic_upwind,
    default_slip_geometry,
    dispersion_rate_s_inv,
    homogeneous_gate_a_step,
    initialize_signed_state,
    kinematics_from_populations,
    load_signed_checkpoint,
    nye_tensor_1d,
    pattern_step,
    plastic_slip_rate_from_signed_flux_s_inv,
    reaction_step,
    save_signed_checkpoint,
    signed_internal_stress_Pa,
    structure_factor_metrics,
    wall_diagnostics,
)


class SignedTransportUnitTests(unittest.TestCase):
    def setUp(self):
        self.parameters = SignedTransportParameters()

    def test_homogeneous_reduction_is_exact_gate_A_step(self):
        gate_a = BertinBCCParameters()
        base = BertinBCCState(
            np.eye(3), initial_orientation((4, 1, 9)),
            np.asarray([4.8, 4.7, 4.9, 4.75]) * 1.0e14,
        )
        states = (base, base, base)
        plus_one = 0.55 * base.densities_m2
        minus_one = base.densities_m2 - plus_one
        plus = np.repeat(plus_one[:, None], 3, axis=1)
        minus = np.repeat(minus_one[:, None], 3, axis=1)
        expected, _ = bertin_bcc_step(base, -2.0e8, -0.001, 300.0, gate_a)
        advanced, next_plus, next_minus = homogeneous_gate_a_step(
            states, plus, minus, -2.0e8, -0.001, 300.0, gate_a
        )
        for state in advanced:
            np.testing.assert_array_equal(state.plastic_deformation_gradient, expected.plastic_deformation_gradient)
            np.testing.assert_array_equal(state.densities_m2, expected.densities_m2)
        expected_density = np.repeat(expected.densities_m2[:, None], 3, axis=1)
        np.testing.assert_allclose(next_plus + next_minus, expected_density, rtol=2e-16)

    def test_zero_flux_preserves_each_signed_population(self):
        state = initialize_signed_state(64, self.parameters, perturbation=0.0)
        updated = pattern_step(state, 0.01, self.parameters)
        np.testing.assert_array_equal(updated.mobile_plus_m2, state.mobile_plus_m2)
        np.testing.assert_array_equal(updated.mobile_minus_m2, state.mobile_minus_m2)

    def test_sign_exchange_symmetry(self):
        state = initialize_signed_state(64, self.parameters)
        swapped_slip, swapped_beta, swapped_rotation = kinematics_from_populations(
            state.mobile_minus_m2, state.mobile_plus_m2,
            self.parameters.domain_m, self.parameters.burgers_m,
        )
        swapped = SignedTransportState(
            state.mobile_minus_m2, state.mobile_plus_m2,
            state.junction_minus_m2, state.junction_plus_m2,
            swapped_slip, swapped_beta, swapped_rotation,
        )
        a = pattern_step(state, 0.002, self.parameters)
        b = pattern_step(swapped, 0.002, self.parameters)
        np.testing.assert_allclose(a.mobile_plus_m2, b.mobile_minus_m2, rtol=2e-14)
        np.testing.assert_allclose(a.mobile_minus_m2, b.mobile_plus_m2, rtol=2e-14)
        np.testing.assert_allclose(signed_internal_stress_Pa(state, self.parameters), -signed_internal_stress_Pa(swapped, self.parameters), rtol=2e-13, atol=1e-7)

    def test_family_permutation_with_geometry_is_invariant(self):
        state = initialize_signed_state(64, self.parameters)
        directions, normals = default_slip_geometry()
        permutation = np.asarray([2, 0, 3, 1])
        slip, beta, _ = kinematics_from_populations(
            state.mobile_plus_m2[permutation], state.mobile_minus_m2[permutation],
            self.parameters.domain_m, self.parameters.burgers_m,
            directions[permutation], normals[permutation],
        )
        np.testing.assert_allclose(beta, state.plastic_distortion, atol=2e-16)
        np.testing.assert_allclose(slip, state.slip[permutation], atol=2e-16)

    def test_BCC_family_basis_and_kinematics_are_cubic_symmetric(self):
        state = initialize_signed_state(64, self.parameters)
        directions, normals = default_slip_geometry()
        cubic = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        rotated_directions = np.einsum("ij,aj->ai", cubic, directions)
        matches = np.argmax(np.abs(rotated_directions @ directions.T), axis=1)
        self.assertEqual(len(set(matches.tolist())), 4)
        np.testing.assert_allclose(
            np.max(np.abs(rotated_directions @ directions.T), axis=1), 1.0, atol=2e-15
        )
        _, transformed_beta, _ = kinematics_from_populations(
            state.mobile_plus_m2, state.mobile_minus_m2,
            self.parameters.domain_m, self.parameters.burgers_m,
            np.einsum("ij,aj->ai", cubic, directions),
            np.einsum("ij,aj->ai", cubic, normals),
        )
        expected = np.einsum("ij,njk,lk->nil", cubic, state.plastic_distortion, cubic)
        np.testing.assert_allclose(transformed_beta, expected, atol=3e-16)

    def test_nye_manufactured_gradient_and_objectivity(self):
        n = 96
        x = np.arange(n) * self.parameters.domain_m / n
        amplitude = 0.03
        mode = 3
        beta = np.zeros((n, 3, 3))
        beta[:, 1, 2] = amplitude * np.sin(2.0 * np.pi * mode * x / self.parameters.domain_m)
        derivative = amplitude * 2.0 * np.pi * mode / self.parameters.domain_m * np.cos(2.0 * np.pi * mode * x / self.parameters.domain_m)
        expected = np.zeros_like(beta)
        for point in range(n):
            expected[point, 1] = -np.cross(np.asarray([1.0, 0.0, 0.0]), np.asarray([0.0, 0.0, derivative[point]]))
        alpha = nye_tensor_1d(beta, self.parameters.domain_m)
        np.testing.assert_allclose(alpha, expected, rtol=2e-13, atol=2e-9)

        angle = 0.37
        q = np.asarray([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
        transformed_beta = np.einsum("ij,njk,lk->nil", q, beta, q)
        transformed = nye_tensor_1d(transformed_beta, self.parameters.domain_m, q @ np.asarray([1.0, 0.0, 0.0]))
        expected_transformed = np.einsum("ij,njk,lk->nil", q, alpha, q)
        np.testing.assert_allclose(transformed, expected_transformed, rtol=3e-13, atol=3e-9)

    def test_population_kinematics_matches_nye_sign_and_units(self):
        state = initialize_signed_state(128, self.parameters)
        alpha = nye_tensor_1d(state.plastic_distortion, self.parameters.domain_m)
        directions, normals = default_slip_geometry()
        kappa = state.mobile_plus_m2 - state.mobile_minus_m2
        expected = np.zeros_like(alpha)
        for family in range(4):
            expected += self.parameters.burgers_m * kappa[family, :, None, None] * np.outer(
                directions[family], np.cross(np.asarray([1.0, 0.0, 0.0]), normals[family])
            )[None, :, :]
        np.testing.assert_allclose(alpha, expected, rtol=2e-12, atol=3e-6)
        self.assertEqual(alpha.shape, (128, 3, 3))

    def test_plastic_slip_rate_is_signed_flux_and_preserves_compatibility(self):
        state = initialize_signed_state(128, self.parameters)
        rate = plastic_slip_rate_from_signed_flux_s_inv(state, self.parameters)
        epsilon = 1.0e-7
        updated = pattern_step(state, epsilon, self.parameters)
        finite_difference = (updated.slip - state.slip) / epsilon
        np.testing.assert_allclose(finite_difference, rate, rtol=2.0e-5, atol=2.0e-5)

    def test_prescribed_velocity_transports_packet(self):
        n = 64
        packet = np.zeros(n)
        packet[9:14] = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0]) * 1.0e14
        dx = self.parameters.domain_m / n
        velocity = 2.5e-4
        shifted = advect_periodic_upwind(packet, velocity, dx / velocity, self.parameters.domain_m)
        np.testing.assert_array_equal(shifted, np.roll(packet, 1))
        self.assertEqual(float(np.sum(shifted)), float(np.sum(packet)))
        self.assertGreaterEqual(float(np.min(shifted)), 0.0)

    def test_annihilation_and_locking_ledgers_close_without_net_burgers_creation(self):
        state = initialize_signed_state(64, self.parameters)
        updated, ledger = reaction_step(
            state, 0.02, self.parameters,
            multiplication_rate_m2_s=2.0e14,
            annihilation_rate_s_inv=3.0,
            locking_rate_s_inv=1.5,
        )
        self.assertLess(abs(ledger.mobile_balance_residual_m_inv), 2e-8 * ledger.mobile_before_m_inv)
        self.assertLess(abs(ledger.junction_balance_residual_m_inv), 2e-8 * ledger.mobile_before_m_inv)
        self.assertLess(abs(ledger.net_burgers_change_m_inv), 2e-8 * ledger.mobile_before_m_inv)
        self.assertLess(ledger.maximum_family_signed_change_m_inv, 2e-8 * ledger.mobile_before_m_inv)
        self.assertLess(ledger.burgers_vector_residual, 2e-8 * ledger.mobile_before_m_inv * self.parameters.burgers_m)
        self.assertGreater(ledger.annihilation_removed_m_inv, 0.0)
        self.assertGreater(ledger.locking_mobile_to_junction_m_inv, 0.0)
        self.assertTrue(np.all(updated.mobile_plus_m2 >= 0.0))
        self.assertTrue(np.all(updated.mobile_minus_m2 >= 0.0))

    def test_balanced_high_total_density_is_not_a_wall(self):
        high = replace(self.parameters, reference_density_m2=2.0e16)
        state = initialize_signed_state(64, high, perturbation=0.0)
        x = np.arange(64) * high.domain_m / 64
        total = high.reference_density_m2 * (1.0 + 0.4 * np.cos(2.0 * np.pi * 8 * x / high.domain_m))
        plus = np.repeat((0.5 * total)[None, :], 4, axis=0)
        minus = plus.copy()
        slip, beta, rotation = kinematics_from_populations(plus, minus, high.domain_m, high.burgers_m)
        zeros = np.zeros_like(plus)
        state = SignedTransportState(plus, minus, zeros, zeros, slip, beta, rotation)
        diagnostic = wall_diagnostics(state, high)
        self.assertEqual(diagnostic["gnd_rms_m2"], 0.0)
        self.assertFalse(diagnostic["physical_wall"])
        self.assertGreater(diagnostic["total_density_contrast"], 0.1)
        self.assertEqual(diagnostic["classification"], "total_density_band_only")

    def test_differential_slip_produces_compatible_gnd_orientation_wall(self):
        n = 128
        x = np.arange(n) * self.parameters.domain_m / n
        q = np.zeros((4, n))
        q[0] = 0.25 * np.sin(2.0 * np.pi * 8.0 * x / self.parameters.domain_m)
        rho = self.parameters.reference_density_m2
        plus = 0.5 * rho * (1.0 + q)
        minus = 0.5 * rho * (1.0 - q)
        slip, beta, rotation = kinematics_from_populations(plus, minus, self.parameters.domain_m, self.parameters.burgers_m)
        zeros = np.zeros_like(plus)
        state = SignedTransportState(plus, minus, zeros, zeros, slip, beta, rotation)
        diagnostic = wall_diagnostics(state, self.parameters)
        self.assertTrue(diagnostic["physical_wall"])
        self.assertGreater(diagnostic["gnd_rms_m2"], 1.0e13)
        self.assertGreater(diagnostic["orientation_gradient_rms_m_inv"], 100.0)
        self.assertEqual(diagnostic["wall_spacing_m"], self.parameters.selected_wavelength_m)

    def test_no_grain_allocation_surface_exists(self):
        state = initialize_signed_state(64, self.parameters)
        final = advance_pattern(state, 0.1, 0.002, self.parameters)
        self.assertEqual(state.physical_grain_count, 1)
        self.assertEqual(final.physical_grain_count, 1)
        self.assertFalse(hasattr(final, "grain_labels"))
        self.assertFalse(hasattr(final, "phase_fields"))


class SignedTransportPatternGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parameters = SignedTransportParameters()
        cls.results = {}
        for n in (64, 128, 256):
            cls.results[(n, 0.002)] = advance_pattern(
                initialize_signed_state(n, cls.parameters), 1.0, 0.002, cls.parameters
            )
        cls.results[(128, 0.004)] = advance_pattern(
            initialize_signed_state(128, cls.parameters), 1.0, 0.004, cls.parameters
        )

    def test_dispersion_and_nonlinear_pattern_select_finite_wall_spacing(self):
        p = self.parameters
        k = 2.0 * np.pi * np.fft.rfftfreq(256, d=p.domain_m / 256)
        fastest = k[int(np.argmax(dispersion_rate_s_inv(k, p)))]
        self.assertAlmostEqual(fastest, p.selected_wavenumber_m_inv, places=7)
        initial = wall_diagnostics(initialize_signed_state(128, p), p)
        final = wall_diagnostics(self.results[(128, 0.002)], p)
        self.assertTrue(final["physical_wall"])
        self.assertEqual(final["wall_spacing_m"], p.selected_wavelength_m)
        self.assertGreater(final["gnd_rms_m2"], 4.0 * initial["gnd_rms_m2"])
        self.assertGreater(final["spectral_peak_fraction"], 0.45)

    def test_grid_and_timestep_refinement_below_five_percent(self):
        diagnostics = {
            key: wall_diagnostics(value, self.parameters)
            for key, value in self.results.items()
        }
        grid_coarse = diagnostics[(128, 0.002)]
        grid_fine = diagnostics[(256, 0.002)]
        time_coarse = diagnostics[(128, 0.004)]
        time_fine = diagnostics[(128, 0.002)]
        fields = ("wall_spacing_m", "gnd_rms_m2", "orientation_gradient_rms_m_inv", "spectral_peak_fraction")
        grid_changes = [abs(grid_coarse[name] - grid_fine[name]) / abs(grid_fine[name]) for name in fields]
        time_changes = [abs(time_coarse[name] - time_fine[name]) / abs(time_fine[name]) for name in fields]
        self.assertLess(max(grid_changes), 0.05)
        self.assertLess(max(time_changes), 0.05)

    def test_checkpoint_restart_is_bitwise(self):
        initial = initialize_signed_state(128, self.parameters)
        continuous = self.results[(128, 0.002)]
        first = advance_pattern(initial, 0.5, 0.002, self.parameters)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "signed.npz"
            save_signed_checkpoint(path, first)
            restored = load_signed_checkpoint(path)
        segmented = advance_pattern(restored, 0.5, 0.002, self.parameters)
        for name in (
            "mobile_plus_m2", "mobile_minus_m2", "junction_plus_m2",
            "junction_minus_m2", "slip", "plastic_distortion", "orientation",
        ):
            self.assertTrue(np.array_equal(getattr(continuous, name), getattr(segmented, name)))
        self.assertEqual(continuous.time_s, segmented.time_s)
        self.assertEqual(continuous.accepted_steps, segmented.accepted_steps)


if __name__ == "__main__":
    unittest.main()
