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
)
from asb_drx.driven_cdd import (
    DrivenCDDParameters,
    DrivenCDDState,
    _upwind_step,
    correlation_stresses_Pa,
    derived_nye_tensor,
    driven_cdd_step,
    driven_structure_diagnostics,
    evaluate_driven_cdd,
    homogeneous_dispersion_snapshot,
    initialize_driven_cdd,
    load_driven_checkpoint,
    save_driven_checkpoint,
)


class DrivenCDDUnitTests(unittest.TestCase):
    def setUp(self):
        self.gate_a = BertinBCCParameters()
        self.parameters = DrivenCDDParameters()

    def test_zero_load_has_no_flux_or_finite_mode_instability(self):
        state = initialize_driven_cdd(32, self.parameters, self.gate_a)
        response = evaluate_driven_cdd(state, self.parameters, self.gate_a)
        self.assertEqual(float(np.max(np.abs(response.actual_shear_rates_s_inv))), 0.0)
        self.assertFalse(np.any(response.patterning_active))
        dispersion = homogeneous_dispersion_snapshot(
            state, self.parameters, self.gate_a
        )
        self.assertFalse(any(item["unstable"] for item in dispersion["families"]))

    def test_homogeneous_reduction_is_exact_gate_a(self):
        state = initialize_driven_cdd(16, self.parameters, self.gate_a)
        local = BertinBCCState(
            state.plastic_deformation_gradient[0], state.initial_orientation[0],
            state.mobile_plus_m2[:, 0] + state.mobile_minus_m2[:, 0],
        )
        expected, _ = bertin_bcc_step(
            local, 2.0e8, 1.0e-3, 300.0, self.gate_a
        )
        actual, _, ledger = driven_cdd_step(
            state, 2.0e8, 1.0e-3, self.parameters, self.gate_a
        )
        np.testing.assert_array_equal(
            actual.plastic_deformation_gradient[0],
            expected.plastic_deformation_gradient,
        )
        np.testing.assert_array_equal(
            actual.mobile_plus_m2[:, 0] + actual.mobile_minus_m2[:, 0],
            expected.densities_m2,
        )
        self.assertLess(abs(ledger.total_balance_residual_m_inv), 1.0e-12 * ledger.mobile_before_m_inv)
        self.assertEqual(ledger.maximum_family_signed_residual_m_inv, 0.0)

    def test_total_and_signed_noise_are_independent_and_mode_can_be_removed(self):
        state = initialize_driven_cdd(
            64, self.parameters, self.gate_a, total_noise_amplitude=0.01,
            signed_noise_amplitude=0.0, seed=4, remove_mode=9,
        )
        total = state.mobile_plus_m2 + state.mobile_minus_m2
        kappa = state.mobile_plus_m2 - state.mobile_minus_m2
        self.assertGreater(float(np.std(total)), 0.0)
        np.testing.assert_allclose(kappa, 0.0, atol=0.2)
        self.assertLess(float(np.max(np.abs(np.fft.fft(total, axis=1)[:, 9]))), 2.0)

    def test_density_weighted_transport_is_positive_and_conservative(self):
        density = np.zeros((4, 32))
        density[:, 5:8] = np.asarray([1.0, 2.0, 1.0])[None, :] * 1.0e14
        velocity = np.full_like(density, 3.0)
        dx = self.parameters.domain_m / 32
        updated = _upwind_step(density, velocity, 0.25 * dx / 3.0, dx)
        self.assertGreaterEqual(float(np.min(updated)), 0.0)
        np.testing.assert_allclose(np.sum(updated, axis=1), np.sum(density, axis=1), rtol=2.0e-16)

    def test_sign_exchange_reverses_odd_stresses_only(self):
        state = initialize_driven_cdd(
            32, self.parameters, self.gate_a,
            total_noise_amplitude=0.01, signed_noise_amplitude=0.01, seed=8,
        )
        swapped = DrivenCDDState(
            state.plastic_deformation_gradient, state.initial_orientation,
            state.mobile_minus_m2, state.mobile_plus_m2,
            state.locked_minus_m2, state.locked_plus_m2,
            state.temperature_K,
        )
        sc, back, diff = correlation_stresses_Pa(state, self.parameters, self.gate_a)
        sc_s, back_s, diff_s = correlation_stresses_Pa(swapped, self.parameters, self.gate_a)
        np.testing.assert_allclose(sc_s, -sc, rtol=2.0e-14, atol=1.0e-6)
        np.testing.assert_allclose(back_s, -back, rtol=2.0e-14, atol=1.0e-6)
        np.testing.assert_allclose(diff_s, diff, rtol=2.0e-14, atol=1.0e-6)

    def test_pair_sources_and_locking_close_line_and_burgers_ledgers(self):
        parameters = replace(
            self.parameters, same_family_lock_rate_s_inv=2.0e8,
            unlock_rate_s_inv=1.0e8, cross_family_lock_rate_s_inv=3.0e8,
        )
        state = initialize_driven_cdd(
            16, parameters, self.gate_a,
            total_noise_amplitude=0.002, signed_noise_amplitude=0.002, seed=2,
        )
        advanced, _, ledger = driven_cdd_step(
            state, 2.0e8, 1.0e-3, parameters, self.gate_a
        )
        self.assertGreaterEqual(ledger.pair_multiplication_m_inv, 0.0)
        self.assertGreaterEqual(ledger.pair_annihilation_m_inv, 0.0)
        self.assertLess(abs(ledger.total_balance_residual_m_inv), 2.0e-12 * ledger.mobile_before_m_inv)
        self.assertLess(ledger.maximum_family_signed_residual_m_inv, 2.0e-12 * ledger.mobile_before_m_inv)
        self.assertLess(ledger.vector_burgers_residual, 2.0e-12 * ledger.mobile_before_m_inv * self.gate_a.burgers_m)
        self.assertGreaterEqual(float(np.min(advanced.mobile_plus_m2)), 0.0)
        self.assertGreater(float(np.sum(advanced.locked_plus_m2 + advanced.locked_minus_m2)), 0.0)

    def test_differential_signed_content_has_nye_and_orientation_gradient(self):
        state = initialize_driven_cdd(
            64, self.parameters, self.gate_a,
            total_noise_amplitude=0.0, signed_noise_amplitude=0.03, seed=11,
        )
        alpha = derived_nye_tensor(state, self.parameters.domain_m)
        diagnostic = driven_structure_diagnostics(
            state, self.parameters, self.gate_a
        )
        self.assertGreater(float(np.linalg.norm(alpha)), 0.0)
        self.assertGreater(diagnostic["gnd_rms_m2"], 0.0)
        self.assertGreater(diagnostic["orientation_gradient_rms_m_inv"], 0.0)

    def test_balanced_high_density_band_is_not_a_wall(self):
        state = initialize_driven_cdd(32, self.parameters, self.gate_a)
        x = np.arange(32)
        band = 1.0 + 0.4 * np.cos(2.0 * np.pi * 5.0 * x / 32)
        plus = state.mobile_plus_m2 * band[None, :]
        balanced = DrivenCDDState(
            state.plastic_deformation_gradient, state.initial_orientation,
            plus, plus, state.locked_plus_m2, state.locked_minus_m2,
            state.temperature_K,
        )
        diagnostic = driven_structure_diagnostics(
            balanced, self.parameters, self.gate_a
        )
        self.assertEqual(diagnostic["classification"], "total-density modulation")
        self.assertEqual(diagnostic["gnd_rms_m2"], 0.0)
        self.assertFalse(diagnostic["compatible_LAGB_candidate"])

    def test_checkpoint_restart_is_exact(self):
        state = initialize_driven_cdd(
            16, self.parameters, self.gate_a,
            total_noise_amplitude=0.002, signed_noise_amplitude=0.001, seed=5,
        )
        state, _, _ = driven_cdd_step(
            state, 2.0e8, 5.0e-4, self.parameters, self.gate_a
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.npz"
            save_driven_checkpoint(path, state)
            restored = load_driven_checkpoint(path)
        for name in (
            "plastic_deformation_gradient", "initial_orientation",
            "mobile_plus_m2", "mobile_minus_m2", "locked_plus_m2",
            "locked_minus_m2", "temperature_K",
        ):
            np.testing.assert_array_equal(getattr(restored, name), getattr(state, name))
        self.assertEqual(restored.axial_true_strain, state.axial_true_strain)
        self.assertEqual(restored.time_s, state.time_s)

    def test_driven_dispersion_has_emergent_finite_mode_and_no_label_surface(self):
        state = initialize_driven_cdd(64, self.parameters, self.gate_a)
        for _ in range(20):
            state, _, _ = driven_cdd_step(
                state, 1.0e6, 2.5e-4, self.parameters, self.gate_a
            )
        records = homogeneous_dispersion_snapshot(
            state, self.parameters, self.gate_a
        )["families"]
        unstable = [item for item in records if item["unstable"]]
        self.assertGreaterEqual(len(unstable), 1)
        self.assertTrue(all(0 < item["fastest_discrete_mode"] < 32 for item in unstable))
        self.assertFalse(hasattr(state, "grain_labels"))
        self.assertFalse(hasattr(state, "phase_fields"))
        self.assertEqual(state.physical_grain_count, 1)


if __name__ == "__main__":
    unittest.main()
