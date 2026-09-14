import unittest
from dataclasses import replace

import numpy as np

from full_model.production.moving_front import (
    DefectState, activated_front_fraction, advance_front,
    apply_common_constitutive_increment, initialize_sparse_front,
    reconstruct_mixture, signed_density, state_arrays, state_from_checkpoint,
    state_metadata_json,
)
from full_model.production.arrhenius_kinetics import ActivatedProcess


class MovingFrontTest(unittest.TestCase):
    def parent(self):
        return DefectState(
            np.full((4, 4, 2), 4e14), np.full((4, 4, 2), 3e14),
            np.full((4, 4, 2), 2e14), np.full((4, 4), 1e14))

    def state(self):
        return initialize_sparse_front(
            self.parent(), np.zeros((4, 4)), 5e14, 0, 1)

    def advance(self, state, chi):
        return advance_front(
            state, chi, cell_area_m2=1e-14,
            represented_thickness_m=1e-6, line_energy_J_m=2e-9,
            boundary_storage_fraction=0.1, sink_fraction=0.05)

    def test_stationary_interface_has_zero_processing(self):
        state = self.state()
        updated, mixture = self.advance(state, state.chi)
        self.assertEqual(updated.ledger.parent_line_processed_m, 0.0)
        self.assertTrue(np.array_equal(mixture.rp, state.parent.rp))

    def test_advance_closes_line_signed_burgers_and_energy(self):
        state = self.state(); chi = np.zeros((4, 4)); chi[:, :2] = 1.0
        updated, mixture = self.advance(state, chi)
        ledger = updated.ledger
        self.assertLessEqual(abs(ledger.line_closure_m), 1e-20)
        self.assertEqual(ledger.signed_burgers_change_m2, 0.0)
        self.assertEqual(ledger.line_energy_released_J, ledger.heat_released_J)
        self.assertTrue(np.array_equal(
            signed_density(mixture), signed_density(state.parent)))
        self.assertTrue(np.all(mixture.rp[:, 2:] == state.parent.rp[:, 2:]))

    def test_equal_retreat_and_readvance_does_not_double_process(self):
        state = self.state(); advanced = np.zeros((4, 4)); advanced[:, :2] = 1.0
        state, _ = self.advance(state, advanced)
        first = state.ledger
        state, retreat = self.advance(state, np.zeros((4, 4)))
        expected = reconstruct_mixture(state)
        self.assertTrue(np.array_equal(retreat.rp, expected.rp))
        self.assertTrue(np.all(retreat.rp[:, :2] < state.parent.rp[:, :2]))
        self.assertEqual(state.ledger, first)
        state, _ = self.advance(state, advanced)
        self.assertEqual(state.ledger, first)

    def test_monotonic_advance_processes_only_increment(self):
        state = self.state(); half = np.full((4, 4), 0.5)
        state, _ = self.advance(state, half); first = state.ledger.swept_volume_m3
        state, _ = self.advance(state, np.ones((4, 4)))
        self.assertAlmostEqual(state.ledger.swept_volume_m3, 2.0*first)

    def test_subcritical_collapse_leaves_recovered_wake_without_reversing_heat(self):
        state = self.state(); chi = np.zeros((4, 4)); chi[1:3, 1:3] = 1.0
        state, _ = self.advance(state, chi); heat = state.ledger.heat_released_J
        state, mixture = self.advance(state, np.zeros((4, 4)))
        self.assertTrue(np.all(mixture.rp[1:3, 1:3] < state.parent.rp[1:3, 1:3]))
        self.assertTrue(np.array_equal(
            mixture.rp[0, :], state.parent.rp[0, :]))
        self.assertEqual(state.ledger.heat_released_J, heat)

    def test_restart_during_front_cycle_is_exact(self):
        state = self.state(); chi = np.full((4, 4), 0.5)
        state, _ = self.advance(state, chi)
        restored = state_from_checkpoint(state_metadata_json(state), state_arrays(state))
        for key, value in state_arrays(state).items():
            self.assertTrue(np.array_equal(value, state_arrays(restored)[key]))
        self.assertEqual(state.ledger, restored.ledger)
        final_a, mix_a = self.advance(state, np.ones((4, 4)))
        final_b, mix_b = self.advance(restored, np.ones((4, 4)))
        self.assertEqual(final_a.ledger, final_b.ledger)
        self.assertTrue(np.array_equal(mix_a.forest, mix_b.forest))

    def test_common_increment_reconstructs_and_has_pure_phase_limits(self):
        state = self.state()
        old = reconstruct_mixture(state)
        updated = DefectState(
            old.rp+2.0, old.rm+3.0, old.forest+4.0, old.wall+5.0)
        advanced = apply_common_constitutive_increment(state, updated)
        mixture = reconstruct_mixture(advanced)
        for actual, expected in zip(
                (mixture.rp, mixture.rm, mixture.forest, mixture.wall),
                (updated.rp, updated.rm, updated.forest, updated.wall)):
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.25)
        parent = reconstruct_mixture(advanced, np.zeros_like(state.chi))
        np.testing.assert_array_equal(parent.rp, advanced.parent.rp)
        swept, child_mix = self.advance(state, np.ones_like(state.chi))
        np.testing.assert_array_equal(child_mix.rm, swept.child.rm)

    def test_common_increment_projects_all_populations_without_mixture_clipping(self):
        state = self.state()
        old = reconstruct_mixture(state)
        updated = DefectState(old.rp, old.rm, old.forest, np.zeros_like(old.wall))
        projected = apply_common_constitutive_increment(state, updated)
        np.testing.assert_array_equal(
            reconstruct_mixture(projected).wall, updated.wall)
        mobile_zero = DefectState(
            np.zeros_like(old.rp), old.rm, old.forest, old.wall)
        projected = apply_common_constitutive_increment(state, mobile_zero)
        np.testing.assert_array_equal(
            reconstruct_mixture(projected).rp, mobile_zero.rp)

    def test_differential_signed_state_is_carried_by_boundary(self):
        state = self.state()
        altered = DefectState(
            np.zeros_like(state.child.rp), state.child.rm,
            state.child.forest, state.child.wall)
        state = replace(state, child=altered, recovered_wake=altered)
        chi = np.ones_like(state.chi)
        state, _ = self.advance(state, chi)
        self.assertEqual(state.ledger.signed_burgers_change_m2, 0.0)
        self.assertGreater(np.max(np.abs(state.boundary_signed_density_m2)), 0.0)
        self.assertGreaterEqual(
            np.min(state.boundary_line_density_m2
                   -np.sum(np.abs(state.boundary_signed_density_m2), axis=2)),
            -1.0)

    def test_exp_floor_front_fraction_varies_with_stress_temperature_entropy(self):
        process = ActivatedProcess(
            "front", 2.0e6, entropy_over_kB=0.0, drag_rate_s=2.0e6)
        kwargs = dict(dt_s=1e-6, h0_J=1.2e-19,
                      critical_stress_pa=1e9, exp_a=2.0,
                      exp_n=1.5, exp_floor=0.1)
        cold = activated_front_fraction(process, 1e8, 500.0, **kwargs)
        hot = activated_front_fraction(process, 1e8, 1000.0, **kwargs)
        stressed = activated_front_fraction(process, 8e8, 500.0, **kwargs)
        entropy = activated_front_fraction(
            ActivatedProcess("front", 2.0e6, entropy_over_kB=2.0,
                             drag_rate_s=2.0e6), 1e8, 500.0, **kwargs)
        self.assertGreater(float(hot), float(cold))
        self.assertGreater(float(stressed), float(cold))
        self.assertGreater(float(entropy), float(cold))
        self.assertTrue(0.0 <= float(cold) <= 1.0)


if __name__ == "__main__":
    unittest.main()
