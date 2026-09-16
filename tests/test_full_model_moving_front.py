import unittest
from dataclasses import replace

import numpy as np

from full_model.production.moving_front import (
    DefectState, activated_front_fraction, advance_front,
    apply_common_constitutive_increment, initialize_sparse_front,
    initialize_declared_boundary_front, initialize_existing_boundary_front,
    conservative_front_transfer, front_feasibility_fields,
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
            boundary_storage_fraction=0.1, sink_fraction=0.05,
            transmission_fraction=0.5)

    def test_stationary_interface_has_zero_processing(self):
        state = self.state()
        updated, mixture = self.advance(state, state.chi)
        self.assertEqual(updated.ledger.parent_line_processed_m, 0.0)
        self.assertTrue(np.array_equal(mixture.rp, state.parent.rp))

    def test_declared_boundary_preserves_phase_contrast_without_processing(self):
        parent = self.parent()
        child = DefectState(*(0.25*np.asarray(value) for value in (
            parent.rp, parent.rm, parent.forest, parent.wall)))
        chi = np.linspace(0.0, 1.0, 16).reshape(4, 4)
        state = initialize_declared_boundary_front(parent, child, chi, 0, 1)
        mixture = reconstruct_mixture(state)
        for actual, high, low in zip(
                (mixture.rp, mixture.rm, mixture.forest, mixture.wall),
                (parent.rp, parent.rm, parent.forest, parent.wall),
                (child.rp, child.rm, child.forest, child.wall)):
            weight = chi[:, :, None] if actual.ndim == 3 else chi
            np.testing.assert_allclose(actual, (1.0-weight)*high+weight*low)
        self.assertEqual(state.ledger.parent_line_processed_m, 0.0)
        self.assertEqual(np.max(state.boundary_line_density_m2), 0.0)

    def test_advance_closes_line_signed_burgers_and_energy(self):
        state = self.state(); chi = np.zeros((4, 4)); chi[:, :2] = 1.0
        updated, mixture = self.advance(state, chi)
        ledger = updated.ledger
        self.assertLessEqual(abs(ledger.line_closure_m), 1e-20)
        self.assertEqual(ledger.signed_burgers_change_m2, 0.0)
        self.assertEqual(ledger.line_energy_released_J, ledger.heat_released_J)
        np.testing.assert_allclose(
            signed_density(mixture)
            +np.sum(updated.boundary_signed_density_m2, axis=2),
            signed_density(state.parent), rtol=0.0, atol=0.25)
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

    def test_diffuse_profile_change_without_contour_sweep_is_not_processed(self):
        state = self.state()
        broadened = np.full((4, 4), 0.25)
        state, _ = advance_front(
            state, broadened, cell_area_m2=1e-14,
            represented_thickness_m=1e-6, line_energy_J_m=2e-9,
            newly_swept_fraction=np.zeros((4, 4)))
        self.assertEqual(state.ledger.swept_volume_m3, 0.0)
        np.testing.assert_array_equal(state.cleanup_max, 0.0)
        swept = np.full((4, 4), 0.25)
        state, _ = advance_front(
            state, broadened, cell_area_m2=1e-14,
            represented_thickness_m=1e-6, line_energy_J_m=2e-9,
            newly_swept_fraction=swept)
        self.assertGreater(state.ledger.swept_volume_m3, 0.0)
        np.testing.assert_array_equal(state.cleanup_max, swept)

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

    def test_v6_fraction_semantics_and_v5_migration_are_explicit(self):
        state = self.state(); state, _ = self.advance(
            state, np.full_like(state.chi, 0.75))
        state, _ = self.advance(state, np.full_like(state.chi, 0.25))
        parent, child, wake = state.material_support_weights()
        np.testing.assert_array_equal(child, state.current_child_fraction)
        np.testing.assert_array_equal(
            state.maximum_swept_fraction, state.processed_max)
        np.testing.assert_array_equal(wake, state.recovered_wake_fraction)
        np.testing.assert_allclose(parent+child+wake, 1.0, rtol=0.0, atol=0.0)
        self.assertGreaterEqual(min(np.min(x) for x in (parent, child, wake)), 0.0)

        metadata = __import__("json").loads(state_metadata_json(state))
        self.assertEqual(metadata["schema"], "full-v34-sparse-front/v6")
        arrays = state_arrays(state)
        np.testing.assert_array_equal(
            arrays["current_child_fraction"], arrays["chi"])
        np.testing.assert_array_equal(
            arrays["maximum_swept_fraction"], arrays["processed_max"])

        # Historical v5 files contain only the two legacy array names.
        metadata["schema"] = "full-v34-sparse-front/v5"
        legacy = {k: v for k, v in arrays.items() if k not in (
            "current_child_fraction", "maximum_swept_fraction")}
        restored = state_from_checkpoint(__import__("json").dumps(metadata), legacy)
        np.testing.assert_array_equal(restored.chi, state.chi)
        np.testing.assert_array_equal(restored.processed_max, state.processed_max)

    def test_v6_rejects_conflicting_fraction_aliases(self):
        state = self.state()
        arrays = state_arrays(state)
        arrays["current_child_fraction"] = np.ones_like(state.chi)
        with self.assertRaisesRegex(ValueError, "conflicting current-child"):
            state_from_checkpoint(state_metadata_json(state), arrays)

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

    def test_existing_boundary_initialization_is_exact_and_future_sweep_processes(self):
        chi = np.zeros((4, 4)); chi[:, :2] = 1.0
        state = initialize_existing_boundary_front(
            self.parent(), chi, 5e14, 0, 1)
        mixture = reconstruct_mixture(state)
        np.testing.assert_allclose(mixture.rp, self.parent().rp, rtol=1e-15, atol=0.0)
        self.assertEqual(state.ledger.parent_line_processed_m, 0.0)
        grown = chi.copy(); grown[:, 2] = 1.0
        state, _ = self.advance(state, grown)
        self.assertGreater(state.ledger.parent_line_processed_m, 0.0)

    def test_existing_boundary_map_is_feasible_without_latent_target(self):
        chi = np.linspace(0.0, 1.0, 16).reshape(4, 4)
        state = initialize_existing_boundary_front(
            self.parent(), chi, 1e8, 0, 1)
        fields = front_feasibility_fields(state)
        np.testing.assert_array_equal(fields["feasibility_margin_m2"], 0.0)
        np.testing.assert_allclose(
            reconstruct_mixture(state).rp, self.parent().rp,
            rtol=2*np.finfo(float).eps, atol=0.0)

    def test_support_owned_increment_does_not_evolve_absent_child(self):
        state = self.state()
        old = reconstruct_mixture(state)
        updated = DefectState(
            old.rp+3.0, old.rm+2.0, old.forest+1.0, old.wall+4.0)
        result = apply_common_constitutive_increment(state, updated)
        np.testing.assert_array_equal(result.child.rp, 0.0)
        np.testing.assert_array_equal(result.recovered_wake.rp, 0.0)
        np.testing.assert_allclose(reconstruct_mixture(result).rp, updated.rp)

    def test_randomized_per_sign_transfer_closes_and_is_permutation_symmetric(self):
        rng = np.random.default_rng(417)
        parent = DefectState(
            rng.uniform(0, 8e15, (3, 5, 4)),
            rng.uniform(0, 8e15, (3, 5, 4)),
            rng.uniform(0, 2e15, (3, 5, 4)),
            rng.uniform(0, 1e15, (3, 5)))
        transfer = conservative_front_transfer(
            parent, transmission_fraction=np.array([0.1, 0.4, 0.7, 0.9]),
            boundary_storage_fraction=0.2, neutral_sink_fraction=0.1)
        self.assertGreaterEqual(min(np.min(x) for x in (
            transfer.child.rp, transfer.child.rm, transfer.child.forest,
            transfer.child.wall, transfer.boundary_excess_line_density_m2,
            transfer.annihilated_line_density_m2,
            transfer.sink_line_density_m2)), 0.0)
        self.assertLessEqual(np.max(np.abs(transfer.line_closure_density_m2)), 8.0)
        self.assertLess(np.max(np.abs(transfer.signed_closure_density_m2)), 1.0)
        permutation = np.array([2, 0, 3, 1])
        permuted = DefectState(
            parent.rp[..., permutation], parent.rm[..., permutation],
            parent.forest[..., permutation], parent.wall)
        other = conservative_front_transfer(
            permuted,
            transmission_fraction=np.array([0.1, 0.4, 0.7, 0.9])[permutation],
            boundary_storage_fraction=0.2, neutral_sink_fraction=0.1)
        np.testing.assert_allclose(
            other.boundary_excess_signed_density_m2,
            transfer.boundary_excess_signed_density_m2[..., permutation])
        np.testing.assert_allclose(
            other.boundary_excess_line_density_m2,
            transfer.boundary_excess_line_density_m2)

    def test_signed_dominated_neutral_rich_and_zero_mismatch_transfers(self):
        signed = DefectState(
            np.full((1, 1, 2), 9e14), np.full((1, 1, 2), 1e14),
            np.zeros((1, 1, 2)), np.zeros((1, 1)))
        result = conservative_front_transfer(signed, transmission_fraction=0.25)
        self.assertGreater(result.boundary_excess_line_density_m2.item(), 0.0)
        self.assertEqual(result.annihilated_line_density_m2.item(), 3e14)
        neutral = DefectState(
            np.full((1, 1, 2), 5e14), np.full((1, 1, 2), 5e14),
            np.full((1, 1, 2), 2e14), np.full((1, 1), 1e14))
        result = conservative_front_transfer(neutral, transmission_fraction=0.0)
        self.assertEqual(np.max(np.abs(result.boundary_excess_signed_density_m2)), 0.0)
        self.assertEqual(result.boundary_excess_line_density_m2.item(), 0.0)
        self.assertGreater(result.annihilated_line_density_m2.item(), 0.0)

    def test_explicit_signed_sink_is_separate_and_conservative(self):
        result = conservative_front_transfer(
            self.parent(), transmission_fraction=0.25,
            signed_sink_fraction=0.4)
        self.assertGreater(np.max(np.abs(result.sink_signed_density_m2)), 0.0)
        self.assertGreater(result.sink_line_density_m2.max(), 0.0)
        self.assertLess(np.max(np.abs(result.signed_closure_density_m2)), 1.0)

    def test_capacity_limit_stalls_without_partial_mutation(self):
        state = self.state()
        before = state_arrays(state)
        stalled, mixture = advance_front(
            state, np.ones_like(state.chi), cell_area_m2=1e-14,
            represented_thickness_m=1e-6, line_energy_J_m=2e-9,
            transmission_fraction=0.5, boundary_capacity_density_m2=0.0)
        self.assertEqual(stalled.ledger.swept_volume_m3, 0.0)
        self.assertGreater(stalled.ledger.capacity_limited_volume_m3, 0.0)
        for key, value in before.items():
            np.testing.assert_array_equal(state_arrays(stalled)[key], value)
        np.testing.assert_array_equal(mixture.rp, state.parent.rp)

    def test_normal_sweep_volume_and_profile_relaxation_are_separate(self):
        state = self.state()
        phase_profile = np.full((4, 4), 0.4)
        normal = np.zeros((4, 4)); normal[:, 0] = 0.25
        state, _ = advance_front(
            state, phase_profile, cell_area_m2=2e-14,
            represented_thickness_m=3e-6, line_energy_J_m=2e-9,
            newly_swept_fraction=normal, transmission_fraction=0.5)
        expected = float(np.sum(normal))*2e-14*3e-6
        self.assertEqual(state.ledger.swept_volume_m3, expected)
        np.testing.assert_array_equal(state.chi, normal)

    def test_intrinsic_hagb_does_not_create_excess_reservoir(self):
        chi = np.full((4, 4), 0.5)
        state = initialize_existing_boundary_front(
            self.parent(), chi, 5e14, 0, 1)
        self.assertEqual(np.max(state.boundary_line_density_m2), 0.0)
        self.assertEqual(np.max(np.abs(state.boundary_signed_density_m2)), 0.0)
        translated, _ = advance_front(
            state, chi, cell_area_m2=1e-14,
            represented_thickness_m=1e-6, line_energy_J_m=2e-9,
            newly_swept_fraction=np.zeros_like(chi), transmission_fraction=0.5)
        self.assertEqual(np.max(translated.boundary_line_density_m2), 0.0)


if __name__ == "__main__":
    unittest.main()
