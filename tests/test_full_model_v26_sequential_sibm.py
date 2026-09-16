from full_model.production.symmetric_sibm import (
    exchange_reflect_pair, normalized_production_phase_step,
    production_stage_overrides, sequential_activations,
)
from full_model.production.moving_front import (
    DefectState, complete_front_state_is_exactly_equal,
    advance_front, canonicalize_normal_sweep, initialize_existing_boundary_front,
    front_is_unprocessed_and_reservoir_free, recover_boundary_reservoir,
    supported_front_state_is_exactly_equal,
    translation_sweep_from_profile_change,
)
import numpy as np


def test_normalized_phase_map_is_exchange_reflection_equivariant():
    rng = np.random.default_rng(2801)
    child = rng.uniform(0.05, 0.95, 64)
    eta = np.stack((1.0-child, child), axis=-1)
    energy = rng.uniform(0.0, 2e6, eta.shape)
    kwargs = dict(spacing_m=1e-7, kappa_J_m=5e-7,
                  barrier_J_m3=5e6, step_scale=1e-10)
    direct, _ = normalized_production_phase_step(eta, energy, **kwargs)
    transformed, _ = normalized_production_phase_step(
        exchange_reflect_pair(eta), exchange_reflect_pair(energy), **kwargs)
    np.testing.assert_allclose(
        transformed, exchange_reflect_pair(direct), rtol=0.0, atol=2e-16)


def test_v26_activation_ladder_has_six_strictly_cumulative_stages():
    stages = sequential_activations()
    assert [stage.name for stage in stages] == [f"S{i}" for i in range(6)]
    previous = set()
    for stage in stages:
        current = {name for name in (
            "conservative_transfer", "boundary_storage",
            "neutral_cleanup_and_heat", "boundary_recovery_and_release",
            "constitutive_and_rehardening") if getattr(stage, name)}
        assert previous <= current
        previous = current


def test_stage_switches_separate_transfer_storage_cleanup_and_constitutive():
    s1 = production_stage_overrides("S1")
    s2 = production_stage_overrides("S2")
    s3 = production_stage_overrides("S3")
    s5 = production_stage_overrides("S5")
    assert s1["moving_front_fixture_transmission_fraction"] == 1.0
    assert s1["moving_front_boundary_storage_fraction"] == 0.0
    assert s2["moving_front_boundary_storage_fraction"] == 1.0
    assert s2["moving_front_sink_fraction"] == 0.0
    assert s3["sibm_stage_neutral_cleanup_heat"]
    assert not s3["sibm_stage_constitutive_rehardening"]
    assert s5["sibm_stage_constitutive_rehardening"]


def test_contour_canonicalization_removes_only_machine_roundoff():
    eps = np.finfo(float).eps
    result = canonicalize_normal_sweep(np.array([
        -1000*eps, 1000*eps, -1e-9, 1e-9]))
    np.testing.assert_array_equal(result[:2], 0.0)
    np.testing.assert_array_equal(result[2:], [-1e-9, 1e-9])


def test_translation_sweep_removes_width_change_and_preserves_ray_integral():
    profile_change = np.array([
        [-.2, -.1, .0], [-.1, .0, .2], [.1, .2, .3], [.2, -.1, .1]])
    result = translation_sweep_from_profile_change(profile_change, 0)
    np.testing.assert_allclose(np.sum(result, axis=0),
                               np.sum(profile_change, axis=0), atol=1e-16)
    ray_net = np.sum(profile_change, axis=0)
    for column, net in enumerate(ray_net):
        if net > 0.0:
            assert np.all(result[:, column] >= 0.0)
        elif net < 0.0:
            assert np.all(result[:, column] <= 0.0)


def test_translation_sweep_rejects_pure_profile_broadening():
    change = np.array([[-.2, -.1], [-.1, -.2], [.1, .2], [.2, .1]])
    np.testing.assert_allclose(
        translation_sweep_from_profile_change(change, 0), 0.0, atol=1e-16)


def test_exact_equal_front_state_loses_symmetry_when_boundary_content_appears():
    rp = np.ones((2, 2, 1)); rm = np.ones_like(rp)
    defect = DefectState(rp, rm, np.zeros_like(rp), np.zeros((2, 2)))
    state = initialize_existing_boundary_front(
        defect, np.full((2, 2), .5), 2.0, 0, 1)
    assert complete_front_state_is_exactly_equal(state)
    state.boundary_line_density_m2[0, 0] = 1.0
    assert not complete_front_state_is_exactly_equal(state)


def test_supported_equality_ignores_only_zero_support_latent_wake():
    rp = np.ones((2, 2, 1)); rm = np.ones_like(rp)
    defect = DefectState(rp, rm, np.zeros_like(rp), np.zeros((2, 2)))
    state = initialize_existing_boundary_front(
        defect, np.full((2, 2), .5), 2.0, 0, 1)
    state.recovered_wake.rp[:] = 0.0
    assert not complete_front_state_is_exactly_equal(state)
    assert supported_front_state_is_exactly_equal(state)
    state.child.rp[0, 0, 0] += 1.0
    assert not supported_front_state_is_exactly_equal(state)


def test_unprocessed_invariant_is_permanently_retired_by_first_sweep():
    rp = np.ones((2, 2, 1)); rm = np.ones_like(rp)
    defect = DefectState(rp, rm, np.zeros_like(rp), np.zeros((2, 2)))
    state = initialize_existing_boundary_front(
        defect, np.full((2, 2), .5), 2.0, 0, 1)
    assert front_is_unprocessed_and_reservoir_free(state)
    state, _ = advance_front(
        state, np.full((2, 2), .6), cell_area_m2=1.0,
        represented_thickness_m=1.0, line_energy_J_m=3.0,
        newly_swept_fraction=np.full((2, 2), .1),
        transmission_fraction=.5, boundary_storage_fraction=1.0)
    assert not front_is_unprocessed_and_reservoir_free(state)


def test_boundary_recovery_releases_signed_line_and_heats_only_neutral_line():
    rp = np.full((2, 2, 1), 4.0); rm = np.full_like(rp, 2.0)
    defect = DefectState(rp, rm, np.zeros_like(rp), np.zeros((2, 2)))
    state = initialize_existing_boundary_front(
        defect, np.full((2, 2), .5), 6.0, 0, 1)
    state, _ = advance_front(
        state, np.full((2, 2), .6), cell_area_m2=1.0,
        represented_thickness_m=1.0, line_energy_J_m=3.0,
        newly_swept_fraction=np.full((2, 2), .1),
        transmission_fraction=.5, boundary_storage_fraction=1.0)
    stored = state.boundary_line_density_m2.copy()
    recovered, _ = recover_boundary_reservoir(
        state, .25, cell_area_m2=1.0, represented_thickness_m=1.0,
        line_energy_J_m=3.0)
    np.testing.assert_allclose(
        recovered.boundary_line_density_m2, .75*stored)
    assert recovered.ledger.boundary_line_recovered_m > 0.0
    assert recovered.ledger.boundary_signed_released_m > 0.0
    assert recovered.ledger.heat_released_J == (
        3.0*recovered.ledger.boundary_neutral_recovered_m)


def test_boundary_recovery_releases_signed_line_into_recovered_wake_after_retreat():
    rp = np.full((2, 2, 1), 4.0); rm = np.full_like(rp, 2.0)
    defect = DefectState(rp, rm, np.zeros_like(rp), np.zeros((2, 2)))
    state = initialize_existing_boundary_front(
        defect, np.zeros((2, 2)), 6.0, 0, 1)
    state, _ = advance_front(
        state, np.full((2, 2), .25), cell_area_m2=1.0,
        represented_thickness_m=1.0, line_energy_J_m=3.0,
        newly_swept_fraction=np.full((2, 2), .25),
        transmission_fraction=.5, boundary_storage_fraction=1.0)
    state, mixture_before = advance_front(
        state, np.zeros((2, 2)), cell_area_m2=1.0,
        represented_thickness_m=1.0, line_energy_J_m=3.0,
        newly_swept_fraction=np.full((2, 2), -.25),
        transmission_fraction=.5, boundary_storage_fraction=1.0)
    assert not np.any(state.chi)
    assert np.all(state.recovered_wake_fraction > 0.0)
    released_signed = .25*state.boundary_signed_density_m2.copy()
    recovered, mixture_after = recover_boundary_reservoir(
        state, .25, cell_area_m2=1.0, represented_thickness_m=1.0,
        line_energy_J_m=3.0)
    np.testing.assert_allclose(recovered.child.rp, state.child.rp)
    np.testing.assert_allclose(recovered.child.rm, state.child.rm)
    np.testing.assert_allclose(
        (mixture_after.rp-mixture_after.rm)
        -(mixture_before.rp-mixture_before.rm), released_signed,
        rtol=2e-15, atol=1e-15)


def test_front_advance_preserves_cumulative_boundary_recovery_ledger():
    rp = np.full((2, 2, 1), 4.0); rm = np.full_like(rp, 2.0)
    defect = DefectState(rp, rm, np.zeros_like(rp), np.zeros((2, 2)))
    state = initialize_existing_boundary_front(
        defect, np.zeros((2, 2)), 6.0, 0, 1)
    state, _ = advance_front(
        state, np.full((2, 2), .25), cell_area_m2=1.0,
        represented_thickness_m=1.0, line_energy_J_m=3.0,
        newly_swept_fraction=np.full((2, 2), .25),
        transmission_fraction=.5, boundary_storage_fraction=1.0)
    state, _ = recover_boundary_reservoir(
        state, .25, cell_area_m2=1.0, represented_thickness_m=1.0,
        line_energy_J_m=3.0)
    recovered = state.ledger.boundary_line_recovered_m
    signed = state.ledger.boundary_signed_released_m
    neutral = state.ledger.boundary_neutral_recovered_m
    state, _ = advance_front(
        state, np.full((2, 2), .20), cell_area_m2=1.0,
        represented_thickness_m=1.0, line_energy_J_m=3.0,
        newly_swept_fraction=np.full((2, 2), -.05),
        transmission_fraction=.5, boundary_storage_fraction=1.0)
    assert state.ledger.boundary_line_recovered_m == recovered
    assert state.ledger.boundary_signed_released_m == signed
    assert state.ledger.boundary_neutral_recovered_m == neutral
