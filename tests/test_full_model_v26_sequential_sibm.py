from full_model.production.symmetric_sibm import (
    production_stage_overrides, sequential_activations,
)
from full_model.production.moving_front import (
    DefectState, complete_front_state_is_exactly_equal,
    advance_front, canonicalize_normal_sweep, initialize_existing_boundary_front,
    recover_boundary_reservoir,
)
import numpy as np


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


def test_exact_equal_front_state_loses_symmetry_when_boundary_content_appears():
    rp = np.ones((2, 2, 1)); rm = np.ones_like(rp)
    defect = DefectState(rp, rm, np.zeros_like(rp), np.zeros((2, 2)))
    state = initialize_existing_boundary_front(
        defect, np.full((2, 2), .5), 2.0, 0, 1)
    assert complete_front_state_is_exactly_equal(state)
    state.boundary_line_density_m2[0, 0] = 1.0
    assert not complete_front_state_is_exactly_equal(state)


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
