from dataclasses import replace

import numpy as np

from tests.test_v50_production_subcell_geometry import (
    intrinsic_kinetics, physical_rectangle,
)
from full_model.production.subcell_segment_geometry import (
    propose_subcell_face_extension, subcell_line_surface_nye,
)
from full_model.production.v24_mechanical_wall import (
    accepted_subcell_face_transaction, accepted_subcell_x_faces_shared_clock,
    accepted_v24_mechanical_step, mechanical_checkpoint_arrays,
    mechanical_from_checkpoint_arrays,
)


def test_boundary_surface_map_resolves_initial_and_event_increment_mismatch():
    initial_rows = []
    event_rows = []
    for n in (32, 64, 128):
        baseline, baseline_data = physical_rectangle(n)
        state, data = physical_rectangle(
            n, offset=(0.0, -.17*baseline_data[9]))
        _, _, audit = subcell_line_surface_nye(
            state.subcell_geometry, len(data[4]))
        proposal = propose_subcell_face_extension(
            state.subcell_geometry, state.density,
            state.reservoir_alignment, state.common, data[4], 1e-8)
        initial_rows.append(audit["line_surface_nye_residual_relative"])
        event_rows.append(proposal[-1][
            "line_surface_event_residual_relative"])
        assert audit["scalar_line_nonnegative"]
        assert audit["surface_fraction_bounded"]
        assert audit["scientific_compatibility_passed"]
        assert proposal[-1]["line_surface_event_compatibility_passed"]
    assert all(value < .05 for value in initial_rows+event_rows)
    assert initial_rows[2] < initial_rows[1] < initial_rows[0]
    assert event_rows[2] < event_rows[1] < event_rows[0]


def test_one_species_climb_count_is_the_affinity_count_and_site_identity():
    state, data = physical_rectangle(32)
    result, ledger = accepted_subcell_face_transaction(
        state, {"proposed_displacement_m": -1e-8},
        data[4], data[5], data[1], data[6], data[7],
        intrinsic_kinetics(), 1e-6)
    assert ledger["accepted"] and result is not state
    assert ledger["event_measure_convention"] == (
        "one_exchanged_species_per_climb_event")
    np.testing.assert_allclose(
        ledger["physical_event_count"],
        abs(ledger["signed_material_exchange_count"]), rtol=2e-14)
    np.testing.assert_allclose(
        ledger["physical_event_count"],
        ledger["event_count_from_site_jump_identity"], rtol=2e-12)
    assert abs(ledger["event_count_site_jump_identity_residual"])/ledger[
        "physical_event_count"] < 2e-12
    assert ledger["legacy_mixed_event_count_comparator"] > 2.0*ledger[
        "physical_event_count"]
    assert ledger["event_rate_s"] > 2.0*ledger[
        "legacy_mixed_affinity_rate_s_comparator"]
    assert not ledger["legacy_mixed_count_used_by_affinity"]


def test_undefined_event_measure_fails_atomically_without_a_tiny_floor():
    state, data = physical_rectangle(32)
    undefined = replace(
        intrinsic_kinetics(), atomic_volume_m3_per_atom=0.0,
        exchange_stoichiometry_defects_per_atom=0.0,
        chemical_species="none")
    result, ledger = accepted_subcell_face_transaction(
        state, {"proposed_displacement_m": -1e-8},
        data[4], data[5], data[1], data[6], data[7], undefined, 1e-6)
    assert result is state
    assert not ledger["accepted"]
    assert ledger["classification"] == "UNDEFINED_PHYSICAL_EVENT_MEASURE"
    assert ledger["consumed_duration_s"] == 0.0
    assert np.all(ledger["irreversible_heat_increment_J_m3"] == 0.0)


def test_declared_no_exchange_rejects_climb_geometry_atomically():
    state, data = physical_rectangle(32)
    area = data[6].burgers_m**2
    kinetics = replace(
        intrinsic_kinetics(), material_exchange_model="glide_no_exchange",
        atomic_volume_m3_per_atom=0.0,
        exchange_stoichiometry_defects_per_atom=0.0,
        chemical_species="none", physical_event_area_m2=area)
    result, ledger = accepted_subcell_face_transaction(
        state, {"proposed_displacement_m": -1e-8},
        data[4], data[5], data[1], data[6], data[7], kinetics, 1e-6)
    assert result is state and not ledger["accepted"]
    assert ledger["classification"] == "MECHANISM_GEOMETRY_MISMATCH"
    assert ledger["burgers_dot_surface_normal_m"] != 0.0
    np.testing.assert_allclose(
        ledger["signed_plastic_volume_increment_m3"],
        -1.1454629341e-24, rtol=2e-10)
    assert ledger["consumed_duration_s"] == 0.0
    assert np.all(ledger["irreversible_heat_increment_J_m3"] == 0.0)


def test_shared_no_exchange_rejects_each_climb_face_even_if_global_cancels():
    state, data = physical_rectangle(32)
    area = data[6].burgers_m**2
    kinetics = replace(
        intrinsic_kinetics(), material_exchange_model="glide_no_exchange",
        atomic_volume_m3_per_atom=0.0,
        exchange_stoichiometry_defects_per_atom=0.0,
        chemical_species="none", physical_event_area_m2=area)
    # Equal translations make the global swept area and volume vanish, but
    # each moving face remains climb for this BCC xy geometry.
    events = [
        {"face": "lower_x", "proposed_displacement_m": 1e-8,
         "fixed_rate_s": 1e9},
        {"face": "upper_x", "proposed_displacement_m": 1e-8,
         "fixed_rate_s": 1e9},
    ]
    result, ledger = accepted_subcell_x_faces_shared_clock(
        state, events, data[4], data[5], data[1], data[6], data[7],
        kinetics, 1e-7)
    assert result is state and not ledger["accepted"]
    assert ledger["classification"] == "MECHANISM_GEOMETRY_MISMATCH"
    assert abs(ledger["global_signed_plastic_volume_increment_m3"]) < 1e-38
    assert all(value != 0.0 for value in ledger[
        "face_signed_plastic_volume_increments_m3"].values())


def test_two_faces_share_elapsed_time_and_are_permutation_invariant():
    state, data = physical_rectangle(32)
    events = [
        {"face": "lower_x", "proposed_displacement_m": 1e-7,
         "fixed_rate_s": 1e8},
        {"face": "upper_x", "proposed_displacement_m": -1e-7,
         "fixed_rate_s": 2e8},
    ]
    args = (data[4], data[5], data[1], data[6], data[7],
            intrinsic_kinetics(), 1e-7)
    first, ledger = accepted_subcell_x_faces_shared_clock(
        state, events, *args)
    permuted, permuted_ledger = accepted_subcell_x_faces_shared_clock(
        state, list(reversed(events)), *args)
    assert ledger["accepted"] and permuted_ledger["accepted"]
    assert ledger["common_elapsed_time_s"] == 1e-7
    assert ledger["sum_of_face_active_durations_s"] == 2e-7
    assert ledger["clock_combination_rule"] == (
        "maximum_concurrent_face_exposure_not_sum")
    assert ledger["total_nonnegative_physical_event_count"] == sum(
        ledger["face_physical_event_counts"].values())
    np.testing.assert_allclose(
        ledger["signed_material_exchange_count"],
        ledger["independent_trace_exchange_count"], rtol=2e-12)
    np.testing.assert_array_equal(
        first.subcell_geometry.lower_left_m,
        permuted.subcell_geometry.lower_left_m)
    np.testing.assert_array_equal(
        first.subcell_geometry.upper_right_m,
        permuted.subcell_geometry.upper_right_m)
    np.testing.assert_allclose(
        first.density.wall_ordered_plus_m2,
        permuted.density.wall_ordered_plus_m2, rtol=0.0, atol=0.0)
    assert ledger["complete_energy_change_J_m3_cells"] == permuted_ledger[
        "complete_energy_change_J_m3_cells"]


def test_shared_face_clock_converges_under_common_time_splitting():
    state, data = physical_rectangle(32)
    events = [
        {"face": "lower_x", "proposed_displacement_m": 1e-7,
         "fixed_rate_s": 1e8},
        {"face": "upper_x", "proposed_displacement_m": -1e-7,
         "fixed_rate_s": 2e8},
    ]
    args = (data[4], data[5], data[1], data[6], data[7], intrinsic_kinetics())
    whole, whole_ledger = accepted_subcell_x_faces_shared_clock(
        state, events, *args, 1e-7)
    split = state; split_energy = 0.0; split_time = 0.0
    for _ in range(2):
        split, ledger = accepted_subcell_x_faces_shared_clock(
            split, events, *args, .5e-7)
        assert ledger["accepted"]
        split_energy += ledger["complete_energy_change_J_m3_cells"]
        split_time += ledger["common_elapsed_time_s"]
    assert whole_ledger["accepted"]
    assert split_time == whole_ledger["common_elapsed_time_s"]
    np.testing.assert_allclose(
        split.subcell_geometry.lower_left_m,
        whole.subcell_geometry.lower_left_m, rtol=0.0, atol=2e-21)
    np.testing.assert_allclose(
        split.subcell_geometry.upper_right_m,
        whole.subcell_geometry.upper_right_m, rtol=0.0, atol=2e-21)
    np.testing.assert_allclose(
        split_energy, whole_ledger["complete_energy_change_J_m3_cells"],
        rtol=2e-12)


def test_large_uphill_probe_finds_real_admissible_initial_advance():
    state, data = physical_rectangle(32)
    capability = replace(
        intrinsic_kinetics(), chemical_potential_J_per_defect=6e-21)
    event = {
        "proposed_displacement_m": 5e-8,
        "search_partial_displacement": True,
        "partial_displacement_levels": 16,
    }
    advanced, ledger = accepted_subcell_face_transaction(
        state, event, data[4], data[5], data[1], data[6], data[7],
        capability, 1.0)
    assert ledger["accepted"] and advanced is not state
    assert 0.0 < ledger["physical_displacement_m"] < event[
        "proposed_displacement_m"]
    assert ledger["complete_energy_change_J_m3_cells"] < 0.0
    search = ledger["same_state_partial_displacement_search"]
    assert search["full_proposal_affinity_blocked"]
    assert search["rows"][0]["complete_energy_change_J_m3_cells"] > 0.0
    assert any(not row["accepted"] for row in search["rows"][:-1])
    assert search["rows"][-1]["accepted"]
    # The selected state is exactly a direct production transaction at the
    # discovered displacement, not a repriced large-proposal candidate.
    direct, direct_ledger = accepted_subcell_face_transaction(
        state, {"proposed_displacement_m": ledger["physical_displacement_m"]},
        data[4], data[5], data[1], data[6], data[7], capability, 1.0)
    assert direct_ledger["accepted"]
    np.testing.assert_allclose(
        direct.subcell_geometry.upper_right_m,
        advanced.subcell_geometry.upper_right_m, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        direct.common.beta_p, advanced.common.beta_p, rtol=0.0, atol=0.0)


def test_geometry_owner_survives_two_coupled_cycles_and_restart():
    state, data = physical_rectangle(32)
    step_args = (data[1], data[3], data[4], data[5], data[6], data[7], data[8])
    first, ledger1 = accepted_v24_mechanical_step(
        state, *step_args, 1e-9, topology_route_enabled=False,
        mura_transport_operator="compatible_dealiased",
        subcell_geometry_event={"proposed_displacement_m": -1e-8},
        subcell_geometry_kinetics=intrinsic_kinetics())
    assert ledger1["subcell_geometry_event_energy_kinematics"]["accepted"]
    restarted = mechanical_from_checkpoint_arrays(
        mechanical_checkpoint_arrays(first), data[4], data[5])
    second, ledger2 = accepted_v24_mechanical_step(
        restarted, *step_args, 1e-9, topology_route_enabled=False,
        mura_transport_operator="compatible_dealiased",
        subcell_geometry_event={"proposed_displacement_m": -1e-8},
        subcell_geometry_kinetics=intrinsic_kinetics())
    assert ledger2["subcell_geometry_event_energy_kinematics"]["accepted"]
    assert int(second.subcell_geometry.accepted_event_count) == 2
    assert ledger1["nye_suboperator_audit"][
        "accepted_step_hard_invariant_passed"]
    assert ledger2["nye_suboperator_audit"][
        "accepted_step_hard_invariant_passed"]
