from dataclasses import replace

import numpy as np

from tests.test_v50_production_subcell_geometry import (
    intrinsic_kinetics, physical_rectangle,
)
from full_model.production.subcell_segment_geometry import (
    propose_subcell_face_extension, subcell_line_surface_nye,
)
from full_model.production.v24_mechanical_wall import (
    accepted_subcell_face_transaction,
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


def test_declared_volume_preserving_area_event_has_its_own_site_measure():
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
    assert ledger["accepted"] and result is not state
    assert ledger["event_measure_convention"] == (
        "declared_volume_preserving_area_event")
    np.testing.assert_allclose(
        ledger["physical_event_count"],
        abs(ledger["signed_swept_area_m2"])/area, rtol=2e-14)
    np.testing.assert_allclose(
        ledger["physical_event_count"],
        ledger["event_count_from_site_jump_identity"], rtol=2e-14)
