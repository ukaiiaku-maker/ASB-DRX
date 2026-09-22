import numpy as np

from full_model.analysis.run_v24_mechanical_supply import build_case
from full_model.analysis.run_v46_geometry_representation import (
    LENGTH_M, REPRESENTATION_LENGTH_M, THICKNESS_M, energy_terms_J,
)
from full_model.production.arrhenius_kinetics import ActivatedProcess
from full_model.production.subcell_segment_geometry import (
    initialize_subcell_rectangle, propose_subcell_face_extension,
    subcell_line_fields, subcell_face_field_derivative,
)
from full_model.production.v24_mechanical_wall import (
    V24MechanicalWallState, V43GeometryKinetics,
    accepted_subcell_face_transaction, mechanical_checkpoint_arrays,
    mechanical_from_checkpoint_arrays, synchronize_common,
    accepted_v24_mechanical_step,
)


def physical_rectangle(n=32, offset=(0.0, 0.0)):
    data = build_case(n, length_m=LENGTH_M, periodic_nye_consistent=True)
    base = data[0]; dx = data[9]
    lower = np.asarray((1.2e-6, 1.2e-6))+np.asarray(offset)
    initialized = initialize_subcell_rectangle(
        base.density, base.reservoir_alignment, base.common, data[4],
        base.common.orientation_rad, lower_left_m=lower,
        upper_right_m=lower+np.asarray((.8e-6, .8e-6)), family=0,
        burgers_sign=1, spacing_m=dx, section_thickness_m=THICKNESS_M,
        representation_length_m=REPRESENTATION_LENGTH_M)
    state = synchronize_common(V24MechanicalWallState(
        initialized[3], initialized[1], initialized[2], None,
        initialized[0]), data[5])
    state.validate(data[4], data[5])
    return state, data


def intrinsic_kinetics():
    return V43GeometryKinetics(
        ActivatedProcess("v50-subcell-climb", 1e9), enthalpy_J=0.0,
        critical_stress_Pa=1e9, chemical_species="vacancy",
        atomic_volume_m3_per_atom=1.8e-29,
        exchange_stoichiometry_defects_per_atom=1.0,
        continuum_representation_length_m=REPRESENTATION_LENGTH_M)


def test_physical_rectangle_gradient_increment_converges_to_continuous_reference():
    reference = 3.51163714e-16
    rows = []
    for n in (32, 64, 128):
        state, data = physical_rectangle(n)
        before = energy_terms_J(state, data)
        proposal = propose_subcell_face_extension(
            state.subcell_geometry, state.density, state.reservoir_alignment,
            state.common, data[4], 1e-8)
        candidate = synchronize_common(V24MechanicalWallState(
            proposal[3], proposal[1], proposal[2], None, proposal[0]), data[5])
        after = energy_terms_J(candidate, data)
        rows.append(after["ordered_gradient"]-before["ordered_gradient"])
    assert all(value > 0.0 for value in rows)
    assert abs(rows[-1]-reference)/reference < .01
    assert abs(rows[-1]-reference) < abs(rows[-2]-reference)


def test_subcell_map_is_positive_closed_and_has_bounded_moment():
    state, data = physical_rectangle(32, offset=(.23e-7, -.17e-7))
    rho, moment = subcell_line_fields(state.subcell_geometry, len(data[4]))
    assert np.min(rho) >= 0.0
    assert np.max(np.linalg.norm(moment, axis=-1)-rho) < 1e-6
    geometry = state.subcell_geometry
    assert int(geometry.accepted_event_count) == 0
    expected_length = 3.2e-6
    represented_length = float(np.sum(rho)*data[9]**2*THICKNESS_M)
    assert abs(represented_length-expected_length)/expected_length < .03


def test_physical_face_shape_derivative_closes_line_and_oriented_content():
    state, data = physical_rectangle(64)
    derivative = subcell_face_field_derivative(
        state.subcell_geometry, len(data[4]))
    volume = data[9]**2*THICKNESS_M
    line_length_derivative = float(np.sum(
        derivative["scalar_density_derivative_m3"])*volume)
    np.testing.assert_allclose(line_length_derivative, 2.0, rtol=.01)
    oriented_integral = np.sum(
        derivative["alignment_derivative_m3"], axis=(0, 1))*volume
    np.testing.assert_allclose(oriented_integral, 0.0, atol=1e-12)


def test_production_event_rejects_uphill_and_accepts_two_evolved_downhill_moves():
    state, data = physical_rectangle(32)
    args = (data[4], data[5], data[1], data[6], data[7], intrinsic_kinetics())
    rejected, uphill = accepted_subcell_face_transaction(
        state, {"proposed_displacement_m": 1e-8}, *args, 1e-6)
    assert rejected is state
    assert not uphill["accepted"]
    assert uphill["classification"] == "AFFINITY_BLOCKED_ZERO_EVENT"
    first, ledger1 = accepted_subcell_face_transaction(
        state, {"proposed_displacement_m": -1e-8}, *args, 1e-6)
    assert ledger1["accepted"] and first is not state
    assert ledger1["complete_energy_change_J_m3_cells"] < 0.0
    assert abs(ledger1["exchange_count_identity_residual"])/abs(
        ledger1["signed_material_exchange_count"]) < 1e-12
    np.testing.assert_allclose(
        abs(ledger1["signed_material_exchange_count"]),
        ledger1["active_site_measure_geometry"]
        *abs(ledger1["physical_displacement_m"])
        /ledger1["physical_event_jump_m"], rtol=2e-12)
    assert abs(ledger1["heat_plus_complete_energy_residual_J_m3_cells"]) < 1e-6
    second, ledger2 = accepted_subcell_face_transaction(
        first, {"proposed_displacement_m": -1e-8}, *args, 1e-6)
    assert ledger2["accepted"]
    assert int(second.subcell_geometry.accepted_event_count) == 2


def test_subcell_restart_and_reverse_proposal_are_exact():
    state, data = physical_rectangle(32)
    proposal = propose_subcell_face_extension(
        state.subcell_geometry, state.density, state.reservoir_alignment,
        state.common, data[4], -1e-8)
    moved = synchronize_common(V24MechanicalWallState(
        proposal[3], proposal[1], proposal[2], None, proposal[0]), data[5])
    restored = mechanical_from_checkpoint_arrays(
        mechanical_checkpoint_arrays(moved), data[4], data[5])
    for name, value in mechanical_checkpoint_arrays(moved).items():
        np.testing.assert_array_equal(
            value, mechanical_checkpoint_arrays(restored)[name])
    reverse = propose_subcell_face_extension(
        moved.subcell_geometry, moved.density, moved.reservoir_alignment,
        moved.common, data[4], 1e-8)
    roundtrip = synchronize_common(V24MechanicalWallState(
        reverse[3], reverse[1], reverse[2], None, reverse[0]), data[5])
    np.testing.assert_allclose(
        roundtrip.density.wall_ordered_plus_m2,
        state.density.wall_ordered_plus_m2, rtol=2e-14, atol=1e-5)
    np.testing.assert_allclose(
        roundtrip.common.beta_p, state.common.beta_p, rtol=2e-14, atol=1e-16)


def test_full_production_step_publishes_subcell_transaction():
    state, data = physical_rectangle(16)
    result, ledger = accepted_v24_mechanical_step(
        state, data[1], data[3], data[4], data[5], data[6], data[7], data[8],
        1e-9, topology_route_enabled=False,
        mura_transport_operator="compatible_dealiased",
        subcell_geometry_event={"proposed_displacement_m": -1e-8},
        subcell_geometry_kinetics=intrinsic_kinetics())
    event = ledger["subcell_geometry_event_energy_kinematics"]
    assert event["accepted"]
    assert event["operator"] == "physical_subcell_rectangle_face_extension"
    assert int(result.subcell_geometry.accepted_event_count) == 1
    assert ledger["nye_suboperator_audit"][
        "accepted_step_hard_invariant_passed"]
