import numpy as np

from full_model.production.arrhenius_kinetics import ActivatedProcess
from full_model.production.lattice_line_geometry import (
    empty_lattice_geometry, geometry_checkpoint_arrays,
    geometry_from_checkpoint_arrays, geometry_reservoir_fields,
    link_node_balance, propose_plaquette_sweep,
)
from full_model.production.v24_mechanical_wall import (
    V24MechanicalWallState, V43GeometryKinetics,
    accepted_geometry_plaquette_transaction, accepted_v24_mechanical_step,
    mechanical_checkpoint_arrays, mechanical_from_checkpoint_arrays,
    synchronize_common,
)
from tests.test_full_model_v24_mechanical_wall import mechanical_fixture


def prepared_loop(n=12, cell=(5, 5)):
    args = mechanical_fixture(n)
    old, systems, topologies = args[0], args[3], args[4]
    geometry = empty_lattice_geometry(
        old.common.orientation_rad.shape, len(systems), args[5].spacing_m)
    candidate = propose_plaquette_sweep(
        geometry, old.density, old.reservoir_alignment, old.common, systems,
        old.common.orientation_rad, cell, 0, 1, 1.0)
    state = synchronize_common(V24MechanicalWallState(
        candidate[3], candidate[1], candidate[2], candidate[0]), topologies)
    state.validate(systems, topologies)
    return state, args, candidate[-1]


def kinetics():
    return V43GeometryKinetics(
        ActivatedProcess("represented-plaquette-sweep", 1e12),
        enthalpy_J=0.0, critical_stress_Pa=1e9)


def verification_event(cell=(6, 5), extent=1.0, work=1e12):
    return {
        "cell": cell, "family": 0, "burgers_sign": 1,
        "proposed_extent": extent,
        "external_work_J_m3_cells": work,
        "verification_external_work": True,
    }


def all_state_arrays(state):
    return mechanical_checkpoint_arrays(state)


def test_nonzero_event_owns_closed_links_swept_beta_nye_and_energy():
    state, args, preparation = prepared_loop()
    result, ledger = accepted_geometry_plaquette_transaction(
        state, verification_event(), args[3], args[4], args[1], args[5],
        args[6], kinetics(), 1e-9)
    assert ledger["accepted"]
    assert ledger["accepted_extent"] > 0.0
    assert ledger["swept_area_m2"] > 0.0
    assert ledger["line_length_change_m"] != 0.0
    assert ledger["nye_curl_increment_rms_residual_m1"] < 1e-14
    assert ledger["maximum_node_balance_residual"] < 1e-14
    assert abs(ledger["heat_plus_complete_energy_residual_J_m3_cells"]) < 1e-3
    assert int(result.geometry.accepted_event_count) == 2
    assert preparation["line_length_after_m"] > 0.0
    result.validate(args[3], args[4])


def test_rejected_event_is_exact_atomic_rollback_and_zero_heat():
    state, args, _ = prepared_loop()
    # A labeled work-extraction challenge makes the otherwise feasible sweep
    # uphill and exercises atomic rollback without changing its geometry ray.
    rejected = verification_event(work=-1e12)
    result, ledger = accepted_geometry_plaquette_transaction(
        state, rejected, args[3], args[4], args[1], args[5], args[6],
        kinetics(), 1e-9)
    assert not ledger["accepted"]
    assert result is state
    assert np.array_equal(ledger["irreversible_heat_increment_J_m3"],
                          np.zeros_like(state.common.orientation_rad))
    for name, value in all_state_arrays(state).items():
        np.testing.assert_array_equal(all_state_arrays(result)[name], value)


def test_second_event_consumes_first_state_and_restart_is_exact():
    state, args, _ = prepared_loop()
    first, first_ledger = accepted_geometry_plaquette_transaction(
        state, verification_event(), args[3], args[4], args[1], args[5],
        args[6], kinetics(), 1e-9)
    restored = mechanical_from_checkpoint_arrays(
        mechanical_checkpoint_arrays(first), args[3], args[4])
    second_event = verification_event(cell=(6, 6))
    continuous, ledger_a = accepted_geometry_plaquette_transaction(
        first, second_event, args[3], args[4], args[1], args[5], args[6],
        kinetics(), 1e-9)
    restarted, ledger_b = accepted_geometry_plaquette_transaction(
        restored, second_event, args[3], args[4], args[1], args[5], args[6],
        kinetics(), 1e-9)
    assert first_ledger["accepted"] and ledger_a["accepted"] and ledger_b["accepted"]
    assert int(continuous.geometry.accepted_event_count) == 3
    for name, value in all_state_arrays(continuous).items():
        np.testing.assert_array_equal(all_state_arrays(restarted)[name], value)


def test_translation_symmetry_and_geometry_checkpoint_contract():
    first, args, _ = prepared_loop(cell=(4, 4))
    second, _, _ = prepared_loop(cell=(6, 7))
    shift = (2, 3)
    for name in ("swept_quanta", "edge_x_quanta", "edge_y_quanta",
                 "swept_burgers_area_m3", "edge_x_burgers_m",
                 "edge_y_burgers_m"):
        np.testing.assert_allclose(
            np.roll(getattr(first.geometry, name), shift, axis=(0, 1)),
            getattr(second.geometry, name), atol=1e-30, rtol=2e-14)
    payload = geometry_checkpoint_arrays(first.geometry)
    restored = geometry_from_checkpoint_arrays(payload, len(args[3]))
    restored.validate(len(args[3]))
    np.testing.assert_array_equal(restored.swept_quanta,
                                  first.geometry.swept_quanta)


def test_full_production_step_exercises_enabled_and_disabled_paths():
    state, args, _ = prepared_loop()
    disabled, off = accepted_v24_mechanical_step(
        state, *args[1:], dt_s=1e-9, topology_route_enabled=False)
    enabled, on = accepted_v24_mechanical_step(
        state, *args[1:], dt_s=1e-9, topology_route_enabled=False,
        geometry_event=verification_event(), geometry_kinetics=kinetics())
    assert off["geometry_event_energy_kinematics"] is None
    assert on["geometry_event_energy_kinematics"]["accepted"]
    assert int(disabled.geometry.accepted_event_count) == 1
    assert int(enabled.geometry.accepted_event_count) == 2
    assert on["nye_suboperator_audit"]["accepted_step_hard_invariant_passed"]
    assert not np.array_equal(disabled.common.beta_p, enabled.common.beta_p)


def test_geometry_density_is_nonnegative_realizable_and_endpoint_free():
    state, args, _ = prepared_loop()
    rho, moment = geometry_reservoir_fields(state.geometry)
    assert np.min(rho) >= 0.0
    assert np.max(np.linalg.norm(moment, axis=-1)-rho) < 1e-6
    node = link_node_balance(state.geometry.edge_x_quanta,
                             state.geometry.edge_y_quanta)
    assert np.max(np.abs(node)) == 0.0
