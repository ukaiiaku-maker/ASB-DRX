from dataclasses import replace
from unittest.mock import patch

import numpy as np

from tests.test_v50_production_subcell_geometry import (
    intrinsic_kinetics, physical_rectangle,
)
from full_model.production.state_dependent_subcell import (
    accepted_state_dependent_subcell_x_faces, state_dependent_face_rates,
)
from full_model.production.v24_mechanical_wall import (
    mechanical_checkpoint_arrays, mechanical_from_checkpoint_arrays,
)
def fixture_args(state, data):
    return (data[4], data[5], data[1], data[6], data[7],
            intrinsic_kinetics())


def test_homogeneous_face_sampling_refines_and_recovers_uniform_site_rate():
    state, data = physical_rectangle(32)
    directions = {"lower_x": 1.0, "upper_x": -1.0}
    args = fixture_args(state, data)
    coarse = state_dependent_face_rates(
        state, directions, *args, quadrature_order=8)
    fine = state_dependent_face_rates(
        state, directions, *args, quadrature_order=32)
    for face in directions:
        np.testing.assert_allclose(
            coarse[face]["generalized_rate_s"],
            fine[face]["generalized_rate_s"], rtol=2e-12)
        assert fine[face]["site_rate_minimum_s"] == fine[face][
            "site_rate_maximum_s"]
        np.testing.assert_allclose(
            fine[face]["generalized_rate_s"],
            fine[face]["rate_at_weighted_mean_inputs_s"], rtol=2e-14)
        assert fine[face]["available_energy_per_event_J"] > 0.0


def test_heterogeneous_temperature_averages_local_rates_not_mean_inputs():
    state, data = physical_rectangle(32)
    x, y = np.indices(state.common.temperature_K.shape)
    temperature = 900.0+400.0*(y/(y.shape[1]-1))**2
    state = replace(state, common=replace(
        state.common, temperature_K=temperature))
    directions = {"lower_x": 1.0, "upper_x": -1.0}
    rates = state_dependent_face_rates(
        state, directions, *fixture_args(state, data), quadrature_order=32)
    for row in rates.values():
        assert row["temperature_maximum_K"] > row["temperature_minimum_K"]
        assert row["site_rate_maximum_s"] > row["site_rate_minimum_s"]
        assert not np.isclose(
            row["generalized_rate_s"],
            row["rate_at_weighted_mean_inputs_s"], rtol=1e-8)


def test_state_dependent_two_face_clock_refreshes_at_first_cap_and_restarts():
    state, data = physical_rectangle(32)
    directions = {"lower_x": 1.0, "upper_x": -1.0}
    args = fixture_args(state, data)
    # Long enough to reach at least one numerical displacement cap.
    duration = 1.0e-6
    whole, ledger = accepted_state_dependent_subcell_x_faces(
        state, directions, *args, duration, quadrature_order=8)
    assert ledger["accepted"] and whole is not state
    assert ledger["consumed_duration_s"] == duration
    assert ledger["rate_refresh_count"] >= 2
    assert ledger["clock_combination_rule"] == (
        "first_numerical_cap_then_refresh")
    split = state
    for _ in range(2):
        split, half = accepted_state_dependent_subcell_x_faces(
            split, directions, *args, duration/2, quadrature_order=8)
        assert half["accepted"]
    initial_lower = state.subcell_geometry.lower_left_m[0]
    initial_upper = state.subcell_geometry.upper_right_m[0]
    lower_scale = abs(whole.subcell_geometry.lower_left_m[0]-initial_lower)
    upper_scale = abs(whole.subcell_geometry.upper_right_m[0]-initial_upper)
    assert abs(split.subcell_geometry.lower_left_m[0]
               -whole.subcell_geometry.lower_left_m[0])/lower_scale < .005
    assert abs(split.subcell_geometry.upper_right_m[0]
               -whole.subcell_geometry.upper_right_m[0])/upper_scale < .005
    assert abs(np.mean(split.common.temperature_K)
               -np.mean(whole.common.temperature_K)) < 5e-5


def test_state_dependent_multiple_cycles_restart_exactly():
    state, data = physical_rectangle(32)
    directions = {"lower_x": 1.0, "upper_x": -1.0}
    args = fixture_args(state, data)
    first, first_ledger = accepted_state_dependent_subcell_x_faces(
        state, directions, *args, 2e-7, quadrature_order=8)
    restored = mechanical_from_checkpoint_arrays(
        mechanical_checkpoint_arrays(first), data[4], data[5])
    continuous, ledger_a = accepted_state_dependent_subcell_x_faces(
        first, directions, *args, 2e-7, quadrature_order=8)
    restarted, ledger_b = accepted_state_dependent_subcell_x_faces(
        restored, directions, *args, 2e-7, quadrature_order=8)
    assert first_ledger["accepted"] and ledger_a["accepted"] and ledger_b[
        "accepted"]
    for name, value in mechanical_checkpoint_arrays(continuous).items():
        np.testing.assert_array_equal(
            mechanical_checkpoint_arrays(restarted)[name], value)


def test_signed_affinity_matches_shared_ledger_for_both_burgers_signs():
    from full_model.production.state_dependent_subcell import (
        _face_complete_affinity,
    )
    from full_model.production.v24_mechanical_wall import (
        accepted_subcell_x_faces_shared_clock,
    )
    for burgers_sign in (1.0, -1.0):
        state, data = physical_rectangle(32, burgers_sign=int(burgers_sign))
        kinetics = replace(
            intrinsic_kinetics(), chemical_potential_J_per_defect=2.5e-21)
        direction = -1.0
        probe = 1e-10
        local = _face_complete_affinity(
            state, "upper_x", direction, probe, data[4], data[5], data[1],
            data[6], data[7], kinetics)
        events = [
            {"face": "lower_x", "proposed_displacement_m": 0.0,
             "fixed_rate_s": 0.0},
            {"face": "upper_x", "proposed_displacement_m": direction*probe,
             "fixed_rate_s": 1.0},
        ]
        _, ledger = accepted_subcell_x_faces_shared_clock(
            state, events, data[4], data[5], data[1], data[6], data[7],
            kinetics, probe/data[6].burgers_m)
        np.testing.assert_allclose(
            local["signed_material_exchange_count"],
            ledger["signed_material_exchange_count"], rtol=2e-14)
        np.testing.assert_allclose(
            local["chemical_reservoir_work_J"],
            ledger["chemical_reservoir_work_J"], rtol=2e-14)
        assert np.sign(local["chemical_reservoir_work_J"]) == -burgers_sign


def test_one_stalled_face_advances_other_on_same_clock():
    state, data = physical_rectangle(32)
    directions = {"lower_x": 1.0, "upper_x": -1.0}
    args = fixture_args(state, data)
    actual = state_dependent_face_rates(
        state, directions, *args, quadrature_order=8)
    actual["lower_x"]["generalized_rate_s"] = 0.0
    actual["lower_x"]["generalized_velocity_m_s"] = 0.0
    with patch(
            "full_model.production.state_dependent_subcell."
            "state_dependent_face_rates", return_value=actual):
        result, ledger = accepted_state_dependent_subcell_x_faces(
            state, directions, *args, 1e-8, quadrature_order=8)
    assert ledger["accepted"]
    assert ledger["consumed_duration_s"] == 1e-8
    assert result.subcell_geometry.lower_left_m[0] == (
        state.subcell_geometry.lower_left_m[0])
    assert result.subcell_geometry.upper_right_m[0] < (
        state.subcell_geometry.upper_right_m[0])
    assert ledger["substeps"][0]["face_displacements_m"]["lower_x"] == 0.0


def test_both_stalled_faces_are_identity_over_full_geometry_exposure():
    state, data = physical_rectangle(32)
    directions = {"lower_x": 1.0, "upper_x": -1.0}
    args = fixture_args(state, data)
    rates = state_dependent_face_rates(
        state, directions, *args, quadrature_order=8)
    for row in rates.values():
        row["generalized_rate_s"] = 0.0
        row["generalized_velocity_m_s"] = 0.0
    with patch(
            "full_model.production.state_dependent_subcell."
            "state_dependent_face_rates", return_value=rates):
        result, ledger = accepted_state_dependent_subcell_x_faces(
            state, directions, *args, 3e-8, quadrature_order=8)
    assert result is state
    assert ledger["accepted"] and ledger["macro_complete"]
    assert ledger["classification"] == (
        "ALL_FACES_STALLED_IDENTITY_OVER_REMAINDER")
    assert ledger["consumed_duration_s"] == 3e-8


def test_rejected_trial_after_valid_prefix_reports_exact_partial_duration():
    from full_model.production import state_dependent_subcell as module
    state, data = physical_rectangle(32)
    directions = {"lower_x": 1.0, "upper_x": -1.0}
    args = fixture_args(state, data)
    original = module.accepted_subcell_x_faces_shared_clock
    calls = {"count": 0}

    def accept_then_reject(*call_args, **call_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return original(*call_args, **call_kwargs)
        current = call_args[0]
        return current, {"accepted": False, "classification": "TEST_REJECTION"}

    with patch(
            "full_model.production.state_dependent_subcell."
            "accepted_subcell_x_faces_shared_clock",
            side_effect=accept_then_reject):
        result, ledger = accepted_state_dependent_subcell_x_faces(
            state, directions, *args, 1e-6, quadrature_order=8)
    assert result is not state
    assert ledger["accepted"] and ledger["accepted_prefix_valid"]
    assert not ledger["macro_complete"]
    assert 0.0 < ledger["consumed_duration_s"] < 1e-6
    assert ledger["classification"] == (
        "VALID_PARTIAL_PREFIX_JOINT_SUBSTEP_REJECTED")
    assert not ledger["state_mutation_beyond_consumed_duration"]
