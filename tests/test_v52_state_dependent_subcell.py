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
    from dataclasses import replace
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
