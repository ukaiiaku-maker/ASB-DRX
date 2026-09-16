from dataclasses import replace

import numpy as np

from full_model.production.common_tensorial_wall import (
    CommonWallDriving,
    CommonWallParameters,
    CommonWallState,
    resolved_driving_components,
    wall_residual,
)
from full_model.production.tensorial_nye import (
    bcc_four_family_systems,
    make_junction_topology,
)


def fixture(temperature_K):
    systems = bcc_four_family_systems()
    topologies = (make_junction_topology(
        systems, 0, 1, 1, -1, line_tension_J_m=0.0),)
    shape = (4, 4, 4)
    scalar = (4, 4)
    state = CommonWallState(
        np.full(shape, 8e13), np.full(shape, 7e13),
        np.full(shape, 5e13), np.full(shape, 4e13),
        np.full(shape, 2e13), np.full(shape, 1.5e13),
        np.full(scalar+(1,), 1e12), np.full(scalar, 0.2),
        np.zeros(scalar), np.zeros(shape), np.zeros(scalar+(3, 3)),
        np.zeros(shape+(3,)), np.zeros(shape+(3, 3)),
        np.zeros(scalar), np.full(scalar, temperature_K))
    parameters = CommonWallParameters(spacing_m=2e-7)
    return systems, topologies, state, parameters


def test_flow_override_reaches_authoritative_common_glide_operator():
    systems, topologies, cold, parameters = fixture(900.0)
    hot = replace(cold, temperature_K=np.full((4, 4), 1300.0))
    stress = np.full((4, 4, 4), 2e9)
    driving = CommonWallDriving(resolved_stress_Pa=stress)
    cold_speed = resolved_driving_components(
        cold, driving, systems, topologies, parameters)["speed_m_s"]
    hot_speed = resolved_driving_components(
        hot, driving, systems, topologies, parameters)["speed_m_s"]
    assert not np.array_equal(cold_speed, hot_speed)
    frozen = replace(parameters, flow_temperature_override_K=900.0)
    frozen_speed = resolved_driving_components(
        hot, driving, systems, topologies, frozen)["speed_m_s"]
    np.testing.assert_array_equal(frozen_speed, cold_speed)


def test_recovery_override_freezes_reactions_but_not_physical_temperature():
    systems, topologies, cold, parameters = fixture(900.0)
    hot = replace(cold, temperature_K=np.full((4, 4), 1300.0))
    zeros = np.zeros((4, 4, 4))
    driving = CommonWallDriving(
        resolved_stress_Pa=np.full((4, 4, 4), 5e8),
        glide_speed_m_s=zeros)
    cold_residual = wall_residual(cold, driving, systems, topologies, parameters)
    hot_residual = wall_residual(hot, driving, systems, topologies, parameters)
    assert not np.array_equal(
        cold_residual.channel_rates_m2_s["annihilation_pairs"],
        hot_residual.channel_rates_m2_s["annihilation_pairs"])
    frozen = replace(parameters, recovery_temperature_override_K=900.0)
    frozen_residual = wall_residual(hot, driving, systems, topologies, frozen)
    for channel in (
            "lock_plus", "lock_minus", "wall_plus", "wall_minus",
            "annihilation_pairs", "junction", "order"):
        np.testing.assert_array_equal(
            frozen_residual.channel_rates_m2_s[channel],
            cold_residual.channel_rates_m2_s[channel])
    # The causal control changes constitutive inputs, never the heat state.
    np.testing.assert_array_equal(hot.temperature_K, np.full((4, 4), 1300.0))


def test_temperature_override_validation():
    for value in (0.0, -1.0, np.nan):
        try:
            CommonWallParameters(
                spacing_m=2e-7, flow_temperature_override_K=value)
        except ValueError as error:
            assert "temperature overrides" in str(error)
        else:
            raise AssertionError("invalid temperature override was accepted")
