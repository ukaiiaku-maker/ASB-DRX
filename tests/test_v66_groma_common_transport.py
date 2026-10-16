from dataclasses import replace

import numpy as np
import pytest

from full_model.production.common_tensorial_wall import (
    CommonWallDriving, balance_ledger, resolved_driving_components,
    wall_free_energy_density_J_m3, wall_free_energy_derivatives, wall_residual,
)
from tests.test_full_model_common_tensorial_wall import fixture


def candidate_parameters(parameters):
    poisson = parameters.c12_Pa/(parameters.c11_Pa+parameters.c12_Pa)
    reduced_modulus = parameters.c44_Pa/(2.0*np.pi*(1.0-poisson))
    gb2 = reduced_modulus*parameters.burgers_m**2
    # Literature-constrained mechanism ablation: A about 1.25 and D in the
    # reported 0.25--0.8 interval. No material calibration is implied.
    return replace(
        parameters, groma_family_correlation_energy_J_m=1.25*gb2,
        groma_gnd_correlation_energy_J_m=.5*gb2,
        groma_sign_resolved_transport_enabled=True,
        mobile_correlation_diffusivity_m2_s=0.0)


def test_sign_resolved_branch_rejects_unqualified_supported_upwind_transport():
    _, _, parameters, _, _ = fixture(n=8)
    with pytest.raises(ValueError, match="periodic spectral"):
        replace(candidate_parameters(parameters), transport_scheme="upwind")


def test_groma_upwind_transport_is_conservative_positive_and_sign_symmetric():
    systems, topologies, parameters, state, _ = fixture(n=16)
    parameters = replace(
        candidate_parameters(parameters), transport_scheme="groma_upwind")
    state = patterned_state(state)
    stress = np.full_like(state.mobile_plus_m2, 2.0e9)
    result = wall_residual(
        state, CommonWallDriving(resolved_stress_Pa=stress),
        systems, topologies, parameters)
    for key in ("transport_mobile_plus_rate_m2_s",
                "transport_mobile_minus_rate_m2_s"):
        rate = result.channel_rates_m2_s[key]
        imbalance = np.abs(np.sum(rate, axis=(0, 1)))
        turnover = np.sum(np.abs(rate), axis=(0, 1))
        assert np.max(imbalance/turnover) < 2e-15
    assert np.min(result.heat_rate_W_m3) >= 0.0
    assert balance_ledger(
        state, result, systems, topologies
    )["relative_energy_balance_residual"] < 1e-5

    exchanged = replace(
        state,
        mobile_plus_m2=state.mobile_minus_m2,
        mobile_minus_m2=state.mobile_plus_m2,
        forest_plus_m2=state.forest_minus_m2,
        forest_minus_m2=state.forest_plus_m2,
        wall_plus_m2=state.wall_minus_m2,
        wall_minus_m2=state.wall_plus_m2)
    reversed_result = wall_residual(
        exchanged, CommonWallDriving(resolved_stress_Pa=-stress),
        systems, topologies, parameters)
    np.testing.assert_allclose(
        reversed_result.channel_rates_m2_s[
            "transport_mobile_plus_rate_m2_s"],
        result.channel_rates_m2_s["transport_mobile_minus_rate_m2_s"],
        rtol=2e-14, atol=1e-3)
    np.testing.assert_allclose(
        reversed_result.channel_rates_m2_s[
            "transport_mobile_minus_rate_m2_s"],
        result.channel_rates_m2_s["transport_mobile_plus_rate_m2_s"],
        rtol=2e-14, atol=1e-3)


def patterned_state(state):
    n = state.orientation_rad.shape[0]
    x = 2.0*np.pi*np.arange(n)[:, None]/n
    y = 2.0*np.pi*np.arange(n)[None, :]/n
    modulation = (1.0+.05*np.sin(x)+.03*np.cos(y))[..., None]
    return replace(
        state, mobile_plus_m2=state.mobile_plus_m2*modulation,
        mobile_minus_m2=state.mobile_minus_m2*(2.0-modulation))


def test_groma_mobile_chemical_potentials_match_complete_energy_derivative():
    systems, topologies, parameters, state, _ = fixture(n=8)
    parameters = candidate_parameters(parameters)
    state = patterned_state(state)
    chemical = wall_free_energy_derivatives(
        state, parameters, topologies, systems)
    increment = np.zeros_like(state.mobile_plus_m2)
    increment[..., 0] = 1.0e7
    for name, key in (("mobile_plus_m2", "mobile_plus_mu_J_m"),
                      ("mobile_minus_m2", "mobile_minus_mu_J_m"),
                      ("forest_plus_m2", "forest_plus_mu_J_m"),
                      ("forest_minus_m2", "forest_minus_mu_J_m"),
                      ("wall_plus_m2", "wall_plus_mu_J_m"),
                      ("wall_minus_m2", "wall_minus_mu_J_m")):
        plus = replace(state, **{name: getattr(state, name)+increment})
        minus = replace(state, **{name: getattr(state, name)-increment})
        numerical = (
            wall_free_energy_density_J_m3(
                plus, parameters, topologies, systems)
            -wall_free_energy_density_J_m3(
                minus, parameters, topologies, systems))/(2.0e7)
        np.testing.assert_allclose(
            numerical, chemical[key][..., 0], rtol=3e-7, atol=5e-15)


def test_groma_energy_is_invariant_to_internal_reservoir_relabeling():
    systems, topologies, parameters, state, _ = fixture(n=8)
    enabled = candidate_parameters(parameters)
    amount = np.zeros_like(state.mobile_plus_m2)
    amount[..., 0] = 1.0e10
    transferred = replace(
        state, mobile_plus_m2=state.mobile_plus_m2-amount,
        forest_plus_m2=state.forest_plus_m2+amount)
    enabled_delta = wall_free_energy_density_J_m3(
        transferred, enabled, topologies, systems)-wall_free_energy_density_J_m3(
            state, enabled, topologies, systems)
    baseline_delta = wall_free_energy_density_J_m3(
        transferred, parameters, topologies, systems)-wall_free_energy_density_J_m3(
            state, parameters, topologies, systems)
    np.testing.assert_allclose(enabled_delta, baseline_delta, rtol=0.0, atol=2e-7)


def test_polarized_absorption_nucleates_wall_but_zero_gnd_control_does_not():
    systems, topologies, parameters, state, _ = fixture(n=8)
    parameters = replace(
        candidate_parameters(parameters),
        polarized_wall_absorption_enabled=True)
    equal_mobile = np.full_like(state.mobile_plus_m2, 8.0e13)
    equal_forest = np.full_like(state.forest_plus_m2, 5.0e13)
    empty_wall = np.zeros_like(state.wall_plus_m2)
    control = replace(
        state, mobile_plus_m2=equal_mobile, mobile_minus_m2=equal_mobile,
        forest_plus_m2=equal_forest, forest_minus_m2=equal_forest,
        wall_plus_m2=empty_wall, wall_minus_m2=empty_wall,
        wall_order=np.zeros_like(state.wall_order))
    driving = CommonWallDriving(
        resolved_stress_Pa=np.zeros_like(control.mobile_plus_m2))
    zero_gnd = wall_residual(
        control, driving, systems, topologies, parameters)
    np.testing.assert_array_equal(
        zero_gnd.channel_rates_m2_s["wall_absorption_activity_plus"], 0.0)
    np.testing.assert_array_equal(
        zero_gnd.channel_rates_m2_s["wall_absorption_activity_minus"], 0.0)
    np.testing.assert_array_equal(
        zero_gnd.channel_rates_m2_s["wall_plus"], 0.0)
    np.testing.assert_array_equal(
        zero_gnd.channel_rates_m2_s["wall_minus"], 0.0)

    polarized = replace(
        control, mobile_plus_m2=control.mobile_plus_m2.copy())
    polarized.mobile_plus_m2[..., 0] *= 1.25
    result = wall_residual(
        polarized, driving, systems, topologies, parameters)
    activity_plus = result.channel_rates_m2_s["wall_absorption_activity_plus"]
    activity_minus = result.channel_rates_m2_s["wall_absorption_activity_minus"]
    assert np.all(activity_plus[..., 0] > 0.0)
    np.testing.assert_array_equal(activity_minus[..., 0], 0.0)
    assert np.all(result.channel_rates_m2_s["wall_plus"][..., 0] > 0.0)
    np.testing.assert_array_equal(result.channel_rates_m2_s[
        "wall_minus"][..., 0], 0.0)
    # The new reaction is a signed reservoir relabeling: its affinity-rate
    # product dissipates and the complete production balance remains closed.
    assert np.min(result.channel_rates_m2_s[
        "wall_exchange_dissipation_W_m3"]) >= 0.0
    assert balance_ledger(
        polarized, result, systems, topologies
    )["relative_energy_balance_residual"] < 2e-14
    reversed_state = replace(
        polarized,
        mobile_plus_m2=polarized.mobile_minus_m2.copy(),
        mobile_minus_m2=polarized.mobile_plus_m2.copy(),
        forest_plus_m2=polarized.forest_minus_m2.copy(),
        forest_minus_m2=polarized.forest_plus_m2.copy(),
        wall_plus_m2=polarized.wall_minus_m2.copy(),
        wall_minus_m2=polarized.wall_plus_m2.copy())
    reversed_result = wall_residual(
        reversed_state, driving, systems, topologies, parameters)
    np.testing.assert_allclose(
        reversed_result.channel_rates_m2_s["wall_minus"],
        result.channel_rates_m2_s["wall_plus"], rtol=2e-15)
    np.testing.assert_allclose(
        reversed_result.channel_rates_m2_s["wall_plus"],
        result.channel_rates_m2_s["wall_minus"], atol=0.0)


def test_isolated_mobile_wall_capture_cannot_bypass_nucleation_barrier():
    systems, topologies, parameters, state, _ = fixture(n=8)
    parameters = replace(
        candidate_parameters(parameters),
        polarized_wall_absorption_enabled=True,
        polarized_mobile_wall_absorption_enabled=True)
    zero = np.zeros_like(state.wall_plus_m2)
    polarized = replace(
        state,
        mobile_plus_m2=np.full_like(state.mobile_plus_m2, 1.2e14),
        mobile_minus_m2=np.full_like(state.mobile_minus_m2, 8.0e13),
        forest_plus_m2=np.zeros_like(state.forest_plus_m2),
        forest_minus_m2=np.zeros_like(state.forest_minus_m2),
        wall_plus_m2=zero, wall_minus_m2=zero,
        wall_order=np.zeros_like(state.wall_order))
    driving = CommonWallDriving(
        resolved_stress_Pa=np.zeros_like(polarized.mobile_plus_m2))
    result = wall_residual(
        polarized, driving, systems, topologies, parameters)
    capture = result.channel_rates_m2_s["mobile_wall_plus"]
    np.testing.assert_array_equal(capture, 0.0)
    np.testing.assert_array_equal(
        result.channel_rates_m2_s["mobile_wall_minus"], 0.0)
    reversed_state = replace(
        polarized,
        mobile_plus_m2=polarized.mobile_minus_m2,
        mobile_minus_m2=polarized.mobile_plus_m2)
    reversed_result = wall_residual(
        reversed_state, driving, systems, topologies, parameters)
    np.testing.assert_array_equal(
        reversed_result.channel_rates_m2_s["mobile_wall_minus"], 0.0)
    np.testing.assert_array_equal(
        reversed_result.channel_rates_m2_s["mobile_wall_plus"], 0.0)


def test_neutral_pair_annihilation_cannot_run_uphill_in_declared_energy():
    systems, topologies, parameters, state, _ = fixture(n=8)
    parameters = candidate_parameters(parameters)
    # The low-density family-correlation branch has a negative neutral-pair
    # chemical potential. Unconditional mass-action annihilation would then
    # increase the complete free energy and report negative dissipation.
    dilute = replace(
        state,
        mobile_plus_m2=np.full_like(state.mobile_plus_m2, 1.0e12),
        mobile_minus_m2=np.full_like(state.mobile_minus_m2, 1.0e12),
        forest_plus_m2=np.zeros_like(state.forest_plus_m2),
        forest_minus_m2=np.zeros_like(state.forest_minus_m2),
        wall_plus_m2=np.zeros_like(state.wall_plus_m2),
        wall_minus_m2=np.zeros_like(state.wall_minus_m2),
        junction_m2=np.zeros_like(state.junction_m2),
        wall_order=np.zeros_like(state.wall_order))
    residual = wall_residual(
        dilute, CommonWallDriving(resolved_stress_Pa=np.zeros_like(
            dilute.mobile_plus_m2)), systems, topologies, parameters)
    np.testing.assert_array_equal(
        residual.channel_rates_m2_s["annihilation_affinity_factor"], 0.0)
    np.testing.assert_array_equal(
        residual.channel_rates_m2_s["annihilation_pairs"], 0.0)
    assert np.min(residual.channel_rates_m2_s[
        "neutral_pair_annihilation_dissipation_W_m3"]) >= 0.0


def test_sign_resolved_arrhenius_transport_conserves_and_closes_energy():
    systems, topologies, parameters, state, _ = fixture(n=32)
    parameters = candidate_parameters(parameters)
    state = patterned_state(state)
    stress = np.full_like(state.mobile_plus_m2, 2.0e9)
    residual = wall_residual(
        state, CommonWallDriving(resolved_stress_Pa=stress),
        systems, topologies, parameters)
    plus_rate = residual.channel_rates_m2_s[
        "transport_mobile_plus_rate_m2_s"]
    minus_rate = residual.channel_rates_m2_s[
        "transport_mobile_minus_rate_m2_s"]
    for rate in (plus_rate, minus_rate):
        imbalance = np.abs(np.sum(rate, axis=(0, 1)))
        turnover = np.sum(np.abs(rate), axis=(0, 1))
        assert np.max(imbalance/turnover) < 3e-15
    assert np.max(np.abs(
        residual.channel_rates_m2_s["speed_plus_m_s"]
        -residual.channel_rates_m2_s["speed_minus_m_s"])) > 0.0
    assert np.min(residual.heat_rate_W_m3) >= 0.0
    assert balance_ledger(
        state, residual, systems, topologies
    )["relative_energy_balance_residual"] < 2e-14


def test_groma_branch_uses_polarization_weighted_production_taylor_friction():
    systems, topologies, parameters, state, _ = fixture(n=8)
    parameters = candidate_parameters(parameters)
    stress = np.full_like(state.mobile_plus_m2, 5.0e8)
    drive = resolved_driving_components(
        state, CommonWallDriving(resolved_stress_Pa=stress),
        systems, topologies, parameters)
    polarization = ((state.mobile_plus_m2-state.mobile_minus_m2)
                    /(state.mobile_plus_m2+state.mobile_minus_m2))
    expected_plus = stress-(1.0-polarization)*drive["taylor_resistance_Pa"]
    expected_minus = stress-(1.0+polarization)*drive["taylor_resistance_Pa"]
    np.testing.assert_allclose(
        drive["effective_stress_plus_Pa"], expected_plus, rtol=2e-15)
    np.testing.assert_allclose(
        drive["effective_stress_minus_Pa"], expected_minus, rtol=2e-15)


def test_groma_taylor_friction_includes_mobile_self_content_only_when_enabled():
    systems, topologies, parameters, state, driving = fixture(n=8)
    baseline = resolved_driving_components(
        state, driving, systems, topologies, parameters)["taylor_resistance_Pa"]
    doubled = replace(
        state, mobile_plus_m2=2.0*state.mobile_plus_m2,
        mobile_minus_m2=2.0*state.mobile_minus_m2)
    np.testing.assert_array_equal(
        resolved_driving_components(
            doubled, driving, systems, topologies,
            parameters)["taylor_resistance_Pa"], baseline)
    enabled = candidate_parameters(parameters)
    arrhenius_driving = CommonWallDriving(
        resolved_stress_Pa=driving.resolved_stress_Pa)
    first = resolved_driving_components(
        state, arrhenius_driving, systems, topologies,
        enabled)["taylor_resistance_Pa"]
    second = resolved_driving_components(
        doubled, arrhenius_driving, systems, topologies,
        enabled)["taylor_resistance_Pa"]
    assert np.all(second > first)


def test_sign_and_stress_reversal_exchange_positive_negative_transport():
    systems, topologies, parameters, state, _ = fixture(n=16)
    parameters = candidate_parameters(parameters)
    state = patterned_state(state)
    stress = np.full_like(state.mobile_plus_m2, 1.8e9)
    original = wall_residual(
        state, CommonWallDriving(resolved_stress_Pa=stress),
        systems, topologies, parameters)
    reversed_state = replace(
        state, mobile_plus_m2=state.mobile_minus_m2.copy(),
        mobile_minus_m2=state.mobile_plus_m2.copy(),
        forest_plus_m2=state.forest_minus_m2.copy(),
        forest_minus_m2=state.forest_plus_m2.copy(),
        wall_plus_m2=state.wall_minus_m2.copy(),
        wall_minus_m2=state.wall_plus_m2.copy())
    reversed_residual = wall_residual(
        reversed_state, CommonWallDriving(resolved_stress_Pa=-stress),
        systems, topologies, parameters)
    np.testing.assert_allclose(
        reversed_residual.channel_rates_m2_s[
            "transport_mobile_plus_rate_m2_s"],
        original.channel_rates_m2_s["transport_mobile_minus_rate_m2_s"],
        rtol=2e-13, atol=2e5)
    np.testing.assert_allclose(
        reversed_residual.channel_rates_m2_s[
            "transport_mobile_minus_rate_m2_s"],
        original.channel_rates_m2_s["transport_mobile_plus_rate_m2_s"],
        rtol=2e-13, atol=2e5)
