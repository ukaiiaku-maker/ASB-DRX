from dataclasses import replace

import numpy as np

from full_model.production.common_tensorial_wall import (
    CommonWallDriving, CommonWallParameters, CommonWallState,
    accepted_euler_step,
    multiplication_density_argument_m2, raw_multiplication_pair_rate_m2_s,
    wall_residual,
)
from full_model.production.tensorial_nye import bcc_four_family_systems
from full_model.analysis.run_v67_drx_asb_overlap import (
    _front_transmission_fraction, _multiplication_parameters,
)


def _state(rho=1e14, n=4):
    shape = (n, n, 4)
    z = np.zeros(shape)
    return CommonWallState(
        .25*rho+z, .25*rho+z, .25*rho+z, .25*rho+z,
        z.copy(), z.copy(), np.zeros((n, n, 0)), np.zeros((n, n)),
        np.zeros((n, n)), z.copy(), np.zeros((n, n, 3, 3)),
        np.zeros((n, n, 4, 3)), np.zeros((n, n, 4, 3, 3)),
        np.zeros((n, n)), np.full((n, n), 1000.0))


def _parameters(**changes):
    return replace(CommonWallParameters(
        spacing_m=1e-7, multiplication_law="v34_km_forest_storage_v1",
        multiplication_coefficient_m1=5e8,
        multiplication_storage_work_fraction=.1,
        multiplication_forest_mix=.35,
        multiplication_mobile_obstacle_weight=.10,
        multiplication_wall_obstacle_weight=.35,
        multiplication_gnd_obstacle_weight=.10), **changes)


def test_manufactured_km_source_has_m_inverse_units_and_sign_symmetry():
    state = _state()
    parameters = _parameters()
    slip = np.full(state.mobile_plus_m2.shape, 2.0)
    density = multiplication_density_argument_m2(state, parameters)
    pair = raw_multiplication_pair_rate_m2_s(state, slip, parameters)
    np.testing.assert_allclose(
        2.0*pair,
        parameters.multiplication_coefficient_m1*np.sqrt(density)*abs(slip),
        rtol=1e-15)
    np.testing.assert_array_equal(
        pair, raw_multiplication_pair_rate_m2_s(state, -slip, parameters))


def test_recovered_obstacle_argument_matches_saved_v34_partition_semantics():
    state = _state(rho=0.0, n=1)
    shape = state.mobile_plus_m2.shape
    mp = np.broadcast_to(np.array([[[7.0, 3.0, 1.0, 2.0]]]), shape).copy()
    mm = np.broadcast_to(np.array([[[2.0, 4.0, 1.0, 1.0]]]), shape).copy()
    fp = np.broadcast_to(np.array([[[11.0, 13.0, 17.0, 19.0]]]), shape).copy()
    fm = np.broadcast_to(np.array([[[5.0, 7.0, 11.0, 13.0]]]), shape).copy()
    wp = np.broadcast_to(np.array([[[23.0, 0.0, 0.0, 0.0]]]), shape).copy()
    wm = np.broadcast_to(np.array([[[0.0, 29.0, 0.0, 0.0]]]), shape).copy()
    state = replace(
        state, mobile_plus_m2=mp, mobile_minus_m2=mm,
        forest_plus_m2=fp, forest_minus_m2=fm,
        wall_plus_m2=wp, wall_minus_m2=wm)
    parameters = _parameters()
    forest = fp+fm
    expected = (
        .65*forest+.35*np.sum(forest, axis=2)[..., None]
        +.10*(mp+mm)+.35*np.sum(wp+wm, axis=2)[..., None]
        +.10*np.abs(np.sum(mp-mm, axis=2))[..., None])
    np.testing.assert_allclose(
        multiplication_density_argument_m2(state, parameters), expected,
        rtol=0.0, atol=0.0)


def test_recovered_km_routes_generated_line_to_forest_not_mobile():
    state = _state()
    systems = bcc_four_family_systems()
    parameters = _parameters(
        multiplication_coefficient_m1=2e5,
        enforce_multiplication_energy_budget=False,
        attempt_frequency_s=1e-30)
    zero = replace(parameters, multiplication_law="disabled",
                   multiplication_coefficient_m1=0.0)
    driving = CommonWallDriving(
        glide_speed_m_s=np.full(state.mobile_plus_m2.shape, 1e-3),
        resolved_stress_Pa=np.full(state.mobile_plus_m2.shape, 1e9))
    active = wall_residual(state, driving, systems, (), parameters)
    base = wall_residual(state, driving, systems, (), zero)
    pair = active.channel_rates_m2_s["multiplication_pairs"]
    np.testing.assert_allclose(
        active.state_rate.mobile_plus_m2-base.state_rate.mobile_plus_m2, 0.0,
        atol=1e-12)
    np.testing.assert_allclose(
        active.state_rate.mobile_minus_m2-base.state_rate.mobile_minus_m2, 0.0,
        atol=1e-12)
    np.testing.assert_allclose(
        active.state_rate.forest_plus_m2-base.state_rate.forest_plus_m2, pair,
        rtol=1e-14)
    np.testing.assert_allclose(
        active.state_rate.forest_minus_m2-base.state_rate.forest_minus_m2, pair,
        rtol=1e-14)


def test_recovered_km_budget_caps_storage_by_saved_work_fraction():
    state = _state()
    systems = bcc_four_family_systems()
    parameters = _parameters()
    driving = CommonWallDriving(
        glide_speed_m_s=np.full(state.mobile_plus_m2.shape, 2e-4),
        resolved_stress_Pa=np.full(state.mobile_plus_m2.shape, 2e6))
    residual = wall_residual(state, driving, systems, (), parameters)
    channels = residual.channel_rates_m2_s
    scale = channels["multiplication_energy_budget_scale"]
    assert np.max(scale) < 1.0
    assert np.all(channels["multiplication_pairs"]
                  <= channels["multiplication_pairs_raw"])
    accepted_storage_power = np.sum(
        channels["multiplication_pairs"], axis=2
    )/np.maximum(np.sum(
        channels["multiplication_pairs_raw"], axis=2), 1e-300
    )*channels["multiplication_cost_raw_W_m3"]
    assert np.all(accepted_storage_power
                  <= channels["multiplication_power_available_W_m3"]
                  *(1.0+2e-14))


def test_zero_coefficient_is_exact_no_multiplication_comparator():
    state = _state()
    pair = raw_multiplication_pair_rate_m2_s(
        state, np.ones(state.mobile_plus_m2.shape),
        _parameters(multiplication_law="disabled",
                    multiplication_coefficient_m1=0.0))
    assert np.count_nonzero(pair) == 0


def test_storage_law_does_not_treat_immobile_forest_as_operable_source():
    state = _state()
    state = replace(
        state,
        mobile_plus_m2=np.zeros_like(state.mobile_plus_m2),
        mobile_minus_m2=np.zeros_like(state.mobile_minus_m2))
    residual = wall_residual(
        state,
        CommonWallDriving(
            glide_speed_m_s=np.full(state.mobile_plus_m2.shape, 1e-3),
            resolved_stress_Pa=np.full(state.mobile_plus_m2.shape, 1e9)),
        bcc_four_family_systems(), (), _parameters())
    assert np.count_nonzero(residual.state_rate.slip) == 0
    assert np.count_nonzero(
        residual.channel_rates_m2_s["multiplication_pairs_raw"]) == 0


def test_production_kernel_converges_to_pure_storage_scalar_solution():
    rho0 = 2.0e13
    coefficient = 5.0e8
    accumulated_slip = 2.0e-3
    state0 = _state(rho=0.0, n=1)
    forest_plus = np.zeros_like(state0.forest_plus_m2)
    forest_minus = np.zeros_like(state0.forest_minus_m2)
    forest_plus[..., 0] = 0.5*rho0
    forest_minus[..., 0] = 0.5*rho0
    state0 = replace(
        state0, forest_plus_m2=forest_plus, forest_minus_m2=forest_minus)
    parameters = _parameters(
        multiplication_coefficient_m1=coefficient,
        multiplication_mobile_obstacle_weight=0.0,
        multiplication_wall_obstacle_weight=0.0,
        multiplication_gnd_obstacle_weight=0.0)
    slip = np.zeros_like(state0.mobile_plus_m2)
    slip[..., 0] = 1.0

    def integrate(steps):
        state = state0
        dgamma = accumulated_slip/steps
        for _ in range(steps):
            pair = raw_multiplication_pair_rate_m2_s(
                state, slip, parameters)
            state = replace(
                state,
                forest_plus_m2=state.forest_plus_m2+dgamma*pair,
                forest_minus_m2=state.forest_minus_m2+dgamma*pair)
        return (state.forest_plus_m2[..., 0]
                +state.forest_minus_m2[..., 0]).item()

    coarse = integrate(10000)
    rho = integrate(20000)
    exact = (np.sqrt(rho0)+0.5*coefficient*accumulated_slip)**2
    assert abs(rho-exact) < abs(coarse-exact)
    np.testing.assert_allclose(rho, exact, rtol=6e-7)


def test_accepted_production_step_agrees_with_two_half_steps():
    state = _state()
    systems = bcc_four_family_systems()
    parameters = _parameters(
        enforce_multiplication_energy_budget=False,
        attempt_frequency_s=1e-30,
        thermal_diffusivity_m2_s=0.0,
        mobile_correlation_diffusivity_m2_s=0.0)
    driving = CommonWallDriving(
        glide_speed_m_s=np.full(state.mobile_plus_m2.shape, 1e-3),
        resolved_stress_Pa=np.full(state.mobile_plus_m2.shape, 1e9))
    dt = 1e-8
    full, _, _ = accepted_euler_step(
        state, driving, systems, (), parameters, dt)
    half, _, _ = accepted_euler_step(
        state, driving, systems, (), parameters, .5*dt)
    halves, _, _ = accepted_euler_step(
        half, driving, systems, (), parameters, .5*dt)
    full_increment = np.sum(
        full.forest_plus_m2+full.forest_minus_m2
        -state.forest_plus_m2-state.forest_minus_m2)
    half_increment = np.sum(
        halves.forest_plus_m2+halves.forest_minus_m2
        -state.forest_plus_m2-state.forest_minus_m2)
    assert full_increment > 0.0
    assert half_increment > 0.0
    np.testing.assert_allclose(half_increment, full_increment, rtol=2e-6)


def test_multiplication_schema_restart_is_explicit_and_backward_compatible():
    base = CommonWallParameters(spacing_m=1e-7)
    old = _multiplication_parameters(base, "inherited", {})
    assert old.multiplication_law == "legacy_v21_mobile_pair_v1"
    assert old.multiplication_coefficient_m1 == 10.0
    saved = {
        "multiplication_law": "v34_km_forest_storage_v1",
        "multiplication_coefficient_m1": 5.0e8,
        "multiplication_storage_work_fraction": .1,
        "multiplication_forest_mix": .31,
        "multiplication_mobile_obstacle_weight": .11,
        "multiplication_wall_obstacle_weight": .37,
        "multiplication_gnd_obstacle_weight": .09,
    }
    restored = _multiplication_parameters(base, "inherited", saved)
    assert restored.multiplication_law == "v34_km_forest_storage_v1"
    assert restored.multiplication_coefficient_m1 == 5e8
    assert restored.multiplication_forest_mix == .31
    assert restored.multiplication_mobile_obstacle_weight == .11
    disabled = _multiplication_parameters(base, "disabled", saved)
    assert disabled.multiplication_law == "disabled"
    assert disabled.multiplication_coefficient_m1 == 0.0


def test_front_transmission_restart_inherits_parent_unless_overridden():
    assert _front_transmission_fraction(
        None, {"front_transmission_fraction": .25}, .55) == .25
    assert _front_transmission_fraction(
        .40, {"front_transmission_fraction": .25}, .55) == .40
    assert _front_transmission_fraction(None, {}, .55) == .55
