from dataclasses import replace

import numpy as np

from full_model.production.common_tensorial_wall import (
    CommonWallDriving, CommonWallParameters, CommonWallState,
    multiplication_density_argument_m2, raw_multiplication_pair_rate_m2_s,
    wall_residual,
)
from full_model.production.tensorial_nye import bcc_four_family_systems


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


def test_zero_coefficient_is_exact_no_multiplication_comparator():
    state = _state()
    pair = raw_multiplication_pair_rate_m2_s(
        state, np.ones(state.mobile_plus_m2.shape),
        _parameters(multiplication_law="disabled",
                    multiplication_coefficient_m1=0.0))
    assert np.count_nonzero(pair) == 0
