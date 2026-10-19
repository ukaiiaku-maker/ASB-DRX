import numpy as np

from full_model.production.common_tensorial_wall import (
    CommonWallParameters, exp_floor_rate, glide_exp_floor_parameters,
    glide_velocity_m_s,
)


def test_zero_temperature_coefficients_recover_qualified_rate_exactly():
    parameters = CommonWallParameters(spacing_m=1e-7)
    stress = np.array([.5e9, 1.5e9, 3e9])
    temperature = np.array([900.0, 1000.0, 1100.0])
    old = (parameters.glide_speed_attempt_m_s
           *exp_floor_rate(stress, temperature,
                           parameters.glide_barrier_eV, parameters)
           /parameters.attempt_frequency_s
           *np.tanh(stress/parameters.critical_stress_Pa))
    assert np.array_equal(glide_velocity_m_s(
        stress, temperature, parameters), old)


def test_preserved_analytical_scaling_softens_barrier_and_critical_stress():
    parameters = CommonWallParameters(
        spacing_m=1e-7,
        kinetic_reference_temperature_K=1000.0,
        glide_barrier_temperature_coefficient=1.24174,
        critical_stress_temperature_coefficient=.108506)
    barrier, critical = glide_exp_floor_parameters(
        np.array([1000.0, 1050.0]), parameters)
    assert barrier[1] < barrier[0]
    assert critical[1] < critical[0]
    speed = glide_velocity_m_s(1.5e9, np.array([1000.0, 1050.0]), parameters)
    assert speed[1] > speed[0]


def test_reference_temperature_is_an_exact_common_state_anchor():
    baseline = CommonWallParameters(spacing_m=1e-7)
    candidate = CommonWallParameters(
        spacing_m=1e-7,
        kinetic_reference_temperature_K=1000.0,
        glide_barrier_temperature_coefficient=1.24174,
        critical_stress_temperature_coefficient=.108506)
    stress = np.linspace(-3e9, 3e9, 17)
    assert np.array_equal(
        glide_velocity_m_s(stress, 1000.0, baseline),
        glide_velocity_m_s(stress, 1000.0, candidate))
