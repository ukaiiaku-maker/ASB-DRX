import numpy as np

from full_model.production.arrhenius_kinetics import (
    exp_floor_activation_volume_m3, exp_floor_enthalpy_j,
)


def test_exp_floor_activation_volume_is_barrier_derivative_not_event_volume():
    stress = 6.2e8
    arguments = (2.4e-19, 1.5e9, 2.2, 2.5, .05)
    step = 100.0
    finite_difference = -(
        exp_floor_enthalpy_j(stress+step, *arguments)
        -exp_floor_enthalpy_j(stress-step, *arguments))/(2*step)
    analytic = exp_floor_activation_volume_m3(stress, *arguments)
    np.testing.assert_allclose(analytic, finite_difference, rtol=2e-8)
    assert analytic > 0.0
    assert exp_floor_activation_volume_m3(-stress, *arguments) == 0.0
