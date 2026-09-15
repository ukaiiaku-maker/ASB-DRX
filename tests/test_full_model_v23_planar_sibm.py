import numpy as np

from full_model.production.sibm_geometry import pinned_cap_energy
from full_model.production.stored_energy_coupling import (
    common_variational_stored_energy, frozen_planar_sibm_response,
)


def test_zero_pressure_planar_equal_favorable_reverse_and_mobility_off():
    mobility = 2e-12
    equal = frozen_planar_sibm_response(4e7, 4e7, mobility)
    favorable = frozen_planar_sibm_response(8e7, 2e7, mobility)
    reverse = frozen_planar_sibm_response(2e7, 8e7, mobility)
    off = frozen_planar_sibm_response(8e7, 2e7, 0.0)
    assert equal["normal_velocity_m_s"] == 0
    assert favorable["normal_velocity_m_s"] > 0
    assert reverse["normal_velocity_m_s"] < 0
    assert off["normal_velocity_m_s"] == 0
    for case in (equal, favorable, reverse, off):
        assert case["external_pressure_Pa"] == 0
        assert case["curvature_m1"] == 0


def test_planar_velocity_sign_matches_exact_common_functional_derivative():
    eta = np.array([[0.5, 0.5]])
    energies = np.array([[8e7, 2e7]])
    _, derivative = common_variational_stored_energy(eta, energies)
    direction = np.array([[-1.0, 1.0]])
    analytical = float(np.sum(derivative*direction))
    epsilon = 1e-7
    fp = common_variational_stored_energy(eta+epsilon*direction, energies)[0][0]
    fm = common_variational_stored_energy(eta-epsilon*direction, energies)[0][0]
    finite_difference = (fp-fm)/(2*epsilon)
    np.testing.assert_allclose(analytical, finite_difference, rtol=1e-9)
    response = frozen_planar_sibm_response(8e7, 2e7, 2e-12)
    assert analytical < 0 and response["normal_velocity_m_s"] > 0


def test_cap_uses_actual_amplitude_derivative_not_flat_pressure_sign():
    amplitude = 6e-7; chord = 1.5e-6; gamma = .5
    base = pinned_cap_energy(
        amplitude, chord, boundary_energy_J_m2=gamma, stored_pressure_Pa=0.0)
    critical = gamma*base["d_arc_length_da"]/base["d_swept_area_da_m"]
    for factor, expected in ((.8, 1), (1.2, -1)):
        kwargs = dict(boundary_energy_J_m2=gamma,
                      stored_pressure_Pa=factor*critical,
                      represented_thickness_m=1e-6)
        audit = pinned_cap_energy(amplitude, chord, **kwargs)
        h = 1e-10
        fd = (pinned_cap_energy(amplitude+h, chord, **kwargs)["total_energy_J"]
              -pinned_cap_energy(amplitude-h, chord, **kwargs)["total_energy_J"])/(2*h)
        np.testing.assert_allclose(audit["d_total_energy_da_J_m"], fd, rtol=2e-7)
        assert np.sign(audit["d_total_energy_da_J_m"]) == expected
