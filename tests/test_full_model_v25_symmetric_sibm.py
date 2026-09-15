import numpy as np

from full_model.production.symmetric_sibm import (
    clean_periodic_bicrystal, common_complete_phase_energies, complete_energy,
    flat_front_variational_audit, phase_only_directional_audit,
    sequential_activations,
)


def test_clean_bicrystal_is_zero_load_and_all_defect_freeze_is_exact():
    state = clean_periodic_bicrystal()
    assert np.max(np.abs(state["stress_Pa"])) == 0.0
    assert np.max(np.abs(state["plastic_strain"])) == 0.0
    stages = sequential_activations()
    assert stages[0].all_defects_frozen
    assert not any(stage.all_defects_frozen for stage in stages[1:])


def test_complete_functional_is_label_permutation_symmetric():
    state = clean_periodic_bicrystal(n=128)
    eta = state["eta"]; dx = state["x_m"][1]-state["x_m"][0]
    energy = common_complete_phase_energies(
        [8e7, 2e7], [3e6, 7e6], [1e6, 4e6], [2e6, 5e6], [0, 0])
    first = complete_energy(eta, energy, dx, 5e-7, 5e6)["total_J_m2"]
    second = complete_energy(eta[:, ::-1], energy[::-1], dx, 5e-7, 5e6)["total_J_m2"]
    np.testing.assert_allclose(first, second, rtol=2e-15)


def test_equal_stationary_and_contrast_reversal_reverses_phase_only_motion():
    equal = phase_only_directional_audit(4e7, 4e7)
    favorable = phase_only_directional_audit(8e7, 2e7)
    reverse = phase_only_directional_audit(2e7, 8e7)
    assert abs(equal["displacement_interface_widths"]) < 1e-8
    assert favorable["observed_sign"] == 1
    assert reverse["observed_sign"] == -1
    assert favorable["stable_velocity_sign"] and reverse["stable_velocity_sign"]


def test_termwise_variational_derivative_matches_centered_difference_and_reverses():
    state = clean_periodic_bicrystal(); eta = state["eta"]
    dx = state["x_m"][1]-state["x_m"][0]
    base = dict(phase=[0, 0], defect=[8e7, 2e7], elastic=[0, 0],
                gnd=[0, 0], compatibility=[0, 0], boundary=[0, 0])
    forward = flat_front_variational_audit(eta, base, dx, 5e-7, 5e6)
    reverse = flat_front_variational_audit(
        eta, {**base, "defect": [2e7, 8e7]}, dx, 5e-7, 5e6)
    assert forward["relative_derivative_mismatch"] < 2e-5
    assert reverse["relative_derivative_mismatch"] < 2e-5
    assert forward["analytical_total_Pa"] < 0
    assert reverse["analytical_total_Pa"] > 0
