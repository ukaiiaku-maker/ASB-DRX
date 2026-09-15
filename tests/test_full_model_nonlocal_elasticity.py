import numpy as np

from full_model.production.nonlocal_elasticity import (
    elastic_energy_density, solve_periodic_eigenstrain,
)


CONSTANTS = (228e9, 132e9, 116.5e9)


def patterned_eigenstrain(n):
    x = np.arange(n)[:, None]/n
    y = np.arange(n)[None, :]/n
    field = np.zeros((n, n, 2, 2))
    field[..., 0, 1] = field[..., 1, 0] = .003*np.sin(2*np.pi*x)*np.cos(2*np.pi*y)
    return field


def test_nonlocal_stress_reverses_with_signed_plastic_distortion():
    eigen = patterned_eigenstrain(24)
    stress, strain = solve_periodic_eigenstrain(
        eigen, np.zeros((2, 2)), 2e-7, *CONSTANTS)
    reversed_stress, reversed_strain = solve_periodic_eigenstrain(
        -eigen, np.zeros((2, 2)), 2e-7, *CONSTANTS)
    np.testing.assert_allclose(reversed_stress, -stress, rtol=2e-13, atol=1e-3)
    np.testing.assert_allclose(reversed_strain, -strain, rtol=2e-13, atol=1e-15)


def test_zero_mode_and_elastic_energy_are_closed():
    eigen = patterned_eigenstrain(20)
    mean = np.array([[.002, 0.0], [0.0, -.0005]])
    stress, strain = solve_periodic_eigenstrain(eigen, mean, 3e-7, *CONSTANTS)
    np.testing.assert_allclose(np.mean(strain, axis=(0, 1)), mean, atol=2e-16)
    assert float(np.sum(elastic_energy_density(stress, strain, eigen))) > 0.0


def test_work_conjugacy_by_directional_energy_derivative():
    n = 18
    eigen = patterned_eigenstrain(n)
    perturbation = patterned_eigenstrain(n)*.37
    mean = np.zeros((2, 2))

    def energy(scale):
        current = eigen+scale*perturbation
        stress, strain = solve_periodic_eigenstrain(
            current, mean, 2.5e-7, *CONSTANTS, iterations=8)
        return float(np.sum(elastic_energy_density(stress, strain, current)))

    h = 1e-5
    derivative = (energy(h)-energy(-h))/(2*h)
    stress, _ = solve_periodic_eigenstrain(
        eigen, mean, 2.5e-7, *CONSTANTS, iterations=8)
    conjugate = -float(np.sum(stress*perturbation))
    np.testing.assert_allclose(derivative, conjugate, rtol=2e-7, atol=2e-2)
