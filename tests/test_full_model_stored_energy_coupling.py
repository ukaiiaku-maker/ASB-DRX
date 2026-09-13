import unittest

import numpy as np

from full_model.production.stored_energy_coupling import (
    common_variational_stored_energy,
    phase_mean_stored_energy_states,
)


class CommonStoredEnergyCouplingTest(unittest.TestCase):
    def test_two_phase_derivative_has_thermodynamic_sign(self):
        eta = np.array([[0.5, 0.5]])
        energy = np.array([[3.0e8, 2.0e7]])
        mixture, derivative = common_variational_stored_energy(eta, energy)
        self.assertEqual(mixture[0], 1.6e8)
        self.assertGreater(derivative[0, 0], 0.0)
        self.assertLess(derivative[0, 1], 0.0)

    def test_derivative_matches_finite_difference_of_declared_functional(self):
        eta = np.array([[0.31, 0.69]])
        energy = np.array([[4.0e8, 5.0e7]])
        _, derivative = common_variational_stored_energy(eta, energy)
        eps = 1e-7
        for i in range(2):
            plus = eta.copy(); plus[0, i] += eps
            minus = eta.copy(); minus[0, i] -= eps
            f_plus = common_variational_stored_energy(plus, energy)[0][0]
            f_minus = common_variational_stored_energy(minus, energy)[0][0]
            self.assertAlmostEqual(
                derivative[0, i], (f_plus-f_minus)/(2*eps), delta=50.0)

    def test_pure_and_equal_energy_states_are_stationary(self):
        pure = np.array([[1.0, 0.0]])
        _, derivative = common_variational_stored_energy(
            pure, np.array([[4.0e8, 1.0e7]]))
        self.assertTrue(np.array_equal(derivative, np.zeros_like(derivative)))
        eta = np.array([[0.2, 0.8]])
        _, derivative = common_variational_stored_energy(
            eta, np.full_like(eta, 7.0e7))
        self.assertTrue(np.array_equal(derivative, np.zeros_like(derivative)))

    def test_label_permutation_symmetry(self):
        eta = np.array([[[0.2, 0.3, 0.5]]])
        energy = np.array([[[2.0e8, 8.0e7, 4.0e7]]])
        permutation = [2, 0, 1]
        f0, d0 = common_variational_stored_energy(eta, energy)
        f1, d1 = common_variational_stored_energy(
            eta[:, :, permutation], energy[:, :, permutation])
        inverse = np.argsort(permutation)
        np.testing.assert_allclose(f0, f1, rtol=3e-16, atol=0.0)
        np.testing.assert_allclose(d0, d1[:, :, inverse], rtol=2e-15, atol=0.0)

    def test_phase_states_follow_current_density_and_rehardening(self):
        eta = np.zeros((2, 2, 2))
        eta[:, 0, 0] = 1.0
        eta[:, 1, 1] = 1.0
        stored = np.array([[3.0e8, 2.0e7], [4.0e8, 4.0e7]])
        _, values = phase_mean_stored_energy_states(eta, stored)
        self.assertTrue(np.array_equal(values, np.array([3.5e8, 3.0e7])))
        stored[:, 1] += 1.0e8
        _, rehard = phase_mean_stored_energy_states(eta, stored)
        self.assertEqual(rehard[0], values[0])
        self.assertGreater(rehard[1], values[1])


if __name__ == "__main__":
    unittest.main()
