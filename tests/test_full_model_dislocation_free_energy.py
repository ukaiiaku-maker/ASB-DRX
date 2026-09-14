import unittest

import numpy as np

from full_model.production.dislocation_free_energy import (
    DislocationFreeEnergyParameters,
    chemical_potential_J_m,
    free_energy_J_m3,
    hessian_J_m3_per_m4,
    logarithmic_chemical_potential_J_m,
    logarithmic_energy_J_m3,
    logarithmic_hessian_J_m3_per_m4,
    phase_owned_free_energy_J_m3,
)


class DislocationFreeEnergyTest(unittest.TestCase):
    def setUp(self):
        self.p = DislocationFreeEnergyParameters(
            line_coefficient_J_m=2.5e-9,
            log_coefficient_J_m=1.5e-10,
            reference_density_m2=1.0e14,
            ordering_amplitude_J_m3=4.0e5,
            ordering_density_scale_m2=5.0e14,
            ordering_center_ratio=1.3,
            ordering_width_ratio=0.3,
            low_density_strength=0.2,
        )

    def test_rho_log_rho_has_exact_zero_limit_and_si_scale(self):
        values = logarithmic_energy_J_m3(
            np.array([0.0, 1.0e-100, 1.0e14]), 1.5e-10, 1.0e14
        )
        self.assertEqual(values[0], 0.0)
        self.assertLess(abs(values[1]), 1.0e-95)
        self.assertAlmostEqual(values[2], -1.5e4)

    def test_log_mu_and_hessian_match_centered_finite_differences(self):
        for rho in np.logspace(10, 17, 9):
            step = 2.0e-5*rho
            plus = logarithmic_energy_J_m3(rho+step, 1.5e-10, 1.0e14)
            minus = logarithmic_energy_J_m3(rho-step, 1.5e-10, 1.0e14)
            fd_mu = (plus-minus)/(2.0*step)
            mu = logarithmic_chemical_potential_J_m(rho, 1.5e-10, 1.0e14)
            self.assertAlmostEqual(float(fd_mu/mu), 1.0, places=8)
            mu_plus = logarithmic_chemical_potential_J_m(rho+step, 1.5e-10, 1.0e14)
            mu_minus = logarithmic_chemical_potential_J_m(rho-step, 1.5e-10, 1.0e14)
            fd_h = (mu_plus-mu_minus)/(2.0*step)
            exact_h = logarithmic_hessian_J_m3_per_m4(rho, 1.5e-10)
            self.assertAlmostEqual(float(fd_h/exact_h), 1.0, places=8)
            self.assertGreater(float(exact_h), 0.0)

    def test_complete_phi_prime_and_second_derivative(self):
        for rho in np.logspace(12, 16, 13):
            step = 1.0e-4*rho
            fd_mu = (
                free_energy_J_m3(rho+step, self.p)
                - free_energy_J_m3(rho-step, self.p)
            )/(2.0*step)
            mu = chemical_potential_J_m(rho, self.p)
            self.assertAlmostEqual(float(fd_mu), float(mu), delta=1e-7*max(abs(float(mu)), 1e-12))
            fd_h = (
                chemical_potential_J_m(rho+step, self.p)
                - chemical_potential_J_m(rho-step, self.p)
            )/(2.0*step)
            exact_h = hessian_J_m3_per_m4(rho, self.p)
            self.assertAlmostEqual(float(fd_h), float(exact_h), delta=3e-7*max(abs(float(exact_h)), 1e-30))

    def test_log_floor_is_only_an_evaluation_guard(self):
        rho = np.array([1.0e12, 1.0e14, 1.0e16])
        a = logarithmic_chemical_potential_J_m(
            rho, 1.5e-10, 1.0e14, evaluation_floor_m2=1.0
        )
        b = logarithmic_chemical_potential_J_m(
            rho, 1.5e-10, 1.0e14, evaluation_floor_m2=1.0e8
        )
        np.testing.assert_array_equal(a, b)
        self.assertEqual(logarithmic_energy_J_m3(0.0, 1.5e-10, 1.0e14), 0.0)

    def test_temperature_dependence_has_one_shared_modulus_factor(self):
        def params(mu):
            line = 0.5*mu*(2.48e-10)**2
            return DislocationFreeEnergyParameters(
                line, 0.06*line, 1e14, 1.2*line*5e14*0.3,
                5e14, 1.3, 0.3,
            )
        rho = 6e14
        ratio = free_energy_J_m3(rho, params(40e9))/free_energy_J_m3(rho, params(80e9))
        self.assertAlmostEqual(float(ratio), 0.5, places=13)

    def test_only_organized_wall_receives_ordering_credit(self):
        rho = self.p.ordering_center_ratio*self.p.ordering_density_scale_m2
        disordered = phase_owned_free_energy_J_m3(rho, 0.0, self.p)
        ordered = phase_owned_free_energy_J_m3(rho, rho, self.p)
        self.assertAlmostEqual(float(disordered-ordered),
                               self.p.ordering_amplitude_J_m3, places=8)


if __name__ == '__main__':
    unittest.main()
