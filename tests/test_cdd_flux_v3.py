from __future__ import annotations

import unittest

import numpy as np

from asb_drx.cdd_flux_v3 import (
    correlation_rhs,
    discrete_mode_growth_rates_s_inv,
    logarithmic_mean,
    variational_chemical_potentials_J_m,
    variational_correlation_energy_J_m3,
    variational_correlation_fluxes,
    variational_correlation_imex_fluxes,
    variational_dissipation_J_m2_s,
)
from asb_drx.physical_noise import periodic_physical_noise


class CDDFluxV3Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.n = 64
        self.domain = 16.0e-6
        self.dx = self.domain / self.n
        self.rho = 5.0e14
        self.mobility = np.full((2, self.n), 2.0e-13)
        self.mu = np.full(self.n, 40.0e9)
        self.burgers = 2.86e-10

    def rhs(self, plus, minus, face_density):
        return correlation_rhs(
            plus, minus, self.mobility, self.mu, self.burgers, self.dx,
            backstress_coefficient=1.0, diffusion_coefficient=1.0,
            face_density=face_density,
        )

    def test_both_face_forms_close_each_periodic_population_balance(self) -> None:
        rng = np.random.default_rng(4)
        plus = self.rho * (0.5 + 0.02 * rng.normal(size=(2, self.n)))
        minus = self.rho * (0.5 + 0.02 * rng.normal(size=(2, self.n)))
        for candidate in ("arithmetic", "logarithmic"):
            rhs_plus, rhs_minus = self.rhs(plus, minus, candidate)
            np.testing.assert_allclose(np.sum(rhs_plus, axis=1), 0.0, atol=2.0)
            np.testing.assert_allclose(np.sum(rhs_minus, axis=1), 0.0, atol=2.0)

    def test_both_face_forms_remove_nyquist_null_and_damp_correlation_modes(self) -> None:
        for candidate in ("arithmetic", "logarithmic"):
            spectrum = discrete_mode_growth_rates_s_inv(
                64, self.domain, self.rho, 2.0e-13, 40.0e9,
                self.burgers, backstress_coefficient=1.0,
                diffusion_coefficient=1.0, face_density=candidate,
            )
            self.assertLess(spectrum["total_s_inv"][-1], 0.0)
            self.assertLess(spectrum["signed_s_inv"][-1], 0.0)
            self.assertTrue(all(value < 0.0 for value in spectrum["total_s_inv"]))
            self.assertTrue(all(value < 0.0 for value in spectrum["signed_s_inv"]))

    def test_low_mode_symbol_converges_second_order(self) -> None:
        errors = []
        diffusivity = 2.0e-13 * 40.0e9 * self.burgers
        continuum = -diffusivity * (2.0 * np.pi / self.domain) ** 2
        for n in (32, 64, 128):
            spectrum = discrete_mode_growth_rates_s_inv(
                n, self.domain, self.rho, 2.0e-13, 40.0e9,
                self.burgers, backstress_coefficient=1.0,
                diffusion_coefficient=1.0, face_density="arithmetic",
            )
            errors.append(abs(spectrum["total_s_inv"][0] - continuum))
        self.assertLess(errors[1] / errors[0], 0.27)
        self.assertLess(errors[2] / errors[1], 0.27)

    def test_logarithmic_mean_identity(self) -> None:
        left = np.asarray([1.0, 2.0, 10.0])
        right = np.asarray([1.0, 5.0, 4.0])
        mean = logarithmic_mean(left, right)
        np.testing.assert_allclose(
            mean * (np.log(right) - np.log(left)), right - left,
            rtol=2.0e-15, atol=2.0e-15,
        )

    def test_physical_noise_is_grid_restriction_and_not_fixed_mode_count(self) -> None:
        coarse = periodic_physical_noise(64, self.domain, 0.5e-6, 9)
        fine = periodic_physical_noise(128, self.domain, 0.5e-6, 9)
        np.testing.assert_allclose(coarse, fine[::2], rtol=0.0, atol=2.0e-14)
        # Physical cutoff means a longer box contains proportionally more
        # Fourier modes; it is not the old fixed-12-mode fixture.
        short = periodic_physical_noise(104, 13.0e-6, 0.5e-6, 9)
        long = periodic_physical_noise(152, 19.0e-6, 0.5e-6, 9)
        short_modes = np.count_nonzero(np.abs(np.fft.rfft(short)) > 1.0e-9)
        long_modes = np.count_nonzero(np.abs(np.fft.rfft(long)) > 1.0e-9)
        self.assertGreater(long_modes, short_modes)

    def test_variational_chemical_potential_is_energy_derivative(self) -> None:
        rng = np.random.default_rng(21)
        plus = self.rho * (0.5 + 0.03 * rng.normal(size=(2, self.n)))
        minus = self.rho * (0.5 + 0.03 * rng.normal(size=(2, self.n)))
        kwargs = dict(
            backstress_coefficient=1.0, diffusion_coefficient=1.0,
            reference_density_m2=self.rho, density_floor_m2=1.0e8,
        )
        chemical_plus, _ = variational_chemical_potentials_J_m(
            plus, minus, self.mu, self.burgers, **kwargs
        )
        family, cell = 1, 7
        increment = 1.0e7
        shifted_plus = plus.copy(); shifted_plus[family, cell] += increment
        shifted_minus = plus.copy(); shifted_minus[family, cell] -= increment
        energy0 = variational_correlation_energy_J_m3(
            plus, minus, self.mu, self.burgers, **kwargs
        )
        energy1 = variational_correlation_energy_J_m3(
            shifted_plus, minus, self.mu, self.burgers, **kwargs
        )
        energy_minus = variational_correlation_energy_J_m3(
            shifted_minus, minus, self.mu, self.burgers, **kwargs
        )
        numerical = (energy1[cell] - energy_minus[cell]) / (2.0 * increment)
        self.assertAlmostEqual(
            numerical / chemical_plus[family, cell], 1.0, places=6
        )

    def test_variational_flux_has_exact_nonpositive_dissipation(self) -> None:
        rng = np.random.default_rng(31)
        plus = self.rho * (0.5 + 0.08 * rng.normal(size=(2, self.n)))
        minus = self.rho * (0.5 + 0.08 * rng.normal(size=(2, self.n)))
        kwargs = dict(
            backstress_coefficient=1.0, diffusion_coefficient=1.0,
            reference_density_m2=self.rho, density_floor_m2=1.0e8,
        )
        chain, squares = variational_dissipation_J_m2_s(
            plus, minus, self.mobility, self.mu, self.burgers,
            self.dx, **kwargs
        )
        self.assertLess(chain, 0.0)
        self.assertAlmostEqual(chain / squares, 1.0, places=13)

    def test_variational_floor_refinement_and_zero_population_flux(self) -> None:
        plus = np.full((2, self.n), 0.5 * self.rho)
        minus = plus.copy()
        plus[1] = 0.0
        x = np.arange(self.n)
        minus[0] *= 1.0 + 0.02 * np.cos(2.0 * np.pi * x / self.n)
        records = []
        for floor in (1.0e10, 1.0e8, 1.0e6):
            flux_plus, flux_minus = variational_correlation_fluxes(
                plus, minus, self.mobility, self.mu, self.burgers, self.dx,
                backstress_coefficient=1.0, diffusion_coefficient=1.0,
                reference_density_m2=self.rho, density_floor_m2=floor,
            )
            self.assertEqual(float(np.max(np.abs(flux_plus[1]))), 0.0)
            records.append(float(np.max(np.abs(flux_minus))))
        self.assertLess(abs(records[-1] - records[-2]), abs(records[1] - records[0]))

    def test_imex_flux_reconstructs_conservative_positive_energy_decay(self) -> None:
        x = np.arange(self.n)
        plus = np.full((2, self.n), 0.5 * self.rho)
        minus = plus.copy()
        plus[0] *= 1.0 + 0.2 * np.cos(2.0 * np.pi * 11.0 * x / self.n)
        minus[0] *= 1.0 - 0.1 * np.cos(2.0 * np.pi * 11.0 * x / self.n)
        kwargs = dict(
            backstress_coefficient=0.7, diffusion_coefficient=1.1,
            reference_density_m2=self.rho, density_floor_m2=1.0e8,
        )
        dt = 2.0e-5
        flux_plus, flux_minus = variational_correlation_imex_fluxes(
            plus, minus, float(self.mobility[0, 0]), float(self.mu[0]),
            self.burgers, self.dx, dt, **kwargs,
        )
        updated_plus = plus + dt * (
            -(flux_plus - np.roll(flux_plus, 1, axis=1)) / self.dx
        )
        updated_minus = minus + dt * (
            -(flux_minus - np.roll(flux_minus, 1, axis=1)) / self.dx
        )
        self.assertGreaterEqual(float(np.min(updated_plus)), 0.0)
        self.assertGreaterEqual(float(np.min(updated_minus)), 0.0)
        np.testing.assert_allclose(
            np.sum(updated_plus, axis=1), np.sum(plus, axis=1),
            rtol=2.0e-16, atol=2.0,
        )
        old_energy = float(np.mean(variational_correlation_energy_J_m3(
            plus, minus, self.mu, self.burgers, **kwargs
        )))
        new_energy = float(np.mean(variational_correlation_energy_J_m3(
            updated_plus, updated_minus, self.mu, self.burgers, **kwargs
        )))
        self.assertLess(new_energy, old_energy)


if __name__ == "__main__":
    unittest.main()
