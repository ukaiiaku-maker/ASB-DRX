from __future__ import annotations

import unittest

import numpy as np

from asb_drx.cdd_flux_v3 import (
    correlation_rhs,
    discrete_mode_growth_rates_s_inv,
    logarithmic_mean,
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


if __name__ == "__main__":
    unittest.main()
