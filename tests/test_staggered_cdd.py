from __future__ import annotations

import unittest

import numpy as np

from asb_drx.cdd_flux_v3 import staggered_correlation_fluxes
from asb_drx.staggered_cdd import (
    StaggeredSignedState,
    advance_staggered_flux,
    advance_staggered_flux_with_ledger,
    cell_centered_slip,
    face_traction_from_cells,
    nye_compatibility_residual_m_inv,
    signed_face_orowan_rate_s_inv,
    transfer_mobile_to_stationary,
    work_conjugacy_residual_J_m3,
)


class StaggeredCDDTests(unittest.TestCase):
    def setUp(self) -> None:
        self.n = 64
        self.domain = 16.0e-6
        self.dx = self.domain / self.n
        self.b = 2.86e-10
        plus = np.full((2, self.n), 2.5e14)
        self.state = StaggeredSignedState(
            plus, plus.copy(), np.zeros_like(plus), self.b, self.dx
        )

    def test_face_orowan_update_preserves_nye_compatibility_exactly(self) -> None:
        x = np.arange(self.n)
        flux = 1.0e8 * np.sin(2.0 * np.pi * 5.0 * x / self.n)
        plus_flux = np.repeat(flux[None, :], 2, axis=0)
        minus_flux = -plus_flux
        advanced = advance_staggered_flux(
            self.state, plus_flux, minus_flux, 1.0e-8
        )
        residual = nye_compatibility_residual_m_inv(advanced)
        scale = self.b * float(np.max(
            advanced.mobile_plus_m2 + advanced.mobile_minus_m2
        ))
        self.assertLess(float(np.max(np.abs(residual))), 3.0e-15 * scale)
        np.testing.assert_allclose(
            np.sum(advanced.mobile_plus_m2, axis=1),
            np.sum(self.state.mobile_plus_m2, axis=1), rtol=0.0, atol=2.0,
        )
        np.testing.assert_allclose(
            np.sum(advanced.mobile_minus_m2, axis=1),
            np.sum(self.state.mobile_minus_m2, axis=1), rtol=0.0, atol=2.0,
        )

    def test_logarithmic_correlation_flux_advances_positive_compatible_state(self) -> None:
        x = np.arange(self.n)
        total_wave = 1.0 + 0.01 * np.cos(2.0 * np.pi * 4.0 * x / self.n)
        plus = self.state.mobile_plus_m2 * total_wave[None, :]
        minus = self.state.mobile_minus_m2 * total_wave[None, :]
        state = StaggeredSignedState(
            plus, minus, self.state.face_slip, self.b, self.dx
        )
        plus_flux, minus_flux = staggered_correlation_fluxes(
            plus, minus, np.full_like(plus, 2.0e-13),
            np.full(self.n, 40.0e9), self.b, self.dx,
            backstress_coefficient=1.0, diffusion_coefficient=1.0,
            face_density="logarithmic",
        )
        before_variance = float(np.var(plus + minus))
        advanced = advance_staggered_flux(
            state, plus_flux, minus_flux, 1.0e-8
        )
        self.assertGreaterEqual(float(np.min(advanced.mobile_plus_m2)), 0.0)
        self.assertLess(float(np.var(
            advanced.mobile_plus_m2 + advanced.mobile_minus_m2
        )), before_variance)
        self.assertLess(
            float(np.max(np.abs(nye_compatibility_residual_m_inv(advanced)))),
            1.0e-12,
        )

    def test_cell_reconstruction_is_constant_exact_and_second_order(self) -> None:
        constant = np.full((2, self.n), 0.3)
        np.testing.assert_array_equal(cell_centered_slip(constant), constant)
        x_face = (np.arange(self.n) + 0.5) * self.dx
        face = np.sin(2.0 * np.pi * x_face / self.domain)[None, :]
        cell = cell_centered_slip(face)[0]
        exact = np.sin(2.0 * np.pi * np.arange(self.n) * self.dx / self.domain)
        self.assertLess(float(np.max(np.abs(cell - exact))), 1.3e-3)

    def test_zero_face_flux_preserves_state(self) -> None:
        zero = np.zeros_like(self.state.mobile_plus_m2)
        advanced = advance_staggered_flux(self.state, zero, zero, 1.0)
        np.testing.assert_array_equal(advanced.mobile_plus_m2, self.state.mobile_plus_m2)
        np.testing.assert_array_equal(advanced.mobile_minus_m2, self.state.mobile_minus_m2)
        np.testing.assert_array_equal(advanced.face_slip, self.state.face_slip)
        np.testing.assert_array_equal(
            signed_face_orowan_rate_s_inv(zero, zero, self.b), zero
        )

    def test_locked_and_wall_transfer_preserves_total_nye_inventory(self) -> None:
        x = np.arange(self.n)
        plus = self.state.mobile_plus_m2 * (
            1.0 + 0.02 * np.sin(2.0 * np.pi * x / self.n)
        )[None, :]
        initial = StaggeredSignedState(
            plus, self.state.mobile_minus_m2, self.state.face_slip, self.b, self.dx
        )
        transferred = transfer_mobile_to_stationary(initial, 0.2, 0.35)
        np.testing.assert_allclose(
            transferred.total_signed_density_m2,
            initial.total_signed_density_m2, rtol=0.0, atol=0.0625,
        )
        np.testing.assert_allclose(
            nye_compatibility_residual_m_inv(transferred),
            nye_compatibility_residual_m_inv(initial), rtol=0.0, atol=2.0e-11,
        )

    def test_flux_ledger_reports_zero_clipping_in_accepted_step(self) -> None:
        zero = np.zeros_like(self.state.mobile_plus_m2)
        advanced, ledger = advance_staggered_flux_with_ledger(
            self.state, zero, zero, 1.0
        )
        self.assertEqual(ledger.clipping_added_m_inv, 0.0)
        self.assertEqual(ledger.signed_clipping_added_m_inv, 0.0)
        self.assertEqual(ledger.balance_residual_m_inv, 0.0)
        np.testing.assert_array_equal(
            advanced.total_signed_density_m2, self.state.total_signed_density_m2
        )

    def test_cell_face_projection_is_exactly_work_conjugate(self) -> None:
        rng = np.random.default_rng(92)
        traction = rng.normal(size=(2, self.n)) * 200.0e6
        increment = rng.normal(size=(2, self.n)) * 1.0e-4
        residual = work_conjugacy_residual_J_m3(traction, increment)
        scale = float(np.sum(np.abs(
            face_traction_from_cells(traction) * increment
        )))
        self.assertLessEqual(abs(residual), 3.0e-16 * scale)


if __name__ == "__main__":
    unittest.main()
