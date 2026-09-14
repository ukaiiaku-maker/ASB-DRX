import unittest

import numpy as np

from full_model.production.sibm_geometry import (
    displace_pair_boundary,
    pinned_cap_energy,
    pinned_cap_geometry,
    validate_post_seed_pair,
)


class SIBMGeometryTest(unittest.TestCase):
    def bicrystal(self, n=192, width=2.0e-7):
        dx = 10e-6 / n
        x = (np.arange(n) - n / 2 + 0.5)[:, None] * dx
        child = np.broadcast_to(0.5 * (1 + np.tanh(x / (np.sqrt(2) * width))), (n, n))
        eta = np.zeros((n, n, 2))
        eta[..., 0] = 1 - child
        eta[..., 1] = child
        return eta, dx

    def test_zero_amplitude_is_bitwise_identity(self):
        eta, dx = self.bicrystal()
        seed = displace_pair_boundary(
            eta, parent=0, child=1, spacing_m=dx, interface_width_m=2e-7,
            centre_index=(96, 96), advance_direction_index=(-1, 0),
            amplitude_m=0.0, half_chord_m=1.5e-6,
            active_window_radius_m=3e-6)
        np.testing.assert_array_equal(seed.eta, eta)

    def test_boundary_displacement_preserves_pair_and_simplex(self):
        eta, dx = self.bicrystal()
        seed = displace_pair_boundary(
            eta, parent=0, child=1, spacing_m=dx, interface_width_m=2e-7,
            centre_index=(96, 96), advance_direction_index=(-1, 0),
            amplitude_m=5e-7, half_chord_m=1.2e-6,
            active_window_radius_m=3e-6)
        report = validate_post_seed_pair(
            seed.eta, parent=0, child=1, spacing_m=dx, interface_width_m=2e-7,
            centre_index=(96, 96), advance_direction_index=(-1, 0),
            seed_amplitude_m=5e-7, half_chord_m=1.2e-6,
            active_window_radius_m=3e-6)
        self.assertTrue(report.valid, report.reasons)
        self.assertLessEqual(seed.maximum_simplex_error, 1e-15)
        self.assertEqual(seed.maximum_nonpair_change, 0.0)

    def test_invalid_small_parent_is_rejected(self):
        eta, dx = self.bicrystal(n=128)
        eta[:55, :, 0] = 0.4
        eta[:55, :, 1] = 0.6
        eta /= eta.sum(axis=2, keepdims=True)
        report = validate_post_seed_pair(
            eta, parent=0, child=1, spacing_m=dx, interface_width_m=3.16e-7,
            centre_index=(64, 64), advance_direction_index=(-1, 0),
            seed_amplitude_m=7.5e-7, half_chord_m=1.5e-6,
            active_window_radius_m=3e-6)
        self.assertFalse(report.valid)
        self.assertIn("parent core inradius", report.reasons)

    def test_pinned_cap_derivative_matches_finite_difference(self):
        kwargs = dict(boundary_energy_J_m2=0.5, stored_pressure_Pa=8e5,
                      compatibility_pressure_Pa=1e5, drag_pressure_Pa=5e4,
                      represented_thickness_m=1e-6)
        a, b, h = 6e-7, 1.5e-6, 1e-10
        exact = pinned_cap_energy(a, b, **kwargs)["d_total_energy_da_J_m"]
        fd = (pinned_cap_energy(a+h, b, **kwargs)["total_energy_J"]
              - pinned_cap_energy(a-h, b, **kwargs)["total_energy_J"]) / (2*h)
        self.assertAlmostEqual(exact, fd, delta=abs(exact)*2e-7)

    def test_geometry_specific_critical_pressure_changes_sign_only(self):
        a, b, gamma = 6e-7, 1.5e-6, 0.5
        base = pinned_cap_energy(a, b, boundary_energy_J_m2=gamma,
                                 stored_pressure_Pa=0.0)
        critical = gamma * base["d_arc_length_da"] / base["d_swept_area_da_m"]
        below = pinned_cap_energy(a, b, boundary_energy_J_m2=gamma,
                                  stored_pressure_Pa=0.8*critical)
        above = pinned_cap_energy(a, b, boundary_energy_J_m2=gamma,
                                  stored_pressure_Pa=1.2*critical)
        self.assertGreater(below["d_total_energy_da_J_m"], 0.0)
        self.assertLess(above["d_total_energy_da_J_m"], 0.0)
        self.assertGreater(pinned_cap_geometry(a, b)["arc_length_m"], 2*b)


if __name__ == "__main__":
    unittest.main()
