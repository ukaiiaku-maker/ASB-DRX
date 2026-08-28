from __future__ import annotations

import math
import unittest

from asb_drx.boundary import AnalyticalPeakBoundary
from asb_drx.fixtures import SingleGliderDDDParameterization


class AnalyticalPeakBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = SingleGliderDDDParameterization()
        self.law = self.fixture.law()
        self.boundary = AnalyticalPeakBoundary(self.law)

    def test_boundary_is_exact_analytical_peak(self) -> None:
        peak = self.law.peak(950.0, 4.5)
        point = self.boundary.classify(peak.density_m2, 950.0, 4.5)
        self.assertEqual(point.branch, "analytical_peak")
        self.assertEqual(point.peak_density_m2, peak.density_m2)
        self.assertEqual(point.peak_strength_Pa, peak.macroscopic_strength_Pa)

    def test_branch_labels_are_only_relative_to_independent_peak(self) -> None:
        peak = self.law.peak(1000.0, 4.5)
        below = self.boundary.classify(0.5 * peak.density_m2, 1000.0, 4.5)
        above = self.boundary.classify(2.0 * peak.density_m2, 1000.0, 4.5)
        self.assertEqual(below.branch, "independent_rising_branch")
        self.assertEqual(above.branch, "post_peak_collective_candidate")
        self.assertEqual(below.density_ratio, 0.5)
        self.assertEqual(above.density_ratio, 2.0)

    def test_peak_density_has_exact_rate_power_scaling(self) -> None:
        rate_a = 4.5
        rate_b = 450.0
        rho_a = self.law.peak(950.0, rate_a).density_m2
        rho_b = self.law.peak(950.0, rate_b).density_m2
        expected = (rate_b / rate_a) ** (2.0 / self.law.density_exponent_p)
        self.assertTrue(math.isclose(rho_b / rho_a, expected, rel_tol=2.0e-14))

    def test_surface_is_rate_major_and_complete(self) -> None:
        points = self.boundary.surface((850.0, 1050.0), (4.5, 450.0, 45000.0))
        self.assertEqual(len(points), 6)
        self.assertTrue(all(point.branch == "analytical_peak" for point in points))
        self.assertEqual(
            [(point.shear_rate_s_inv, point.temperature_K) for point in points],
            [(4.5, 850.0), (4.5, 1050.0), (450.0, 850.0),
             (450.0, 1050.0), (45000.0, 850.0), (45000.0, 1050.0)],
        )

    def test_ddd_campaign_upper_density_is_post_peak_at_source_rate(self) -> None:
        for temperature in (850.0, 900.0, 950.0, 1000.0, 1050.0):
            point = self.boundary.classify(
                self.fixture.source_density_range_m2[1],
                temperature,
                self.fixture.source_strain_rate_s_inv,
            )
            self.assertEqual(point.branch, "post_peak_collective_candidate")
            self.assertGreater(point.density_ratio, 3.0)

    def test_invalid_boundary_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.boundary.classify(0.0, 1000.0, 4.5)
        with self.assertRaises(ValueError):
            self.boundary.surface((), (4.5,))


if __name__ == "__main__":
    unittest.main()
