from __future__ import annotations

import unittest

from asb_drx.fixtures import SingleGliderDDDParameterization
from asb_drx.recovery import (
    RecoveryBoundaryPoint,
    fit_recovery_law_to_boundary,
    post_peak_density_growth_rate_s_inv,
)


class RecoveryBoundaryDesignTests(unittest.TestCase):
    def setUp(self) -> None:
        fixture = SingleGliderDDDParameterization()
        self.flow = fixture.law()
        self.K = fixture.spatial_parameters().forest_storage_per_plastic_strain_m2
        self.first = RecoveryBoundaryPoint(850.0, 450.0, 2.0)
        self.second = RecoveryBoundaryPoint(1050.0, 45000.0, 2.0)
        self.fit = fit_recovery_law_to_boundary(
            self.flow, self.K, self.first, self.second,
            reference_temperature_K=950.0,
        )

    def test_two_governing_equation_constraints_close_exactly(self) -> None:
        self.assertLess(self.fit.maximum_log_closure_error, 1.0e-12)
        for point in (self.first, self.second):
            self.assertAlmostEqual(
                post_peak_density_growth_rate_s_inv(
                    self.flow, self.fit.law, self.K, point
                ),
                0.0,
                delta=2.0e-14,
            )

    def test_same_parameter_set_separates_rates_without_flow_retuning(self) -> None:
        stable = RecoveryBoundaryPoint(950.0, 450.0, 2.0)
        unstable = RecoveryBoundaryPoint(950.0, 45000.0, 2.0)
        self.assertLess(
            post_peak_density_growth_rate_s_inv(self.flow, self.fit.law, self.K, stable),
            0.0,
        )
        self.assertGreater(
            post_peak_density_growth_rate_s_inv(self.flow, self.fit.law, self.K, unstable),
            0.0,
        )

    def test_pre_peak_anchor_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "post-peak"):
            fit_recovery_law_to_boundary(
                self.flow,
                self.K,
                RecoveryBoundaryPoint(850.0, 450.0, 0.5),
                self.second,
                reference_temperature_K=950.0,
            )


if __name__ == "__main__":
    unittest.main()
