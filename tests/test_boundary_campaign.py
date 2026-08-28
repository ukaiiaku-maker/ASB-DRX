from __future__ import annotations

import math
import unittest

import numpy as np

from asb_drx.antiplane import solve_periodic_antiplane
from asb_drx.boundary_campaign import BoundarySpatialCase
from asb_drx.fixtures import SingleGliderDDDParameterization


class BoundarySpatialCaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = SingleGliderDDDParameterization()
        self.case = BoundarySpatialCase(950.0, 45000.0, 1.0)

    def test_case_is_anchored_to_exact_analytical_peak(self) -> None:
        state, metadata = self.case.build_state(16, self.fixture)
        peak = self.fixture.law().peak(950.0, 45000.0)
        self.assertEqual(metadata["branch"], "analytical_peak")
        self.assertEqual(metadata["nominal_density_m2"], peak.density_m2)
        self.assertTrue(
            math.isclose(state.stress_Pa, peak.macroscopic_strength_Pa, rel_tol=1.0e-15)
        )

    def test_nominal_peak_closes_applied_rate(self) -> None:
        _, metadata = self.case.build_state(16, self.fixture)
        rate = self.fixture.law().shear_rate_s_inv(
            metadata["initial_stress_Pa"] / self.fixture.law().taylor_ratio(metadata["nominal_density_m2"]),
            metadata["nominal_density_m2"],
            self.case.temperature_K,
        )
        self.assertTrue(math.isclose(rate, self.case.shear_rate_s_inv, rel_tol=2.0e-13))

    def test_local_state_recovers_uniform_peak_stress(self) -> None:
        state, metadata = self.case.build_local_state(16, self.fixture)
        equilibrium = solve_periodic_antiplane(
            state.applied_shear,
            state.plastic_shear,
            self.fixture.spatial_parameters().shear_modulus_Pa,
            metadata["dx_m"],
        )
        self.assertTrue(
            np.allclose(
                equilibrium.stress_x_Pa,
                metadata["initial_stress_Pa"],
                rtol=2.0e-16,
                atol=1.0e-7,
            )
        )

    def test_state_is_resolved_and_admissible_on_both_smoke_grids(self) -> None:
        for points in (16, 32):
            state, metadata = self.case.build_state(points, self.fixture)
            self.assertTrue(np.all(state.forest_density_m2 > 0.0))
            self.assertTrue(np.allclose(np.sum(state.eta_fields, axis=0), 1.0))
            self.assertGreater(metadata["interface_width_m"] / metadata["dx_m"], 1.0)
            self.assertGreater(metadata["nucleus_radius_m"], metadata["critical_radius_m"])

    def test_invalid_case_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            BoundarySpatialCase(950.0, 45000.0, 0.0)
        with self.assertRaises(ValueError):
            self.case.build_state(4, self.fixture)


if __name__ == "__main__":
    unittest.main()
