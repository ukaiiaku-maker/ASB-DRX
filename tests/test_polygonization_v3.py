from __future__ import annotations

import unittest

import numpy as np

from asb_drx.arrhenius_v3 import (
    ArrheniusMechanism,
    BoundedActivationEntropy,
    ExpFloorEnthalpy,
)
from asb_drx.polygonization_v3 import (
    PolygonizationParameters,
    PolygonizationState,
    frank_bilby_misorientation_rad,
    frank_bilby_residual,
    polygonization_step,
)


EV_J = 1.602176634e-19


def mechanism(barrier_eV: float, entropy_kB: float) -> ArrheniusMechanism:
    return ArrheniusMechanism(
        ExpFloorEnthalpy(
            barrier_eV * EV_J, 1.0e9, 900.0, 0.3, 1.5, 2.0
        ),
        BoundedActivationEntropy(reference_kB=entropy_kB),
        1.0e8,
        validity_temperature_K=(500.0, 1500.0),
        validity_stress_Pa=(0.0, 2.0e9),
    )


class PolygonizationV3Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = PolygonizationParameters(
            capture=mechanism(0.45, 0.5),
            climb_annihilation=mechanism(0.8, -0.5),
            wall_ordering=mechanism(0.6, 0.0),
            burgers_m=2.5e-10,
            wall_width_m=0.5e-6,
            line_energy_J_m=2.5e-9,
            capture_driving_stress_Pa=0.4e9,
        )
        self.state = PolygonizationState(
            np.asarray([4.0e14, 3.0e14]),
            np.asarray([2.0e14, 2.5e14]),
            np.asarray([1.0e13, 1.0e13]),
            np.asarray([0.5e13, 1.0e13]),
        )

    def test_capture_and_climb_close_line_and_signed_ledgers(self) -> None:
        advanced, ledger = polygonization_step(
            self.state, 900.0, 1.0e-7, self.parameters
        )
        self.assertGreater(ledger.captured_m2, 0.0)
        self.assertGreater(ledger.annihilated_m2, 0.0)
        self.assertLess(
            abs(ledger.line_balance_residual_m2),
            2.0e-15 * ledger.line_content_before_m2,
        )
        self.assertLess(
            ledger.maximum_signed_burgers_residual_m2,
            2.0e-15 * ledger.line_content_before_m2,
        )
        self.assertGreater(ledger.released_line_energy_J_m3, 0.0)
        self.assertGreaterEqual(float(np.min(advanced.mobile_plus_m2)), 0.0)

    def test_frank_bilby_angle_is_derived_from_wall_inventory(self) -> None:
        theta = frank_bilby_misorientation_rad(self.state, self.parameters)
        self.assertGreater(theta[0], 0.0)
        self.assertEqual(theta[1], 0.0)
        np.testing.assert_allclose(
            frank_bilby_residual(self.state, self.parameters), 0.0,
            rtol=0.0, atol=2.0e-17,
        )

    def test_balanced_wall_does_not_invent_orientation_or_maturity(self) -> None:
        balanced = PolygonizationState(
            self.state.mobile_plus_m2, self.state.mobile_minus_m2,
            np.full(2, 2.0e14), np.full(2, 2.0e14),
        )
        parameters = PolygonizationParameters(
            self.parameters.capture, self.parameters.climb_annihilation,
            self.parameters.wall_ordering, self.parameters.burgers_m,
            self.parameters.wall_width_m, self.parameters.line_energy_J_m,
            capture_driving_stress_Pa=0.0,
        )
        # With no time-integrated capture in the initial observation, balanced
        # wall content has no Frank--Bilby angle or maturity by label alone.
        np.testing.assert_array_equal(
            frank_bilby_misorientation_rad(balanced, parameters), np.zeros(2)
        )
        self.assertEqual(balanced.wall_maturity, 0.0)

    def test_distinct_arrhenius_channels_have_temperature_dependence(self) -> None:
        low, _ = polygonization_step(
            self.state, 700.0, 1.0e-4, self.parameters
        )
        high, _ = polygonization_step(
            self.state, 1100.0, 1.0e-4, self.parameters
        )
        self.assertGreater(high.wall_maturity, low.wall_maturity)
        self.assertFalse(np.array_equal(high.mobile_plus_m2, low.mobile_plus_m2))
        self.assertEqual(high.physical_grain_count, 1)
        self.assertFalse(hasattr(high, "grain_labels"))


if __name__ == "__main__":
    unittest.main()
