from __future__ import annotations

import unittest

from asb_drx.boundary_campaign import BoundarySpatialCase
from asb_drx.fixtures import SingleGliderDDDParameterization
from asb_drx.local_mechanism import (
    classify_local_mechanism_trace,
    matched_local_isothermal_trace,
    run_local_mechanism_trace,
    run_matched_local_strain_pair,
)
from asb_drx.localization import LocalizationCriteria
from asb_drx.mechanism_ladder import MechanismCase
from asb_drx.spatial_coupled import SpatialMechanismControls


class LocalMechanismTraceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = SingleGliderDDDParameterization()
        self.initial, self.metadata = BoundarySpatialCase(
            950.0, 45000.0, 1.0
        ).build_local_state(16, self.fixture)
        self.case = MechanismCase(
            "local_boundary_smoke",
            45000.0,
            SpatialMechanismControls(True, True),
        )

    def test_trace_contains_local_and_mean_stress(self) -> None:
        trace = run_local_mechanism_trace(
            self.initial,
            self.case,
            self.metadata["dx_m"],
            2.0e-8,
            3,
            self.fixture.law(),
            self.fixture.spatial_parameters(),
        )
        self.assertEqual(trace.local_stress_x_Pa.shape, (3, 16, 16))
        self.assertEqual(trace.mean_stress_Pa.shape, (3,))
        self.assertEqual(trace.plastic_rate_s_inv.shape, (3, 16, 16))

    def test_matched_control_and_classifier_execute_without_switching_mechanics(self) -> None:
        trace = run_local_mechanism_trace(
            self.initial, self.case, self.metadata["dx_m"], 2.0e-8, 3,
            self.fixture.law(), self.fixture.spatial_parameters(),
        )
        control = matched_local_isothermal_trace(
            self.initial, self.case, self.metadata["dx_m"], 2.0e-8, 3,
            self.fixture.law(), self.fixture.spatial_parameters(),
        )
        decision = classify_local_mechanism_trace(
            trace,
            control,
            self.metadata["dx_m"],
            self.metadata["interface_width_m"],
            LocalizationCriteria(0.4, 20.0, 0.1, 3.0, 3, 0.05),
        )
        self.assertFalse(control.case.controls.evolve_temperature)
        self.assertEqual(trace.case.applied_shear_rate_s_inv, control.case.applied_shear_rate_s_inv)
        self.assertIsInstance(decision.localized, bool)

    def test_strain_targeted_pair_has_identical_time_grid(self) -> None:
        target = 2.0e-3
        trace, control = run_matched_local_strain_pair(
            self.initial, self.case, self.metadata["dx_m"], 2.0e-8, target,
            self.fixture.law(), self.fixture.spatial_parameters(),
        )
        self.assertEqual(len(trace.steps), len(control.steps))
        self.assertEqual(
            [item.time_s for item in trace.states],
            [item.time_s for item in control.states],
        )
        self.assertAlmostEqual(
            abs(trace.states[-1].applied_shear - self.initial.applied_shear),
            target,
        )

    def test_strain_cadence_limits_retention_and_preserves_step_statistics(self) -> None:
        trace, control = run_matched_local_strain_pair(
            self.initial, self.case, self.metadata["dx_m"], 2.0e-8, 3.0e-3,
            self.fixture.law(), self.fixture.spatial_parameters(),
            retention_strain_increment=1.0e-3,
        )
        self.assertLessEqual(len(trace.states), 4)
        self.assertEqual(len(trace.states), len(control.states))
        self.assertGreaterEqual(trace.statistics.accepted_steps, len(trace.states))
        self.assertEqual(trace.statistics, control.statistics)
        self.assertAlmostEqual(
            abs(trace.states[-1].applied_shear - self.initial.applied_shear),
            3.0e-3,
        )

    def test_strain_pair_reports_numerically_unresolved_step_limit(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "maximum_accepted_steps"):
            run_matched_local_strain_pair(
                self.initial, self.case, self.metadata["dx_m"], 2.0e-8, 3.0e-3,
                self.fixture.law(), self.fixture.spatial_parameters(),
                maximum_accepted_steps=1,
            )


if __name__ == "__main__":
    unittest.main()
