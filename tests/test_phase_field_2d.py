from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from asb_drx.thermodynamics import (
    CircularNucleusLimit,
    GrainEnergyParameters,
    PhaseFieldState2D,
    advance_phase_field_2d,
    diffuse_circle_2d,
    equivalent_support_radius_m,
    save_phase_field_checkpoint,
    load_phase_field_checkpoint,
)


class PhaseField2DTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = GrainEnergyParameters(2.0e6, 2.0e-6, 2.0e5, 5.0e-7)
        self.interface_length_m = (self.parameters.gradient_coefficient_J_m / self.parameters.well_height_J_m3) ** 0.5
        self.gb_energy_J_m2 = (
            2.0
            * self.parameters.gradient_coefficient_J_m
            * self.parameters.well_height_J_m3
        ) ** 0.5 / 6.0
        self.nucleus = CircularNucleusLimit(
            self.gb_energy_J_m2,
            self.parameters.bulk_driving_J_m3,
            1.0,
        )

    def _evolve_radius(self, grid_points: int, radius_ratio: float, dt_s: float, steps: int) -> tuple[float, float]:
        domain_m = 1.6e-5
        dx_m = domain_m / grid_points
        eta = diffuse_circle_2d(
            grid_points,
            dx_m,
            radius_ratio * self.nucleus.critical_radius_m,
            self.interface_length_m,
        )
        initial = equivalent_support_radius_m(eta, dx_m)
        final_state = advance_phase_field_2d(
            PhaseFieldState2D(eta, 0.0, 0), dx_m, dt_s, steps, self.parameters
        )
        return initial, equivalent_support_radius_m(final_state.eta, dx_m)

    def test_diffuse_nucleus_shrinks_below_and_grows_above_critical_radius(self) -> None:
        sub_initial, sub_final = self._evolve_radius(128, 0.72, 1.0e-4, 200)
        super_initial, super_final = self._evolve_radius(128, 1.35, 1.0e-4, 200)
        self.assertLess(sub_final, sub_initial)
        self.assertGreater(super_final, super_initial)

    def test_complete_current_state_restarts_exactly(self) -> None:
        grid_points = 64
        dx_m = 1.6e-5 / grid_points
        eta = diffuse_circle_2d(
            grid_points,
            dx_m,
            1.35 * self.nucleus.critical_radius_m,
            self.interface_length_m,
        )
        initial = PhaseFieldState2D(eta, 0.0, 0)
        continuous = advance_phase_field_2d(initial, dx_m, 1.0e-4, 40, self.parameters)
        first = advance_phase_field_2d(initial, dx_m, 1.0e-4, 20, self.parameters)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "state.npz"
            save_phase_field_checkpoint(checkpoint, first)
            restored = load_phase_field_checkpoint(checkpoint)
        segmented = advance_phase_field_2d(restored, dx_m, 1.0e-4, 20, self.parameters)
        self.assertTrue(np.array_equal(continuous.eta, segmented.eta))
        self.assertEqual(continuous.time_s, segmented.time_s)
        self.assertEqual(continuous.accepted_steps, segmented.accepted_steps)

    def test_final_grid_and_timestep_refinements_are_below_five_percent(self) -> None:
        grid_changes = []
        for grid_points in (64, 96, 128):
            initial, final = self._evolve_radius(grid_points, 1.35, 1.0e-4, 100)
            grid_changes.append(final - initial)
        grid_relative = abs(grid_changes[-1] - grid_changes[-2]) / abs(grid_changes[-1])
        self.assertLess(grid_relative, 0.05)

        time_changes = []
        for dt_s, steps in ((2.0e-4, 50), (1.0e-4, 100), (5.0e-5, 200)):
            initial, final = self._evolve_radius(128, 1.35, dt_s, steps)
            time_changes.append(final - initial)
        time_relative = abs(time_changes[-1] - time_changes[-2]) / abs(time_changes[-1])
        self.assertLess(time_relative, 0.05)


if __name__ == "__main__":
    unittest.main()
