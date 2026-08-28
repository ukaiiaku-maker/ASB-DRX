from __future__ import annotations

from pathlib import Path
import math
import tempfile
import unittest

import numpy as np

from asb_drx.grains import (
    GrainCriteria,
    GrainRecord,
    GrainTrackerState,
    update_grain_tracker,
)
from asb_drx.multi_order import (
    BinaryCircularLimit,
    MultiOrderParameters,
    MultiOrderState,
    advance_multi_order,
    binary_boundary_energy_J_m2,
    diffuse_binary_circle,
    energy_checked_multi_order_step,
    equivalent_child_radius_m,
    load_multi_order_checkpoint,
    multi_order_chemical_potential_J_m3,
    multi_order_free_energy_J_m,
    save_multi_order_checkpoint,
)


class MultiOrderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = MultiOrderParameters(2.0e6, 1.0e-6, 5.0e-7, (0.0, -2.0e5))
        self.interface_length_m = 2.0 * math.sqrt(
            self.parameters.gradient_coefficient_J_m
            / self.parameters.pair_penalty_J_m3
        )
        self.limit = BinaryCircularLimit(
            binary_boundary_energy_J_m2(self.parameters), 2.0e5, 1.0
        )

    def test_projected_variational_derivative_matches_simplex_perturbation(self) -> None:
        points = 16
        dx_m = 1.0e-6
        x = np.linspace(0.0, 2.0 * math.pi, points, endpoint=False)
        pattern = 0.05 * np.sin(x)[None, :] + 0.03 * np.cos(x)[:, None]
        fields = np.stack((0.45 + pattern, 0.55 - pattern))
        direction_pattern = 0.2 + 0.1 * np.sin(x)[None, :]
        direction = np.stack((direction_pattern, -direction_pattern))
        epsilon = 1.0e-7
        plus = multi_order_free_energy_J_m(
            fields + epsilon * direction, dx_m, self.parameters
        )
        minus = multi_order_free_energy_J_m(
            fields - epsilon * direction, dx_m, self.parameters
        )
        finite_difference = (plus - minus) / (2.0 * epsilon)
        chemical = multi_order_chemical_potential_J_m3(fields, dx_m, self.parameters)
        analytical = float(dx_m**2 * np.sum(chemical * direction))
        self.assertAlmostEqual(finite_difference / analytical, 1.0, places=6)

    def test_accepted_step_preserves_simplex_and_decreases_energy(self) -> None:
        dx_m = 1.6e-5 / 64
        fields = diffuse_binary_circle(
            64, dx_m, 1.35 * self.limit.critical_radius_m, self.interface_length_m
        )
        old_energy = multi_order_free_energy_J_m(fields, dx_m, self.parameters)
        accepted = energy_checked_multi_order_step(fields, dx_m, 1.0e-4, self.parameters)
        self.assertTrue(np.allclose(np.sum(accepted.eta_fields, axis=0), 1.0))
        self.assertTrue(np.all((accepted.eta_fields >= 0.0) & (accepted.eta_fields <= 1.0)))
        self.assertLessEqual(accepted.free_energy_J_m, old_energy)

    def test_pure_parent_does_not_create_an_allocated_child(self) -> None:
        fields = np.zeros((2, 16, 16), dtype=float)
        fields[0] = 1.0
        accepted = energy_checked_multi_order_step(
            fields, 1.0e-6, 1.0e-4, self.parameters
        )
        self.assertTrue(np.array_equal(accepted.eta_fields, fields))

    def test_binary_nucleus_shrinks_below_and_grows_above_critical_radius(self) -> None:
        changes = []
        for ratio in (0.72, 1.35):
            dx_m = 1.6e-5 / 128
            fields = diffuse_binary_circle(
                128,
                dx_m,
                ratio * self.limit.critical_radius_m,
                self.interface_length_m,
            )
            initial = equivalent_child_radius_m(fields, dx_m)
            final = advance_multi_order(
                MultiOrderState(fields, 0.0, 0), dx_m, 1.0e-4, 200, self.parameters
            )
            changes.append(equivalent_child_radius_m(final.eta_fields, dx_m) - initial)
        self.assertLess(changes[0], 0.0)
        self.assertGreater(changes[1], 0.0)

    def test_label_permutation_preserves_energy_and_permuted_dynamics(self) -> None:
        dx_m = 1.6e-5 / 64
        fields = diffuse_binary_circle(
            64, dx_m, 1.35 * self.limit.critical_radius_m, self.interface_length_m
        )
        original = energy_checked_multi_order_step(fields, dx_m, 1.0e-4, self.parameters)
        swapped_parameters = MultiOrderParameters(
            self.parameters.pair_penalty_J_m3,
            self.parameters.gradient_coefficient_J_m,
            self.parameters.mobility_m3_J_s,
            tuple(reversed(self.parameters.bulk_energy_J_m3)),
        )
        swapped = energy_checked_multi_order_step(
            fields[::-1], dx_m, 1.0e-4, swapped_parameters
        )
        self.assertEqual(original.free_energy_J_m, swapped.free_energy_J_m)
        self.assertTrue(np.array_equal(original.eta_fields, swapped.eta_fields[::-1]))

    def test_complete_current_state_restarts_exactly(self) -> None:
        dx_m = 1.6e-5 / 64
        fields = diffuse_binary_circle(
            64, dx_m, 1.35 * self.limit.critical_radius_m, self.interface_length_m
        )
        initial = MultiOrderState(fields, 0.0, 0)
        continuous = advance_multi_order(initial, dx_m, 1.0e-4, 40, self.parameters)
        first = advance_multi_order(initial, dx_m, 1.0e-4, 20, self.parameters)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "multi-order.npz"
            save_multi_order_checkpoint(path, first)
            restored = load_multi_order_checkpoint(path)
        segmented = advance_multi_order(restored, dx_m, 1.0e-4, 20, self.parameters)
        self.assertTrue(np.array_equal(continuous.eta_fields, segmented.eta_fields))
        self.assertEqual(continuous.time_s, segmented.time_s)
        self.assertEqual(continuous.accepted_steps, segmented.accepted_steps)

    def test_evolved_fields_feed_tracker_without_creating_labels(self) -> None:
        dx_m = 1.6e-5 / 64
        fields = diffuse_binary_circle(
            64, dx_m, 1.35 * self.limit.critical_radius_m, self.interface_length_m
        )
        phase_state = MultiOrderState(fields, 0.0, 0)
        tracker = GrainTrackerState(
            (
                GrainRecord(0, 0.0, None, "root-0", 0.0),
                GrainRecord(
                    1, math.radians(12.0), 0, "root-0/child-1", 0.0,
                    source_embryo_id="verified-fixture-embryo",
                    embryo_gate_passed=True,
                ),
            )
        )
        criteria = GrainCriteria(
            0.8,
            25.0 * dx_m**2,
            3,
            2,
            math.radians(5.0),
            4,
        )
        metrics = None
        for _ in range(3):
            phase_state = advance_multi_order(
                phase_state, dx_m, 1.0e-4, 10, self.parameters
            )
            tracker, metrics = update_grain_tracker(
                phase_state.eta_fields, tracker, phase_state.time_s, dx_m, criteria
            )
        self.assertEqual(phase_state.eta_fields.shape[0], 2)
        self.assertEqual(metrics.allocated_labels, 2)
        self.assertEqual(metrics.physical_grains, 2)
        self.assertEqual(metrics.recrystallized_grains, 1)


if __name__ == "__main__":
    unittest.main()
