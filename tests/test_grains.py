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
    crystallographic_misorientation_rad,
    load_grain_tracker,
    periodic_component_count,
    save_grain_tracker,
    update_grain_tracker,
)


class GrainTrackerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dx_m = 1.0e-6
        self.criteria = GrainCriteria(
            purity_threshold=0.8,
            minimum_area_m2=9.0 * self.dx_m**2,
            minimum_persistence_steps=3,
            retirement_grace_steps=2,
            minimum_misorientation_rad=math.radians(5.0),
            symmetry_order=4,
        )

    @staticmethod
    def _records(child_orientation_rad: float = math.radians(12.0)) -> tuple[GrainRecord, ...]:
        return (
            GrainRecord(0, 0.0, None, "root-0", 0.0),
            GrainRecord(
                1, child_orientation_rad, 0, "root-0/child-1", 1.0,
                source_embryo_id="embryo-1", embryo_gate_passed=True,
            ),
        )

    @staticmethod
    def _fields(child_slice: tuple[slice, slice] | None = None) -> np.ndarray:
        fields = np.zeros((2, 12, 12), dtype=float)
        fields[0] = 1.0
        if child_slice is not None:
            fields[0][child_slice] = 0.0
            fields[1][child_slice] = 1.0
        return fields

    def _update_three(self, fields: np.ndarray, state: GrainTrackerState) -> tuple[GrainTrackerState, object]:
        metrics = None
        for step in range(1, 4):
            state, metrics = update_grain_tracker(
                fields, state, float(step), self.dx_m, self.criteria
            )
        return state, metrics

    def test_allocated_empty_label_does_not_change_physical_grain_count(self) -> None:
        state, metrics = self._update_three(
            self._fields(), GrainTrackerState(self._records())
        )
        self.assertEqual(metrics.allocated_labels, 2)
        self.assertEqual(metrics.topology_components, 1)
        self.assertEqual(metrics.physical_grains, 1)
        self.assertEqual(metrics.recrystallized_grains, 0)
        self.assertEqual(state.records[1].status, "allocated")

    def test_subresolution_support_is_not_a_physical_grain(self) -> None:
        fields = self._fields((slice(2, 4), slice(2, 4)))
        state, metrics = self._update_three(fields, GrainTrackerState(self._records()))
        self.assertEqual(metrics.resolved_labels, 1)
        self.assertEqual(metrics.physical_grains, 1)
        self.assertEqual(state.records[1].status, "allocated")

    def test_disconnected_islands_are_not_one_physical_grain(self) -> None:
        fields = self._fields()
        fields[0, 2:5, 2:5] = 0.0
        fields[1, 2:5, 2:5] = 1.0
        fields[0, 7:10, 7:10] = 0.0
        fields[1, 7:10, 7:10] = 1.0
        state, metrics = self._update_three(fields, GrainTrackerState(self._records()))
        self.assertEqual(metrics.topology_components, 3)
        self.assertEqual(metrics.resolved_labels, 1)
        self.assertEqual(metrics.physical_grains, 1)
        self.assertEqual(state.records[1].status, "allocated")

    def test_persistent_distinct_child_is_promoted_and_counted(self) -> None:
        fields = self._fields((slice(3, 7), slice(4, 8)))
        state, metrics = self._update_three(fields, GrainTrackerState(self._records()))
        self.assertEqual(state.records[0].status, "active")
        self.assertEqual(state.records[1].status, "promoted")
        self.assertEqual(state.records[1].promoted_time_s, 3.0)
        self.assertEqual(metrics.physical_grains, 2)
        self.assertEqual(metrics.recrystallized_grains, 1)
        self.assertAlmostEqual(metrics.recrystallized_area_fraction, 16.0 / 144.0)

    def test_child_growth_is_recorded_without_being_required_for_stable_survival(self) -> None:
        state = GrainTrackerState(self._records())
        small = self._fields((slice(3, 6), slice(3, 6)))
        state, _ = update_grain_tracker(small, state, 1.0, self.dx_m, self.criteria)
        large = self._fields((slice(3, 7), slice(3, 7)))
        state, _ = update_grain_tracker(large, state, 2.0, self.dx_m, self.criteria)
        state, metrics = update_grain_tracker(large, state, 3.0, self.dx_m, self.criteria)
        self.assertTrue(state.records[1].ever_grew)
        self.assertEqual(state.records[1].status, "promoted")
        self.assertEqual(metrics.recrystallized_grains, 1)

    def test_crystallographically_equivalent_child_is_rejected(self) -> None:
        equivalent = 0.5 * math.pi + math.radians(2.0)
        fields = self._fields((slice(3, 7), slice(4, 8)))
        state, metrics = self._update_three(
            fields, GrainTrackerState(self._records(equivalent))
        )
        self.assertAlmostEqual(
            crystallographic_misorientation_rad(equivalent, 0.0, 4),
            math.radians(2.0),
        )
        self.assertEqual(state.records[1].status, "rejected")
        self.assertEqual(metrics.rejected_labels, 1)
        self.assertEqual(metrics.physical_grains, 1)

    def test_invalid_lineage_is_rejected(self) -> None:
        records = list(self._records())
        records[1] = GrainRecord(1, math.radians(12.0), 0, "unrelated", 1.0)
        state, metrics = self._update_three(
            self._fields((slice(3, 7), slice(4, 8))), GrainTrackerState(tuple(records))
        )
        self.assertEqual(state.records[1].status, "rejected")
        self.assertEqual(metrics.recrystallized_grains, 0)

    def test_phase_label_without_promoted_embryo_never_becomes_drx(self) -> None:
        records = list(self._records())
        records[1] = GrainRecord(
            1, math.radians(12.0), 0, "root-0/child-1", 1.0
        )
        state, metrics = self._update_three(
            self._fields((slice(3, 7), slice(4, 8))),
            GrainTrackerState(tuple(records)),
        )
        self.assertEqual(state.records[1].status, "rejected")
        self.assertEqual(metrics.recrystallized_grains, 0)

    def test_supported_child_retires_after_absence_grace_and_record_survives(self) -> None:
        supported = self._fields((slice(3, 7), slice(4, 8)))
        state, _ = self._update_three(supported, GrainTrackerState(self._records()))
        empty = self._fields()
        state, first = update_grain_tracker(empty, state, 4.0, self.dx_m, self.criteria)
        self.assertEqual(state.records[1].status, "promoted")
        self.assertEqual(first.recrystallized_grains, 0)
        state, second = update_grain_tracker(empty, state, 5.0, self.dx_m, self.criteria)
        self.assertEqual(state.records[1].status, "retired")
        self.assertEqual(state.records[1].lineage_id, "root-0/child-1")
        self.assertEqual(second.retired_labels, 1)
        self.assertEqual(second.physical_grains, 1)

    def test_periodic_boundary_support_is_one_component(self) -> None:
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:6, 0] = True
        mask[2:6, -1] = True
        self.assertEqual(periodic_component_count(mask), 1)

    def test_checkpoint_round_trip_is_exact(self) -> None:
        state, _ = self._update_three(
            self._fields((slice(3, 7), slice(4, 8))),
            GrainTrackerState(self._records()),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grain-tracker.json"
            save_grain_tracker(path, state)
            restored = load_grain_tracker(path)
        self.assertEqual(restored, state)


if __name__ == "__main__":
    unittest.main()
