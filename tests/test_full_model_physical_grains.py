import math
import unittest

import numpy as np

from full_model.production.physical_grains import (
    GrainCriteria, GrainRecord, GrainTracker, tracker_from_json, tracker_to_json,
    update_tracker,
)


class PhysicalGrainTest(unittest.TestCase):
    def setUp(self):
        self.criteria = GrainCriteria(
            interface_width_m=1e-7, minimum_area_width_factor=2.0,
            purity_threshold=0.8, minimum_persistence_s=1e-6,
            stable_support_s=2e-6, retirement_grace_s=1e-6,
            minimum_misorientation_rad=math.radians(3),
            orientation_symmetry_order=4, minimum_stored_energy_drop_J_m3=1e6)

    def fields(self):
        eta = np.zeros((12, 12, 2))
        eta[:, :, 0] = 1.0
        eta[4:8, 4:8, 0] = 0.02
        eta[4:8, 4:8, 1] = 0.98
        energy = np.full((12, 12), 5e7)
        energy[4:8, 4:8] = 1e7
        return eta, energy

    def test_label_without_promoted_embryo_is_rejected(self):
        records = (
            GrainRecord(0, 0.0, None, "initial/0", 0.0, None, False),
            GrainRecord(1, math.radians(5), 0, "initial/0/label1", 0.0, None, False),
        )
        eta, energy = self.fields()
        tracker, metrics = update_tracker(
            eta, energy, GrainTracker(records), 1e-6, 1e-7, 1e-7, self.criteria)
        self.assertEqual(tracker.records[1].status, "rejected")
        self.assertEqual(metrics.physical_drx_grains, 0)

    def test_promoted_lower_energy_persistent_grain_is_physical(self):
        records = (
            GrainRecord(0, 0.0, None, "initial/0", 0.0, None, False),
            GrainRecord(1, math.radians(5), 0, "initial/0/embryo-7", 0.0, 7, True),
        )
        eta, energy = self.fields()
        tracker = GrainTracker(records)
        tracker, _ = update_tracker(eta, energy, tracker, 1e-6, 1e-7, 1e-7, self.criteria)
        tracker, metrics = update_tracker(eta, energy, tracker, 2e-6, 1e-7, 1e-7, self.criteria)
        self.assertEqual(tracker.records[1].status, "recrystallized")
        self.assertEqual(metrics.physical_drx_grains, 1)
        self.assertGreater(metrics.recrystallized_area_fraction, 0.0)

    def test_high_purity_fragment_below_interface_scaled_area_is_not_resolved(self):
        eta, energy = self.fields()
        eta[:, :, 0] = 1.0
        eta[:, :, 1] = 0.0
        eta[5, 5, 0] = 0.0
        eta[5, 5, 1] = 1.0
        records = (
            GrainRecord(0, 0.0, None, "initial/0", 0.0, None, False),
            GrainRecord(1, math.radians(5), 0, "initial/0/embryo-1", 0.0, 1, True),
        )
        _, metrics = update_tracker(
            eta, energy, GrainTracker(records), 2e-6, 1e-7, 1e-7, self.criteria)
        self.assertEqual(metrics.physical_drx_grains, 0)

    def test_restart_json_is_exact(self):
        state = GrainTracker((GrainRecord(0, 0.0, None, "initial/0", 0.0, None, False),), 1e-6)
        encoded = tracker_to_json(state)
        self.assertEqual(tracker_from_json(encoded), state)
        self.assertEqual(tracker_to_json(tracker_from_json(encoded)), encoded)


if __name__ == "__main__":
    unittest.main()
