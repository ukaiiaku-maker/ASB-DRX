import tempfile
import unittest
from pathlib import Path

import numpy as np

from full_model.production.intragranular_subgrain import (
    FAMILY_NORMALS, IntragranularParameters, advance_intragranular,
    incompatibility_by_family_m2, initialize_one_grain,
    load_checkpoint, recognize_subgrain, reconstructed_incompatibility_vector_m2,
    save_checkpoint,
)


class IntragranularSubgrainTest(unittest.TestCase):
    def test_one_grain_has_no_label_allocation_and_balanced_control_stays_unrotated(self):
        p = IntragranularParameters()
        state = initialize_one_grain(48, p)
        self.assertEqual(state.physical_grain_count, 1)
        self.assertTrue(np.all(state.orientation_rad == 0.0))
        self.assertTrue(np.all(state.wall_plus_m2-state.wall_minus_m2 == 0.0))
        self.assertFalse(recognize_subgrain(state, p)["qualified"])

    def test_short_subcritical_horizon_does_not_create_a_subgrain(self):
        p = IntragranularParameters()
        state = advance_intragranular(initialize_one_grain(48, p), 1.0e-4, 5e-6, p)
        self.assertFalse(recognize_subgrain(state, p)["qualified"])

    def test_deformation_generates_compatible_low_angle_subgrain(self):
        p = IntragranularParameters()
        state = advance_intragranular(initialize_one_grain(64, p), 1.5e-3, 5e-6, p)
        result = recognize_subgrain(state, p)
        self.assertTrue(result["qualified"], result)
        self.assertGreater(result["misorientation_deg"], 2.0)
        self.assertLess(result["misorientation_deg"], 15.0)
        self.assertLess(result["frank_bilby_relative_residual"], 0.20)
        self.assertGreater(result["boundary_order_closure_fraction"], 0.70)
        self.assertLess(result["interior_orientation_gradient_rms_m1"],
                        result["boundary_orientation_gradient_rms_m1"])
        self.assertTrue(result["lower_density_interior"])
        self.assertEqual(state.physical_grain_count, 1)

    def test_signed_inventory_converges_to_independent_kinematic_incompatibility(self):
        p = IntragranularParameters()
        state = advance_intragranular(initialize_one_grain(64, p), 1.5e-3, 5e-6, p)
        target_family = incompatibility_by_family_m2(state.orientation_rad, p)
        target_vector = np.einsum("ai,axy->ixy", FAMILY_NORMALS, target_family)
        actual = reconstructed_incompatibility_vector_m2(state)
        relative = np.linalg.norm(actual-target_vector)/np.linalg.norm(target_vector)
        self.assertLess(relative, 0.12)

    def test_restart_is_exact(self):
        p = IntragranularParameters()
        initial = initialize_one_grain(48, p)
        continuous = advance_intragranular(initial, 1.0e-3, 5e-6, p)
        first = advance_intragranular(initial, 4.0e-4, 5e-6, p)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"checkpoint.npz"
            save_checkpoint(path, first)
            restarted = advance_intragranular(load_checkpoint(path), 6.0e-4, 5e-6, p)
        for name in ("differential_slip", "mobile_plus_m2", "mobile_minus_m2",
                     "forest_m2", "wall_plus_m2", "wall_minus_m2", "wall_order"):
            np.testing.assert_array_equal(getattr(restarted, name), getattr(continuous, name))

    def test_grid_refinement_is_below_five_percent(self):
        p = IntragranularParameters()
        metrics = []
        for n in (64, 96):
            state = advance_intragranular(initialize_one_grain(n, p), 1.5e-3, 5e-6, p)
            metrics.append(recognize_subgrain(state, p))
        for name in ("misorientation_deg", "equivalent_radius_m"):
            relative = abs(metrics[1][name]-metrics[0][name])/abs(metrics[1][name])
            self.assertLess(relative, 0.05, (name, metrics))


if __name__ == "__main__":
    unittest.main()
