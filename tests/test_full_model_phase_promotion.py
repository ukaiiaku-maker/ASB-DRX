import unittest

import numpy as np

from full_model.production.phase_promotion import conservative_neutral_density_relief


class PhasePromotionTransferTest(unittest.TestCase):
    def state(self):
        rp = np.full((8, 8, 2), 4.0e14)
        rm = np.full((8, 8, 2), 3.0e14)
        forest = np.full((8, 8, 2), 2.0e14)
        wall = np.full((8, 8), 1.0e14)
        gb = np.full((8, 8), 1.0e13)
        core = np.zeros((8, 8), bool); core[3:5, 3:5] = True
        shell = np.zeros((8, 8), bool); shell[2:6, 2:6] = True; shell[core] = False
        return rp, rm, forest, wall, gb, core, shell

    def test_relief_conserves_line_and_signed_burgers_content(self):
        state = self.state()
        result = conservative_neutral_density_relief(
            *state, target_core_density_m2=5e14, cell_area_m2=1e-14,
            maximum_density_m2=1e18)
        rp, rm, forest, wall, gb, ledger = result
        self.assertEqual(ledger.maximum_signed_burgers_density_change_m2, 0.0)
        self.assertLessEqual(abs(ledger.line_content_closure_m), 1e-15)
        self.assertGreater(ledger.line_content_transferred_m, 0.0)
        self.assertTrue(np.all(forest[state[-2]] == 0.0))
        self.assertTrue(np.all(wall[state[-2]] == 0.0))
        self.assertGreater(np.mean(gb[state[-1]]), 1e13)
        self.assertTrue(np.array_equal(rp-rm, state[0]-state[1]))

    def test_capacity_failure_rejects_without_hidden_clipping(self):
        with self.assertRaisesRegex(ValueError, "insufficient boundary-shell capacity"):
            conservative_neutral_density_relief(
                *self.state(), target_core_density_m2=5e14, cell_area_m2=1e-14,
                maximum_density_m2=1.1e13)

    def test_overlapping_or_empty_masks_are_rejected(self):
        state = list(self.state())
        state[-1] = state[-2].copy()
        with self.assertRaises(ValueError):
            conservative_neutral_density_relief(
                *state, target_core_density_m2=5e14, cell_area_m2=1e-14,
                maximum_density_m2=1e18)


if __name__ == "__main__":
    unittest.main()
