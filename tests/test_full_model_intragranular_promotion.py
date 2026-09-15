import unittest

import numpy as np

from full_model.production.dislocation_free_energy import DislocationFreeEnergyParameters
from full_model.production.intragranular_promotion import (
    promote_qualified_subgrain, signed_growth_drive_J_m3,
)
from full_model.production.intragranular_subgrain import (
    IntragranularParameters, advance_intragranular, initialize_one_grain,
    recognize_subgrain,
)


class IntragranularPromotionTest(unittest.TestCase):
    def setUp(self):
        self.p = IntragranularParameters()
        self.state = advance_intragranular(
            initialize_one_grain(64, self.p), 1.5e-3, 5e-6, self.p
        )
        self.recognition = recognize_subgrain(self.state, self.p)
        line = 0.5*45e9*self.p.burgers_m**2
        self.energy = DislocationFreeEnergyParameters(
            line, .06*line, 1e14, 1.2*line*5e14*.3, 5e14, 1.3, .3, .25
        )

    def test_qualified_subgrain_handoff_is_content_and_energy_neutral(self):
        result = promote_qualified_subgrain(
            self.state, self.recognition, self.p, self.energy
        )
        self.assertEqual(result.parent_label, 0)
        self.assertEqual(result.child_label, 1)
        self.assertEqual(result.inherited_orientation_rad,
                         self.recognition["interior_orientation_rad"])
        self.assertEqual(result.phase_simplex_residual, 0.0)
        self.assertEqual(result.common_energy_after_J, result.common_energy_before_J)
        ledger = result.front_state.ledger
        self.assertEqual(ledger.parent_line_processed_m, 0.0)
        self.assertEqual(ledger.child_line_transmitted_m, 0.0)
        self.assertEqual(ledger.boundary_line_stored_m, 0.0)
        self.assertEqual(ledger.neutral_pair_annihilated_m, 0.0)
        self.assertEqual(ledger.sink_line_m, 0.0)
        self.assertEqual(ledger.line_closure_m, 0.0)
        self.assertEqual(ledger.signed_burgers_change_m2, 0.0)
        self.assertEqual(ledger.heat_released_J, 0.0)
        self.assertEqual(ledger.swept_volume_m3, 0.0)
        np.testing.assert_array_equal(result.front_state.chi, result.eta[:, :, 1])
        np.testing.assert_array_equal(
            result.front_state.processed_max, result.eta[:, :, 1])

    def test_unqualified_state_cannot_allocate_phase(self):
        initial = initialize_one_grain(64, self.p)
        with self.assertRaises(ValueError):
            promote_qualified_subgrain(initial, recognize_subgrain(initial, self.p),
                                       self.p, self.energy)

    def test_wall_without_stored_energy_advantage_cannot_allocate_phase(self):
        no_recovery = IntragranularParameters(recovery_attempt_s=1e-20)
        state = advance_intragranular(
            initialize_one_grain(64, no_recovery), 1.5e-3, 5e-6, no_recovery
        )
        recognition = recognize_subgrain(state, no_recovery)
        self.assertTrue(recognition["qualified"])
        self.assertFalse(recognition["lower_density_interior"])
        with self.assertRaises(ValueError):
            promote_qualified_subgrain(state, recognition, no_recovery, self.energy)

    def test_stored_energy_contrast_reverses_growth_direction(self):
        forward = signed_growth_drive_J_m3(7e15, 5e15, self.energy)
        reverse = signed_growth_drive_J_m3(5e15, 7e15, self.energy)
        self.assertGreater(float(forward), 0.0)
        self.assertEqual(float(reverse), -float(forward))


if __name__ == "__main__":
    unittest.main()
