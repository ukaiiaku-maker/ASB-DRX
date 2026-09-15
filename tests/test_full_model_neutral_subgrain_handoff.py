import unittest

import numpy as np

from full_model.production.moving_front import (
    DefectState, initialize_existing_subgrain_front, reconstruct_mixture,
)


class NeutralSubgrainHandoffTest(unittest.TestCase):
    def test_mapping_is_exact_for_arbitrary_diffuse_support(self):
        rng = np.random.default_rng(19)
        shape = (18, 20, 4)
        parent = DefectState(
            1e12+rng.random(shape)*2e14,
            1e12+rng.random(shape)*2e14,
            rng.random(shape)*3e14,
            rng.random(shape[:2])*4e14,
        )
        x = np.linspace(-1.0, 1.0, shape[0])[:, None]
        y = np.linspace(-1.0, 1.0, shape[1])[None, :]
        for width in (0.04, 0.08, 0.16):
            support = 0.5*(1.0-np.tanh((x*x+y*y-0.35)/width))
            mapped = initialize_existing_subgrain_front(parent, support, 0, 1)
            mixture = reconstruct_mixture(mapped)
            for actual, expected in zip(
                    (mixture.rp, mixture.rm, mixture.forest, mixture.wall),
                    (parent.rp, parent.rm, parent.forest, parent.wall)):
                np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(mapped.chi, support)
            np.testing.assert_array_equal(mapped.processed_max, support)
            self.assertEqual(mapped.ledger.parent_line_processed_m, 0.0)
            self.assertEqual(mapped.ledger.signed_burgers_change_m2, 0.0)
            self.assertEqual(mapped.ledger.heat_released_J, 0.0)


if __name__ == "__main__":
    unittest.main()
