import unittest

import numpy as np

from full_model.production.wall_ordering_energy import (
    WallOrderingParameters, gradient_wall_energy_J_m3,
    local_wall_energy_J_m3,
)


class WallOrderingEnergyTest(unittest.TestCase):
    def setUp(self):
        self.p = WallOrderingParameters(
            8e5, 5e14, 1.3, 0.3, 2e5, 2e-9, 5e14, 1e-6
        )

    def test_tangle_and_ordered_wall_are_distinct(self):
        rho = self.p.center_ratio*self.p.density_scale_m2
        tangle = local_wall_energy_J_m3(rho, 0.0, 0.0, self.p)
        wall = local_wall_energy_J_m3(
            rho, self.p.target_wall_density_m2, 1.0, self.p
        )
        self.assertLess(float(wall), float(tangle))

    def test_balanced_density_does_not_imply_order(self):
        rho = np.full((8, 8), 1e15)
        energy = local_wall_energy_J_m3(rho, np.zeros_like(rho), 0.0, self.p)
        self.assertTrue(np.all(np.isfinite(energy)))
        self.assertTrue(np.all(energy == 0.0))

    def test_gradient_term_is_nonnegative_and_zero_for_uniform_order(self):
        uniform = np.full((16, 16), 0.4)
        self.assertTrue(np.all(gradient_wall_energy_J_m3(uniform, 1e-7, self.p) == 0.0))
        uniform[4:8, 4:8] = 0.8
        self.assertGreater(float(np.sum(gradient_wall_energy_J_m3(uniform, 1e-7, self.p))), 0.0)


if __name__ == '__main__':
    unittest.main()
