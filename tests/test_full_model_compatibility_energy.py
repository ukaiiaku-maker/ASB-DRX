import unittest

import numpy as np

from full_model.production.compatibility_energy import decompose_compatibility_energy


class CompatibilityEnergyTest(unittest.TestCase):
    def test_physical_terms_are_penalty_independent_and_unit_closed(self):
        shape = (8, 8)
        kwargs = dict(
            signed_gnd_density_m2=np.full(shape, 2e12),
            boundary_density_m2=np.full(shape, 3e12),
            orientation_gradient_m1=np.full(shape, 500.0),
            line_tension_J_m=2e-9, cell_area_m2=1e-14,
            represented_thickness_m=1e-6, burgers_m=2.5e-10)
        weak = decompose_compatibility_energy(
            **kwargs, alpha_penalty_coefficient=1e-9,
            gb_penalty_coefficient=2e-9)
        stiff = decompose_compatibility_energy(
            **kwargs, alpha_penalty_coefficient=1e-6,
            gb_penalty_coefficient=2e-6)
        self.assertEqual(weak.physical_total_J, stiff.physical_total_J)
        self.assertGreater(stiff.numerical_total_J, weak.numerical_total_J)
        self.assertGreaterEqual(weak.long_range_gnd_J, 0.0)


if __name__ == "__main__":
    unittest.main()
