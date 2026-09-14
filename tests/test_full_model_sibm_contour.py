import unittest

import numpy as np

from full_model.production.sibm_contour import measure_pair_contour


class SIBMContourTest(unittest.TestCase):
    def fields(self, shift_by_row=None):
        n = 48
        x = np.arange(n)[:, None]
        phi = np.broadcast_to((x-23.5)/2.0, (n, n)).copy()
        if shift_by_row is not None:
            phi -= shift_by_row[None, :]
        child = 0.5*(1.0+np.tanh(phi))
        eta = np.zeros((n, n, 3))
        eta[:, :, 0] = 1.0-child
        eta[:, :, 1] = child
        return eta

    def measure(self, eta, reference):
        mask = np.zeros((48, 48), bool)
        mask[8:40, 8:40] = True
        return measure_pair_contour(
            eta=eta, reference_eta=reference, active_mask=mask,
            centre_index=(24, 24), advance_direction_index=(-1, 0),
            parent_label=0, child_label=1, spacing_m=1e-7, dt_s=1e-6,
            local_pressure_Pa=2e6, active_window_radius_m=1.6e-6)

    def test_matched_flat_boundary_has_zero_displacement(self):
        eta = self.fields()
        result = self.measure(eta, eta.copy())
        self.assertAlmostEqual(result['bulge_tip_displacement_m'], 0.0)
        self.assertAlmostEqual(result['excess_bulge_area_m2'], 0.0)
        self.assertEqual(result['maximum_abs_nonpair_area_change_m2'], 0.0)

    def test_local_bulge_has_positive_normal_tip_and_no_nonpair_change(self):
        reference = self.fields()
        s = np.arange(48)
        bulge = 3.0*np.exp(-0.5*((s-24.0)/4.0)**2)
        eta = self.fields(-bulge)
        eta[:8] = reference[:8]
        eta[40:] = reference[40:]
        eta[:, :8] = reference[:, :8]
        eta[:, 40:] = reference[:, 40:]
        result = self.measure(eta, reference)
        self.assertGreater(result['bulge_tip_displacement_m'], 2e-7)
        self.assertGreater(result['bulge_amplitude_m'], 2e-7)
        self.assertGreater(result['excess_bulge_area_m2'], 0.0)
        self.assertEqual(result['maximum_abs_nonpair_area_change_m2'], 0.0)
        self.assertEqual(result['maximum_phase_change_outside_window'], 0.0)
        self.assertGreater(result['tip_distance_to_window_m'], 0.8e-6)


if __name__ == '__main__':
    unittest.main()
