import unittest

import numpy as np

from full_model.production.arrhenius_kinetics import ActivatedProcess
from full_model.production.sibm_boundary import (
    BoundaryGraphState, SIBMParameters, advance_boundary,
    circular_bulge_pressure_Pa, select_resolved_hagb, state_from_json,
    state_json)


class SIBMBoundaryTest(unittest.TestCase):
    def params(self, mobility=2e-17):
        return SIBMParameters(
            0.5, mobility, ActivatedProcess("gb", 1e7, 0.0, 1e7),
            8e-20, 2e8, 1.0, 1.0, 0.1, 1e-6)

    def state(self, height=None, labels=(2, 7), angles=(0.1, -0.2)):
        return BoundaryGraphState(
            np.zeros(64) if height is None else np.asarray(height, float),
            *labels, *angles)

    def step(self, state, drive, mobility=2e-17):
        return advance_boundary(
            state, spacing_m=1e-7, dt_s=1e-5, temperature_K=1100.0,
            parameters=self.params(mobility), stored_difference_Pa=drive)

    def test_flat_equal_energy_is_stationary(self):
        out, velocity = self.step(self.state(), 0.0)
        np.testing.assert_array_equal(out.height_m, 0.0)
        np.testing.assert_array_equal(velocity, 0.0)

    def test_favorable_and_reversed_drive_reverse_motion(self):
        forward, _ = self.step(self.state(), 2e7)
        reverse, _ = self.step(self.state(), -2e7)
        self.assertGreater(np.mean(forward.height_m), 0.0)
        self.assertLess(np.mean(reverse.height_m), 0.0)

    def test_disabled_mobility_is_stationary(self):
        out, _ = self.step(self.state(), 2e7, mobility=0.0)
        np.testing.assert_array_equal(out.height_m, 0.0)

    def test_curvature_only_relaxes(self):
        x = np.arange(64)*2*np.pi/64
        h = 2e-8*np.cos(x)
        out, _ = self.step(self.state(h), 0.0)
        self.assertLess(np.ptp(out.height_m), np.ptp(h))
        self.assertLessEqual(out.ledger.free_energy_change_J, 0.0)

    def test_circular_critical_condition(self):
        p = self.params(); drive = 2e6; rc = p.boundary_energy_J_m2/drive
        self.assertLess(circular_bulge_pressure_Pa(0.8*rc, drive, 0.0, p), 0.0)
        self.assertGreater(circular_bulge_pressure_Pa(1.2*rc, drive, 0.0, p), 0.0)

    def test_energy_area_and_restart_ledgers(self):
        first, _ = self.step(self.state(), 2e7)
        self.assertLessEqual(abs(first.ledger.energy_closure_J), 1e-25)
        self.assertNotEqual(first.ledger.signed_area_change_m2, 0.0)
        restored = state_from_json(state_json(first), first.height_m)
        np.testing.assert_array_equal(restored.height_m, first.height_m)
        self.assertEqual(restored.ledger, first.ledger)
        a, _ = self.step(first, 2e7); b, _ = self.step(restored, 2e7)
        np.testing.assert_array_equal(a.height_m, b.height_m)
        self.assertEqual(a.ledger, b.ledger)

    def test_label_permutation_common_equation_symmetry(self):
        a, _ = self.step(self.state(labels=(2, 7), angles=(0.1, -0.2)), 2e7)
        b, _ = self.step(self.state(labels=(7, 2), angles=(-0.2, 0.1)), -2e7)
        np.testing.assert_allclose(a.height_m, -b.height_m, rtol=0.0, atol=0.0)
        self.assertEqual(a.left_orientation_rad, b.right_orientation_rad)

    def test_resolved_selector_ignores_unresolved_argmax_phase(self):
        x = np.arange(32)[:, None]
        q = 0.5*(1.0+np.tanh((x-15.5)/2.0))
        eta = np.zeros((32, 24, 3))
        eta[:, :, 0] = 1.0-q
        eta[:, :, 1] = q
        eta[:, :, 2] = 0.0
        # Four isolated pixels can win argmax locally but cannot establish an
        # independently resolved grain or HAGB.
        for i, j in ((14, 3), (15, 8), (16, 13), (17, 18)):
            eta[i, j, :] = (0.20, 0.20, 0.60)
        eta /= np.sum(eta, axis=2, keepdims=True)
        rho = np.where(x < 16, 4e14, 1e14)+np.zeros((32, 24))
        selected = select_resolved_hagb(
            eta, np.deg2rad([30.0, -20.0, 5.0]), rho,
            purity_threshold=0.8, min_pure_core_cells=16)
        self.assertEqual((selected.parent_label, selected.child_label), (0, 1))
        self.assertGreaterEqual(selected.parent_pure_core_cells, 16)
        self.assertGreaterEqual(selected.child_pure_core_cells, 16)

    def test_resolved_selector_rejects_unresolved_pair(self):
        eta = np.full((8, 8, 2), 0.5)
        with self.assertRaisesRegex(ValueError, "no resolved existing HAGB"):
            select_resolved_hagb(
                eta, np.deg2rad([0.0, 30.0]), np.ones((8, 8)),
                min_pure_core_cells=4)


if __name__ == "__main__":
    unittest.main()
