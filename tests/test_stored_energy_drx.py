from __future__ import annotations

from pathlib import Path
import math
import tempfile
import unittest

import numpy as np

from asb_drx.multi_order import (
    BinaryCircularLimit,
    MultiOrderState,
    binary_boundary_energy_J_m2,
    diffuse_binary_circle,
    energy_checked_multi_order_step,
    equivalent_child_radius_m,
)
from asb_drx.stored_energy_drx import (
    StoredEnergyDRXParameters,
    StoredEnergyDRXState,
    advance_stored_energy_drx,
    load_stored_energy_drx_checkpoint,
    save_stored_energy_drx_checkpoint,
    stored_dislocation_energy_J_m,
    stored_energy_drx_step,
)


class StoredEnergyDRXTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = StoredEnergyDRXParameters(
            2.0e6,
            1.0e-6,
            5.0e-7,
            5.0e-9,
            (5.0e13, 1.0e13),
            3.5e6,
        )
        self.interface_length_m = 2.0 * math.sqrt(
            self.parameters.gradient_coefficient_J_m
            / self.parameters.pair_penalty_J_m3
        )
        self.boundary_energy_J_m2 = binary_boundary_energy_J_m2(
            self.parameters.phase_parameters
        )
        self.limit = BinaryCircularLimit(
            self.boundary_energy_J_m2,
            self.parameters.driving_energy_J_m3(0, 1),
            1.0,
        )

    def _state(self, ratio: float = 1.35, grid_points: int = 64) -> tuple[StoredEnergyDRXState, float]:
        dx_m = 1.6e-5 / grid_points
        fields = diffuse_binary_circle(
            grid_points,
            dx_m,
            ratio * self.limit.critical_radius_m,
            self.interface_length_m,
        )
        return StoredEnergyDRXState(MultiOrderState(fields, 0.0, 0), 1000.0), dx_m

    def test_driving_energy_is_explicit_stored_line_energy_difference(self) -> None:
        expected = self.parameters.stored_line_energy_J_m * (5.0e13 - 1.0e13)
        self.assertEqual(self.parameters.driving_energy_J_m3(0, 1), expected)
        self.assertEqual(self.limit.critical_radius_m, self.boundary_energy_J_m2 / expected)

    def test_pure_parent_has_no_reset_no_heat_and_no_child_creation(self) -> None:
        fields = np.zeros((2, 16, 16), dtype=float)
        fields[0] = 1.0
        initial = StoredEnergyDRXState(MultiOrderState(fields, 0.0, 0), 1000.0)
        accepted = stored_energy_drx_step(initial, 1.0e-6, 1.0e-4, self.parameters)
        self.assertTrue(np.array_equal(accepted.state.phase.eta_fields, fields))
        self.assertEqual(accepted.ledger.stored_energy_change_J_m, 0.0)
        self.assertEqual(accepted.ledger.heat_J_m, 0.0)
        self.assertEqual(accepted.state.temperature_K, initial.temperature_K)

    def test_lower_density_supercritical_child_grows(self) -> None:
        initial, dx_m = self._state(grid_points=128)
        radius_before = equivalent_child_radius_m(initial.phase.eta_fields, dx_m)
        final, _ = advance_stored_energy_drx(
            initial, dx_m, 1.0e-4, 200, self.parameters
        )
        self.assertGreater(
            equivalent_child_radius_m(final.phase.eta_fields, dx_m), radius_before
        )

    def test_free_energy_release_is_routed_to_heat_with_exact_ledger(self) -> None:
        initial, dx_m = self._state()
        final, ledgers = advance_stored_energy_drx(
            initial, dx_m, 1.0e-4, 40, self.parameters
        )
        total_stored_change = sum(item.stored_energy_change_J_m for item in ledgers)
        total_interface_change = sum(item.interfacial_energy_change_J_m for item in ledgers)
        total_heat = sum(item.heat_J_m for item in ledgers)
        self.assertLess(total_stored_change, 0.0)
        self.assertAlmostEqual(
            total_stored_change + total_interface_change + total_heat, 0.0, places=18
        )
        self.assertEqual(sum(item.closure_error_J_m for item in ledgers), 0.0)
        expected_temperature = 1000.0 + sum(item.heat_J_m3 for item in ledgers) / 3.5e6
        self.assertAlmostEqual(final.temperature_K, expected_temperature, places=12)

    def test_common_density_offset_does_not_change_binary_dynamics(self) -> None:
        initial, dx_m = self._state()
        base = energy_checked_multi_order_step(
            initial.phase.eta_fields, dx_m, 1.0e-4, self.parameters.phase_parameters
        )
        shifted_parameters = StoredEnergyDRXParameters(
            self.parameters.pair_penalty_J_m3,
            self.parameters.gradient_coefficient_J_m,
            self.parameters.mobility_m3_J_s,
            self.parameters.stored_line_energy_J_m,
            tuple(value + 1.0e14 for value in self.parameters.grain_dislocation_density_m2),
            self.parameters.volumetric_heat_capacity_J_m3_K,
        )
        shifted = energy_checked_multi_order_step(
            initial.phase.eta_fields, dx_m, 1.0e-4, shifted_parameters.phase_parameters
        )
        self.assertTrue(np.allclose(base.eta_fields, shifted.eta_fields, rtol=0.0, atol=1.0e-16))

    def test_stored_energy_changes_continuously_with_phase_support(self) -> None:
        initial, dx_m = self._state()
        before = stored_dislocation_energy_J_m(
            initial.phase.eta_fields, dx_m, self.parameters
        )
        accepted = stored_energy_drx_step(initial, dx_m, 1.0e-4, self.parameters)
        after = stored_dislocation_energy_J_m(
            accepted.state.phase.eta_fields, dx_m, self.parameters
        )
        self.assertEqual(after - before, accepted.ledger.stored_energy_change_J_m)
        self.assertEqual(
            self.parameters.grain_dislocation_density_m2, (5.0e13, 1.0e13)
        )

    def test_complete_current_state_restarts_exactly(self) -> None:
        initial, dx_m = self._state()
        continuous, _ = advance_stored_energy_drx(
            initial, dx_m, 1.0e-4, 40, self.parameters
        )
        first, _ = advance_stored_energy_drx(
            initial, dx_m, 1.0e-4, 20, self.parameters
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stored-energy-drx.npz"
            save_stored_energy_drx_checkpoint(path, first)
            restored = load_stored_energy_drx_checkpoint(path)
        segmented, _ = advance_stored_energy_drx(
            restored, dx_m, 1.0e-4, 20, self.parameters
        )
        self.assertTrue(
            np.array_equal(continuous.phase.eta_fields, segmented.phase.eta_fields)
        )
        self.assertEqual(continuous.phase.time_s, segmented.phase.time_s)
        self.assertEqual(continuous.phase.accepted_steps, segmented.phase.accepted_steps)
        self.assertEqual(continuous.temperature_K, segmented.temperature_K)


if __name__ == "__main__":
    unittest.main()
