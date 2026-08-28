from __future__ import annotations

from pathlib import Path
import math
import tempfile
import unittest

import numpy as np

from asb_drx.analytical import ExpFloorLaw
from asb_drx.coupled import (
    CoupledParameters,
    CoupledState,
    advance_coupled,
    coupled_step,
    load_coupled_checkpoint,
    save_coupled_checkpoint,
)
from asb_drx.material_point import (
    MaterialPointParameters,
    MaterialPointState,
    material_point_step,
)
from asb_drx.multi_order import BinaryCircularLimit, MultiOrderState, binary_boundary_energy_J_m2, diffuse_binary_circle
from asb_drx.stored_energy_drx import (
    StoredEnergyDRXState,
    stored_energy_drx_step,
)
from asb_drx.thermodynamics import DislocationReservoirs


EV_J = 1.602176634e-19


class CoupledTests(unittest.TestCase):
    def setUp(self) -> None:
        self.law = ExpFloorLaw(
            1.5 * EV_J,
            1.2e9,
            1000.0,
            0.2,
            2.0,
            2.5,
            1.0e12,
            4.0,
            2.5e-10,
        )
        self.parameters = CoupledParameters(
            8.0e10,
            3.5e6,
            5.0e-9,
            1.0e14,
            2.0e6,
            1.0e-6,
            5.0e-7,
        )
        self.dx_m = 1.6e-5 / 64
        phase_parameters = self.parameters.phase_parameters((5.0e13, 1.0e13))
        boundary_energy = binary_boundary_energy_J_m2(phase_parameters.phase_parameters)
        limit = BinaryCircularLimit(
            boundary_energy, phase_parameters.driving_energy_J_m3(0, 1), 1.0
        )
        interface_length_m = 2.0 * math.sqrt(
            self.parameters.gradient_coefficient_J_m
            / self.parameters.pair_penalty_J_m3
        )
        self.mixed_fields = diffuse_binary_circle(
            64,
            self.dx_m,
            1.35 * limit.critical_radius_m,
            interface_length_m,
        )

    @staticmethod
    def _pure_parent_fields(points: int = 32) -> np.ndarray:
        fields = np.zeros((2, points, points), dtype=float)
        fields[0] = 1.0
        return fields

    def test_pure_parent_limit_matches_material_point_exactly(self) -> None:
        fields = self._pure_parent_fields()
        coupled_initial = CoupledState(
            1.0e8,
            0.0,
            np.zeros(2),
            np.array((1.0e14, 1.0e13)),
            fields,
            1000.0,
        )
        coupled = coupled_step(
            coupled_initial,
            10.0,
            1.0e-6,
            1.0e-5,
            self.law,
            self.parameters,
        )
        point_initial = MaterialPointState(
            1.0e8,
            0.0,
            0.0,
            1000.0,
            DislocationReservoirs(1.0e13, 1.0e14, 0.0, 0.0),
        )
        point = material_point_step(
            point_initial,
            10.0,
            1.0e-5,
            self.law,
            MaterialPointParameters(8.0e10, 3.5e6, 5.0e-9, 1.0e14),
        )
        self.assertEqual(coupled.state.stress_Pa, point.state.stress_Pa)
        self.assertEqual(coupled.state.applied_shear, point.state.applied_shear)
        self.assertEqual(coupled.state.grain_plastic_shear[0], point.state.plastic_shear)
        self.assertEqual(coupled.state.forest_density_m2[0], point.state.reservoirs.forest_m2)
        self.assertEqual(coupled.state.temperature_K, point.state.temperature_K)
        self.assertTrue(np.array_equal(coupled.state.eta_fields, fields))
        self.assertEqual(coupled.ledger.phase_heat_J_m3, 0.0)

    def test_zero_mechanics_limit_matches_stored_energy_phase_step(self) -> None:
        density = np.array((5.0e13, 1.0e13))
        coupled_initial = CoupledState(
            0.0, 0.0, np.zeros(2), density, self.mixed_fields, 1000.0
        )
        coupled = coupled_step(
            coupled_initial,
            0.0,
            self.dx_m,
            1.0e-4,
            self.law,
            self.parameters,
        )
        phase = stored_energy_drx_step(
            StoredEnergyDRXState(MultiOrderState(self.mixed_fields, 0.0, 0), 1000.0),
            self.dx_m,
            1.0e-4,
            self.parameters.phase_parameters(density),
        )
        self.assertTrue(np.array_equal(coupled.state.eta_fields, phase.state.phase.eta_fields))
        self.assertEqual(coupled.state.temperature_K, phase.state.temperature_K)
        self.assertEqual(coupled.ledger.external_work_J_m3, 0.0)
        self.assertEqual(coupled.ledger.mechanical_heat_J_m3, 0.0)
        self.assertEqual(coupled.ledger.phase_heat_J_m3, phase.ledger.heat_J_m3)

    def test_combined_external_stored_interface_heat_ledger_closes(self) -> None:
        initial = CoupledState(
            1.0e8,
            0.0,
            np.zeros(2),
            np.array((5.0e13, 1.0e13)),
            self.mixed_fields,
            1000.0,
        )
        final, ledgers = advance_coupled(
            initial, 10.0, self.dx_m, 1.0e-5, 100, self.law, self.parameters
        )
        external = sum(item.external_work_J_m3 for item in ledgers)
        accounted = sum(
            item.elastic_energy_change_J_m3
            + item.stored_energy_change_J_m3
            + item.interface_order_energy_change_J_m3
            + item.mechanical_heat_J_m3
            + item.phase_heat_J_m3
            for item in ledgers
        )
        self.assertLess(abs(external - accounted), 1.0e-8 * abs(external))
        self.assertLess(
            abs(sum(item.global_closure_error_J_m3 for item in ledgers)),
            1.0e-8 * abs(external),
        )
        total_heat = sum(
            item.mechanical_heat_J_m3 + item.phase_heat_J_m3 for item in ledgers
        )
        self.assertAlmostEqual(
            self.parameters.volumetric_heat_capacity_J_m3_K
            * (final.temperature_K - initial.temperature_K),
            total_heat,
            places=6,
        )

    def test_pure_parent_loading_never_creates_allocated_child_support(self) -> None:
        fields = self._pure_parent_fields()
        initial = CoupledState(
            1.0e8,
            0.0,
            np.zeros(2),
            np.array((1.0e14, 1.0e13)),
            fields,
            1000.0,
        )
        final, _ = advance_coupled(
            initial, 10.0, 1.0e-6, 1.0e-5, 20, self.law, self.parameters
        )
        self.assertTrue(np.array_equal(final.eta_fields[1], np.zeros_like(fields[1])))
        self.assertGreater(final.forest_density_m2[0], initial.forest_density_m2[0])

    def test_phase_and_mechanics_use_one_accepted_time_increment(self) -> None:
        initial = CoupledState(
            1.0e8,
            0.0,
            np.zeros(2),
            np.array((5.0e13, 1.0e13)),
            self.mixed_fields,
            1000.0,
        )
        accepted = coupled_step(
            initial, 10.0, self.dx_m, 1.0e-4, self.law, self.parameters
        )
        self.assertEqual(accepted.state.time_s, accepted.accepted_dt_s)
        self.assertEqual(accepted.state.applied_shear, 10.0 * accepted.accepted_dt_s)
        self.assertEqual(accepted.state.accepted_steps, 1)

    def test_complete_current_state_restarts_exactly(self) -> None:
        initial = CoupledState(
            1.0e8,
            0.0,
            np.zeros(2),
            np.array((5.0e13, 1.0e13)),
            self.mixed_fields,
            1000.0,
        )
        continuous, _ = advance_coupled(
            initial, 10.0, self.dx_m, 1.0e-5, 40, self.law, self.parameters
        )
        first, _ = advance_coupled(
            initial, 10.0, self.dx_m, 1.0e-5, 20, self.law, self.parameters
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "coupled.npz"
            save_coupled_checkpoint(path, first)
            restored = load_coupled_checkpoint(path)
        segmented, _ = advance_coupled(
            restored, 10.0, self.dx_m, 1.0e-5, 20, self.law, self.parameters
        )
        self.assertEqual(continuous.stress_Pa, segmented.stress_Pa)
        self.assertEqual(continuous.applied_shear, segmented.applied_shear)
        self.assertTrue(
            np.array_equal(continuous.grain_plastic_shear, segmented.grain_plastic_shear)
        )
        self.assertTrue(
            np.array_equal(continuous.forest_density_m2, segmented.forest_density_m2)
        )
        self.assertTrue(np.array_equal(continuous.eta_fields, segmented.eta_fields))
        self.assertEqual(continuous.temperature_K, segmented.temperature_K)
        self.assertEqual(continuous.time_s, segmented.time_s)
        self.assertEqual(continuous.accepted_steps, segmented.accepted_steps)


if __name__ == "__main__":
    unittest.main()
