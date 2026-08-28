from __future__ import annotations

from pathlib import Path
import math
import tempfile
import unittest

import numpy as np

from asb_drx.analytical import ExpFloorLaw
from asb_drx.multi_order import BinaryCircularLimit, binary_boundary_energy_J_m2, diffuse_binary_circle
from asb_drx.shear_layer import ShearLayerParameters, ShearLayerState, shear_layer_step
from asb_drx.spatial_coupled import (
    SpatialCoupledParameters, SpatialCoupledState, advance_spatial_coupled,
    load_spatial_coupled_checkpoint, save_spatial_coupled_checkpoint,
    spatial_coupled_step,
)


EV_J = 1.602176634e-19


class SpatialCoupledTests(unittest.TestCase):
    def setUp(self) -> None:
        self.law = ExpFloorLaw(1.5 * EV_J, 1.2e9, 1000.0, 0.2, 2.0, 2.5, 1.0e12, 4.0, 2.5e-10)
        self.parameters = SpatialCoupledParameters(
            8.0e10, 3.5e6, 5.0, 5.0e-9, 1.0e14, 2.0e6, 1.0e-6, 5.0e-7
        )

    @staticmethod
    def _pure(points: int) -> np.ndarray:
        fields = np.zeros((2, points, points))
        fields[0] = 1.0
        return fields

    def _mixed_state(self, points: int = 32) -> tuple[SpatialCoupledState, float]:
        dx_m = 1.6e-5 / points
        phase = self.parameters
        boundary = math.sqrt(phase.gradient_coefficient_J_m * phase.pair_penalty_J_m3) / 3.0
        driving = phase.stored_line_energy_J_m * 4.0e13
        limit = BinaryCircularLimit(boundary, driving, 1.0)
        interface = 2.0 * math.sqrt(phase.gradient_coefficient_J_m / phase.pair_penalty_J_m3)
        eta = diffuse_binary_circle(points, dx_m, 1.35 * limit.critical_radius_m, interface)
        density = np.empty_like(eta)
        density[0] = 5.0e13
        density[1] = 1.0e13
        state = SpatialCoupledState(
            1.0e8, 0.0, np.zeros((points, points)), np.full((points, points), 1000.0),
            density, eta, 0.0, 0,
        )
        return state, dx_m

    def test_uniform_pure_parent_reduces_to_common_stress_shear_layer(self) -> None:
        points = 8
        dx_m = 1.0e-5
        density = np.empty((2, points, points))
        density[0] = 1.0e14
        density[1] = 1.0e13
        spatial = spatial_coupled_step(
            SpatialCoupledState(1.0e8, 0.0, np.zeros((points, points)), np.full((points, points), 1000.0), density, self._pure(points)),
            10.0, dx_m, 1.0e-5, self.law, self.parameters,
        )
        layer = shear_layer_step(
            ShearLayerState(1.0e8, 0.0, np.zeros(points), np.full(points, 1000.0), np.full(points, 1.0e14)),
            10.0, dx_m, 1.0e-5, self.law,
            ShearLayerParameters(8.0e10, 3.5e6, 5.0, 5.0e-9, 1.0e14),
        )
        self.assertEqual(spatial.state.stress_Pa, layer.state.stress_Pa)
        self.assertTrue(np.all(spatial.state.temperature_K == layer.state.temperature_K[0]))
        self.assertTrue(np.all(spatial.state.forest_density_m2[0] == layer.state.forest_density_m2[0]))
        self.assertTrue(np.array_equal(spatial.state.eta_fields, self._pure(points)))

    def test_periodic_conduction_damps_temperature_and_preserves_mean(self) -> None:
        points = 16
        coordinate = np.linspace(0.0, 2.0 * math.pi, points, endpoint=False)
        temperature = 1000.0 + 0.25 * np.sin(coordinate)[None, :]
        temperature = np.broadcast_to(temperature, (points, points)).copy()
        density = np.empty((2, points, points)); density[0] = 1.0e14; density[1] = 1.0e13
        initial = SpatialCoupledState(0.0, 0.0, np.zeros((points, points)), temperature, density, self._pure(points))
        final, _ = advance_spatial_coupled(initial, 0.0, 2.0e-5, 1.0e-5, 100, self.law, self.parameters)
        self.assertLess(float(np.std(final.temperature_K)), float(np.std(initial.temperature_K)))
        self.assertAlmostEqual(float(np.mean(final.temperature_K)), float(np.mean(initial.temperature_K)))

    def test_global_spatial_ledger_closes(self) -> None:
        initial, dx_m = self._mixed_state()
        final, ledgers = advance_spatial_coupled(initial, 10.0, dx_m, 1.0e-5, 50, self.law, self.parameters)
        external = sum(item.external_work_J_m3 for item in ledgers)
        accounted = sum(item.elastic_change_J_m3 + item.stored_change_J_m3 + item.interface_order_change_J_m3 + item.mechanical_heat_J_m3 + item.phase_heat_J_m3 for item in ledgers)
        self.assertLess(abs(external - accounted), 1.0e-8 * abs(external))
        self.assertLess(abs(sum(item.global_closure_error_J_m3 for item in ledgers)), 1.0e-8 * abs(external))
        self.assertTrue(np.all(np.isfinite(final.temperature_K)))
        self.assertTrue(np.all(final.forest_density_m2 >= initial.forest_density_m2))

    def test_pure_parent_loading_does_not_create_child_field(self) -> None:
        points = 8
        density = np.empty((2, points, points)); density[0] = 1.0e14; density[1] = 1.0e13
        initial = SpatialCoupledState(1.0e8, 0.0, np.zeros((points, points)), np.full((points, points), 1000.0), density, self._pure(points))
        final, _ = advance_spatial_coupled(initial, 10.0, 1.0e-5, 1.0e-5, 20, self.law, self.parameters)
        self.assertTrue(np.array_equal(final.eta_fields[1], np.zeros((points, points))))

    def test_complete_current_state_restarts_exactly(self) -> None:
        initial, dx_m = self._mixed_state()
        continuous, _ = advance_spatial_coupled(initial, 10.0, dx_m, 1.0e-5, 20, self.law, self.parameters)
        first, _ = advance_spatial_coupled(initial, 10.0, dx_m, 1.0e-5, 10, self.law, self.parameters)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spatial.npz"
            save_spatial_coupled_checkpoint(path, first)
            restored = load_spatial_coupled_checkpoint(path)
        segmented, _ = advance_spatial_coupled(restored, 10.0, dx_m, 1.0e-5, 10, self.law, self.parameters)
        self.assertEqual(continuous.stress_Pa, segmented.stress_Pa)
        self.assertEqual(continuous.applied_shear, segmented.applied_shear)
        self.assertTrue(np.array_equal(continuous.plastic_shear, segmented.plastic_shear))
        self.assertTrue(np.array_equal(continuous.temperature_K, segmented.temperature_K))
        self.assertTrue(np.array_equal(continuous.forest_density_m2, segmented.forest_density_m2))
        self.assertTrue(np.array_equal(continuous.eta_fields, segmented.eta_fields))
        self.assertEqual(continuous.time_s, segmented.time_s)
        self.assertEqual(continuous.accepted_steps, segmented.accepted_steps)


if __name__ == "__main__":
    unittest.main()
