from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from asb_drx.analytical import ExpFloorLaw
from asb_drx.material_point import (
    MaterialPointParameters,
    MaterialPointState,
    advance_material_point,
)
from asb_drx.shear_layer import (
    ShearLayerParameters,
    ShearLayerState,
    advance_shear_layer,
    load_shear_layer_checkpoint,
    save_shear_layer_checkpoint,
)
from asb_drx.thermodynamics import DislocationReservoirs


EV_J = 1.602176634e-19


class ShearLayerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.law = ExpFloorLaw(
            1.5 * EV_J, 1.2e9, 1000.0, 0.2, 2.0, 2.5, 1.0e12, 4.0, 2.5e-10
        )
        self.layer_parameters = ShearLayerParameters(
            8.0e10, 3.5e6, 25.0, 5.0e-9, 1.0e14
        )
        self.points = 32
        self.dx_m = 2.0e-5
        self.initial = ShearLayerState(
            1.0e8,
            0.0,
            np.zeros(self.points),
            np.full(self.points, 1000.0),
            np.full(self.points, 1.0e14),
        )

    def test_homogeneous_layer_reduces_to_material_point(self) -> None:
        final_layer, _ = advance_shear_layer(
            self.initial, 10.0, self.dx_m, 1.0e-5, 100, self.law, self.layer_parameters
        )
        point_parameters = MaterialPointParameters(8.0e10, 3.5e6, 5.0e-9, 1.0e14)
        point_initial = MaterialPointState(
            1.0e8,
            0.0,
            0.0,
            1000.0,
            DislocationReservoirs(1.0e13, 1.0e14, 0.0, 0.0),
        )
        final_point, _ = advance_material_point(
            point_initial, 10.0, 1.0e-5, 100, self.law, point_parameters
        )
        self.assertAlmostEqual(final_layer.stress_Pa / final_point.stress_Pa, 1.0, places=13)
        self.assertTrue(np.allclose(final_layer.temperature_K, final_point.temperature_K, rtol=1e-13, atol=0.0))
        self.assertTrue(np.allclose(final_layer.forest_density_m2, final_point.reservoirs.forest_m2, rtol=1e-13, atol=0.0))

    def test_global_mechanical_and_thermal_ledgers_close(self) -> None:
        temperature = np.full(self.points, 1000.0)
        temperature += 0.25 * np.sin(np.linspace(0.0, 2.0 * np.pi, self.points, endpoint=False))
        state = ShearLayerState(
            1.0e8, 0.0, np.zeros(self.points), temperature, np.full(self.points, 1.0e14)
        )
        _, ledgers = advance_shear_layer(
            state, 10.0, self.dx_m, 1.0e-5, 100, self.law, self.layer_parameters
        )
        scale = sum(abs(item.external_work_J_m3) for item in ledgers)
        self.assertLess(abs(sum(item.mechanical_closure_error_J_m3 for item in ledgers)), 1e-11 * scale)
        self.assertLess(abs(sum(item.thermal_closure_error_J_m3 for item in ledgers)), 1e-9 * scale)

    def test_conduction_damps_temperature_variance_and_preserves_mean(self) -> None:
        coordinate = np.linspace(0.0, 2.0 * np.pi, self.points, endpoint=False)
        temperature = 1000.0 + 2.0 * np.sin(coordinate)
        state = ShearLayerState(
            0.0, 0.0, np.zeros(self.points), temperature, np.full(self.points, 1.0e14)
        )
        final, _ = advance_shear_layer(
            state, 0.0, self.dx_m, 1.0e-5, 100, self.law, self.layer_parameters
        )
        self.assertLess(float(np.var(final.temperature_K)), float(np.var(temperature)))
        self.assertAlmostEqual(float(np.mean(final.temperature_K)), float(np.mean(temperature)), places=12)

    def test_complete_current_layer_state_restarts_exactly(self) -> None:
        continuous, _ = advance_shear_layer(
            self.initial, 10.0, self.dx_m, 1.0e-5, 100, self.law, self.layer_parameters
        )
        first, _ = advance_shear_layer(
            self.initial, 10.0, self.dx_m, 1.0e-5, 40, self.law, self.layer_parameters
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "layer.npz"
            save_shear_layer_checkpoint(checkpoint, first)
            restored = load_shear_layer_checkpoint(checkpoint)
        segmented, _ = advance_shear_layer(
            restored, 10.0, self.dx_m, 1.0e-5, 60, self.law, self.layer_parameters
        )
        self.assertEqual(continuous.stress_Pa, segmented.stress_Pa)
        self.assertEqual(continuous.applied_shear, segmented.applied_shear)
        self.assertEqual(continuous.time_s, segmented.time_s)
        self.assertEqual(continuous.accepted_steps, segmented.accepted_steps)
        self.assertTrue(np.array_equal(continuous.plastic_shear, segmented.plastic_shear))
        self.assertTrue(np.array_equal(continuous.temperature_K, segmented.temperature_K))
        self.assertTrue(np.array_equal(continuous.forest_density_m2, segmented.forest_density_m2))


if __name__ == "__main__":
    unittest.main()
