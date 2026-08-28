from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from asb_drx.analytical import ExpFloorLaw
from asb_drx.material_point import (
    MaterialPointParameters,
    MaterialPointState,
    advance_material_point,
    load_material_point_checkpoint,
    material_point_step,
    save_material_point_checkpoint,
)
from asb_drx.thermodynamics import DislocationReservoirs


EV_J = 1.602176634e-19


class MaterialPointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.law = ExpFloorLaw(
            barrier_ref_J=1.5 * EV_J,
            stress_ref_Pa=1.2e9,
            reference_temperature_K=1000.0,
            floor_fraction=0.2,
            shape_a=2.0,
            shape_n=2.5,
            rate_prefactor_s_inv=1.0e12,
            density_exponent_p=4.0,
            burgers_m=2.5e-10,
        )
        self.parameters = MaterialPointParameters(
            shear_modulus_Pa=8.0e10,
            volumetric_heat_capacity_J_m3_K=3.5e6,
            stored_line_energy_J_m=5.0e-9,
            forest_storage_per_plastic_strain_m2=1.0e14,
        )
        self.initial = MaterialPointState(
            stress_Pa=1.0e8,
            applied_shear=0.0,
            plastic_shear=0.0,
            temperature_K=1000.0,
            reservoirs=DislocationReservoirs(1.0e13, 1.0e14, 0.0, 0.0),
        )

    def test_homogeneous_rate_and_finite_loading_update(self) -> None:
        step = material_point_step(self.initial, 10.0, 1.0e-5, self.law, self.parameters)
        expected = self.law.shear_rate_s_inv(
            step.local_activation_stress_Pa,
            self.initial.reservoirs.forest_m2,
            self.initial.temperature_K,
        )
        self.assertAlmostEqual(step.plastic_rate_s_inv / expected, 1.0, places=13)
        elastic_increment = (
            step.state.applied_shear - step.state.plastic_shear
        )
        expected_stress = self.initial.stress_Pa + self.parameters.shear_modulus_Pa * elastic_increment
        self.assertAlmostEqual(step.state.stress_Pa / expected_stress, 1.0, places=13)

    def test_incremental_work_energy_closure_and_positive_heat(self) -> None:
        final, ledgers = advance_material_point(
            self.initial, 10.0, 1.0e-5, 100, self.law, self.parameters
        )
        scale = sum(abs(item.external_work_J_m3) for item in ledgers)
        closure = sum(item.closure_error_J_m3 for item in ledgers)
        self.assertLess(abs(closure), 1.0e-12 * scale)
        self.assertTrue(all(item.heat_J_m3 >= 0.0 for item in ledgers))
        self.assertTrue(all(item.stored_dislocation_J_m3 >= 0.0 for item in ledgers))
        self.assertGreater(final.temperature_K, self.initial.temperature_K)
        self.assertGreater(final.reservoirs.forest_m2, self.initial.reservoirs.forest_m2)

    def test_zero_storage_routes_all_plastic_work_to_heat(self) -> None:
        parameters = MaterialPointParameters(8.0e10, 3.5e6, 5.0e-9, 0.0)
        step = material_point_step(self.initial, 10.0, 1.0e-5, self.law, parameters)
        self.assertEqual(step.ledger.stored_dislocation_J_m3, 0.0)
        self.assertAlmostEqual(
            step.ledger.heat_J_m3 / step.ledger.plastic_work_J_m3, 1.0, places=13
        )

    def test_complete_current_material_point_state_restarts_exactly(self) -> None:
        continuous, _ = advance_material_point(
            self.initial, 10.0, 1.0e-5, 100, self.law, self.parameters
        )
        first, _ = advance_material_point(
            self.initial, 10.0, 1.0e-5, 40, self.law, self.parameters
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "material_point.npz"
            save_material_point_checkpoint(checkpoint, first)
            restored = load_material_point_checkpoint(checkpoint)
        segmented, _ = advance_material_point(
            restored, 10.0, 1.0e-5, 60, self.law, self.parameters
        )
        self.assertEqual(continuous, segmented)

    def test_impossible_storage_partition_is_rejected(self) -> None:
        impossible = MaterialPointParameters(8.0e10, 3.5e6, 5.0e-9, 1.0e20)
        with self.assertRaisesRegex(RuntimeError, "no energetically admissible"):
            material_point_step(
                self.initial,
                10.0,
                1.0e-5,
                self.law,
                impossible,
                maximum_halvings=3,
            )


if __name__ == "__main__":
    unittest.main()
