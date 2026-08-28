from __future__ import annotations

from pathlib import Path
import math
import tempfile
import unittest

import numpy as np

from asb_drx.analytical import ExpFloorLaw
from asb_drx.antiplane import solve_periodic_antiplane
from asb_drx.local_coupled import (
    LocalCoupledState,
    advance_local_coupled,
    diffuse_temperature_periodic_exact,
    load_local_coupled_checkpoint,
    local_coupled_step,
    save_local_coupled_checkpoint,
)
from asb_drx.multi_order import BinaryCircularLimit, diffuse_binary_circle
from asb_drx.spatial_coupled import (
    SpatialCoupledParameters,
    SpatialCoupledState,
    SpatialMechanismControls,
    spatial_coupled_step,
)


EV_J = 1.602176634e-19


class LocalCoupledTests(unittest.TestCase):
    def setUp(self) -> None:
        self.law = ExpFloorLaw(
            1.5 * EV_J, 1.2e9, 1000.0, 0.2, 2.0, 2.5, 1.0e12, 4.0, 2.5e-10
        )
        self.parameters = SpatialCoupledParameters(
            8.0e10, 3.5e6, 5.0, 5.0e-9, 1.0e14, 2.0e6, 1.0e-6, 5.0e-7
        )

    @staticmethod
    def _pure(points: int) -> np.ndarray:
        fields = np.zeros((2, points, points))
        fields[0] = 1.0
        return fields

    def _mixed_state(self, points: int = 16) -> tuple[LocalCoupledState, float]:
        dx_m = 1.6e-5 / points
        boundary = math.sqrt(
            self.parameters.gradient_coefficient_J_m
            * self.parameters.pair_penalty_J_m3
        ) / 3.0
        limit = BinaryCircularLimit(
            boundary, self.parameters.stored_line_energy_J_m * 4.0e13, 1.0
        )
        interface = 2.0 * math.sqrt(
            self.parameters.gradient_coefficient_J_m
            / self.parameters.pair_penalty_J_m3
        )
        eta = diffuse_binary_circle(
            points, dx_m, 1.35 * limit.critical_radius_m, interface
        )
        density = np.empty_like(eta)
        density[0] = 5.0e13
        density[1] = 1.0e13
        return (
            LocalCoupledState(
                1.0e8 / self.parameters.shear_modulus_Pa,
                np.zeros((points, points)),
                np.full((points, points), 1000.0),
                density,
                eta,
            ),
            dx_m,
        )

    def test_uniform_pure_parent_reduces_to_common_stress_kernel(self) -> None:
        points = 8
        dx_m = 1.0e-5
        density = np.empty((2, points, points))
        density[0] = 1.0e14
        density[1] = 1.0e13
        temperature = np.full((points, points), 1000.0)
        plastic = np.zeros((points, points))
        fields = self._pure(points)
        local = local_coupled_step(
            LocalCoupledState(1.0e8 / 8.0e10, plastic, temperature, density, fields),
            10.0, dx_m, 1.0e-5, self.law, self.parameters,
        )
        common = spatial_coupled_step(
            SpatialCoupledState(1.0e8, 0.0, plastic, temperature, density, fields),
            10.0, dx_m, 1.0e-5, self.law, self.parameters,
        )
        self.assertTrue(
            math.isclose(
                local.equilibrium.mean_stress_Pa,
                common.state.stress_Pa,
                rel_tol=3.0e-16,
            )
        )
        self.assertTrue(np.array_equal(local.state.plastic_shear, common.state.plastic_shear))
        self.assertTrue(np.array_equal(local.state.temperature_K, common.state.temperature_K))
        self.assertTrue(np.array_equal(local.state.forest_density_m2, common.state.forest_density_m2))
        self.assertTrue(np.array_equal(local.state.eta_fields, common.state.eta_fields))

    def test_density_heterogeneity_generates_local_stress_redistribution(self) -> None:
        points = 16
        dx_m = 1.0e-6
        coordinate = np.linspace(0.0, 2.0 * math.pi, points, endpoint=False)
        pattern = np.sin(coordinate)[:, None] * np.ones((1, points))
        density = np.empty((2, points, points))
        density[0] = 1.0e14 * (1.0 + 0.2 * pattern)
        density[1] = 1.0e13
        state = LocalCoupledState(
            1.0e8 / 8.0e10,
            np.zeros((points, points)),
            np.full((points, points), 1000.0),
            density,
            self._pure(points),
        )
        accepted = local_coupled_step(
            state, 10.0, dx_m, 1.0e-6, self.law, self.parameters,
            controls=SpatialMechanismControls(evolve_phase=False),
        )
        self.assertGreater(float(np.std(accepted.state.plastic_shear)), 0.0)
        self.assertGreater(float(np.std(accepted.equilibrium.stress_x_Pa)), 0.0)
        self.assertLess(
            accepted.equilibrium.equilibrium_residual_Pa_m_inv,
            1.0e-10 * float(np.max(np.abs(accepted.equilibrium.stress_x_Pa))) / dx_m,
        )

    def test_storage_is_limited_by_available_local_plastic_work(self) -> None:
        points = 8
        density = np.empty((2, points, points))
        density[0] = 1.0e14
        density[1] = 1.0e13
        state = LocalCoupledState(
            1.0e8 / self.parameters.shear_modulus_Pa,
            np.zeros((points, points)),
            np.full((points, points), 1000.0),
            density,
            self._pure(points),
        )
        expensive_storage = SpatialCoupledParameters(
            self.parameters.shear_modulus_Pa,
            self.parameters.volumetric_heat_capacity_J_m3_K,
            self.parameters.thermal_conductivity_W_m_K,
            self.parameters.stored_line_energy_J_m,
            2.0e17,
            self.parameters.pair_penalty_J_m3,
            self.parameters.gradient_coefficient_J_m,
            self.parameters.phase_mobility_m3_J_s,
        )
        accepted = local_coupled_step(
            state, 10.0, 1.0e-6, 1.0e-5, self.law, expensive_storage,
            controls=SpatialMechanismControls(evolve_phase=False),
        )
        self.assertEqual(accepted.halvings, 0)
        self.assertEqual(accepted.storage_limited_fraction, 1.0)
        self.assertGreaterEqual(accepted.ledger.mechanical_heat_J_m3, 0.0)
        stored_increment = expensive_storage.stored_line_energy_J_m * float(
            np.mean(accepted.state.forest_density_m2[0] - density[0])
        )
        self.assertLessEqual(
            stored_increment,
            accepted.ledger.plastic_work_J_m3
            + 1.0e-10 * max(accepted.ledger.plastic_work_J_m3, 1.0),
        )

    def test_exact_periodic_diffusion_removes_explicit_CFL_restriction(self) -> None:
        points = 32
        dx_m = 5.0e-7
        diffusivity = 5.0 / 3.5e6
        dt_s = 1.0e-6
        coordinate = np.linspace(0.0, 2.0 * math.pi, points, endpoint=False)
        initial = 1000.0 + 2.0 * np.sin(coordinate)[None, :]
        initial = np.broadcast_to(initial, (points, points)).copy()
        final = diffuse_temperature_periodic_exact(initial, diffusivity, dt_s, dx_m)
        wave_number = 2.0 * math.pi / (points * dx_m)
        expected_amplitude = 2.0 * math.exp(-diffusivity * wave_number**2 * dt_s)
        measured_amplitude = 0.5 * (float(np.max(final)) - float(np.min(final)))
        self.assertTrue(math.isclose(measured_amplitude, expected_amplitude, rel_tol=2.0e-12, abs_tol=1.0e-13))
        self.assertAlmostEqual(float(np.mean(final)), float(np.mean(initial)))

    def test_global_and_thermal_ledgers_close(self) -> None:
        initial, dx_m = self._mixed_state()
        final, steps = advance_local_coupled(
            initial, 10.0, dx_m, 1.0e-5, 20, self.law, self.parameters
        )
        external = sum(item.ledger.external_work_J_m3 for item in steps)
        accounted = sum(
            item.ledger.elastic_change_J_m3
            + item.ledger.stored_change_J_m3
            + item.ledger.interface_order_change_J_m3
            + item.ledger.mechanical_heat_J_m3
            + item.ledger.phase_heat_J_m3
            for item in steps
        )
        self.assertLess(abs(external - accounted), 1.0e-8 * max(abs(external), 1.0))
        self.assertTrue(np.all(np.isfinite(final.temperature_K)))

    def test_phase_disabled_and_isothermal_controls_are_exact(self) -> None:
        initial, dx_m = self._mixed_state()
        final, steps = advance_local_coupled(
            initial, 10.0, dx_m, 1.0e-5, 5, self.law, self.parameters,
            controls=SpatialMechanismControls(False, False),
        )
        self.assertTrue(np.array_equal(final.temperature_K, initial.temperature_K))
        self.assertTrue(np.array_equal(final.eta_fields, initial.eta_fields))
        for item in steps:
            self.assertEqual(item.ledger.phase_heat_J_m3, 0.0)
            self.assertAlmostEqual(
                item.ledger.bath_heat_J_m3, item.ledger.mechanical_heat_J_m3
            )

    def test_complete_state_restarts_bitwise_exactly(self) -> None:
        initial, dx_m = self._mixed_state()
        continuous, _ = advance_local_coupled(
            initial, 10.0, dx_m, 1.0e-5, 12, self.law, self.parameters
        )
        first, _ = advance_local_coupled(
            initial, 10.0, dx_m, 1.0e-5, 6, self.law, self.parameters
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "local.npz"
            save_local_coupled_checkpoint(path, first)
            restored = load_local_coupled_checkpoint(path)
        segmented, _ = advance_local_coupled(
            restored, 10.0, dx_m, 1.0e-5, 6, self.law, self.parameters
        )
        self.assertEqual(continuous.applied_shear, segmented.applied_shear)
        self.assertTrue(np.array_equal(continuous.plastic_shear, segmented.plastic_shear))
        self.assertTrue(np.array_equal(continuous.temperature_K, segmented.temperature_K))
        self.assertTrue(np.array_equal(continuous.forest_density_m2, segmented.forest_density_m2))
        self.assertTrue(np.array_equal(continuous.eta_fields, segmented.eta_fields))
        self.assertEqual(continuous.time_s, segmented.time_s)
        self.assertEqual(continuous.accepted_steps, segmented.accepted_steps)

    def test_checkpoint_schema_is_enforced(self) -> None:
        initial, _ = self._mixed_state()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            np.savez(
                path,
                schema=np.asarray("asb-drx-local-antiplane-coupled/obsolete"),
                applied_shear=np.asarray(initial.applied_shear),
                plastic_shear=initial.plastic_shear,
                temperature_K=initial.temperature_K,
                forest_density_m2=initial.forest_density_m2,
                eta_fields=initial.eta_fields,
                time_s=np.asarray(initial.time_s),
                accepted_steps=np.asarray(initial.accepted_steps),
            )
            with self.assertRaises(ValueError):
                load_local_coupled_checkpoint(path)

    def test_equilibrium_is_derived_from_complete_state(self) -> None:
        initial, dx_m = self._mixed_state()
        result = solve_periodic_antiplane(
            initial.applied_shear,
            initial.plastic_shear,
            self.parameters.shear_modulus_Pa,
            dx_m,
        )
        self.assertAlmostEqual(result.mean_stress_Pa, 1.0e8)

    def test_invalid_local_ledger_tolerance_is_rejected(self) -> None:
        initial, dx_m = self._mixed_state()
        with self.assertRaises(ValueError):
            local_coupled_step(
                initial,
                10.0,
                dx_m,
                1.0e-5,
                self.law,
                self.parameters,
                relative_ledger_tolerance=1.0,
            )


if __name__ == "__main__":
    unittest.main()
