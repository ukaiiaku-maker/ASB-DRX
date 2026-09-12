from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
import tempfile

import numpy as np

from asb_drx.arrhenius_v3 import (
    ArrheniusMechanism,
    BoundedActivationEntropy,
    ExpFloorEnthalpy,
)
from asb_drx.integrated_cdd_v3 import (
    IntegratedCDDParameters,
    IntegratedCDDState,
    apply_integrated_reactions,
    frozen_mode_eigenvalues_s_inv,
    integrated_cdd_step,
    load_integrated_cdd_checkpoint,
    save_integrated_cdd_checkpoint,
)
from asb_drx.staggered_cdd import StaggeredSignedState, nye_compatibility_residual_m_inv


class IntegratedCDDV3Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.n = 32
        self.dx = 16.0e-6 / self.n
        self.b = 2.86e-10
        enthalpy = ExpFloorEnthalpy(
            0.55 * 1.602176634e-19, 0.7e9, 900.0, 0.2, 1.3, 2.0
        )
        glide = ArrheniusMechanism(
            enthalpy, BoundedActivationEntropy(reference_kB=0.25),
            attempt_frequency_s_inv=2.0e7, event_increment=1.0e-9,
            validity_temperature_K=(500.0, 1400.0),
            validity_stress_Pa=(0.0, 2.0e9),
        )
        dyads = np.zeros((4, 3, 3))
        dyads[:, 0, 2] = np.asarray([0.45, 0.30, -0.30, -0.45])
        self.parameters = IntegratedCDDParameters(
            glide=glide,
            shear_modulus_Pa=45.0e9,
            volumetric_heat_capacity_J_m3_K=2.5e6,
            line_energy_J_m=2.0e-9,
            slip_dyads_crystal=dyads,
            correlation_mobility_m_Pa_s=2.0e-13,
            backstress_coefficient=0.7,
            diffusion_coefficient=1.1,
            reference_density_m2=5.0e14,
        )
        population = np.full((4, self.n), 2.5e14)
        signed = StaggeredSignedState(
            population, population.copy(), np.zeros_like(population),
            self.b, self.dx,
        )
        fp = np.broadcast_to(np.eye(3), (self.n, 3, 3)).copy()
        self.state = IntegratedCDDState(
            signed, fp, fp.copy(), np.full(self.n, 900.0)
        )

    def test_homogeneous_increment_matches_arrhenius_material_point_exactly(self) -> None:
        rate = 2.0e4
        dt = 2.0e-9
        advanced = integrated_cdd_step(self.state, rate, dt, self.parameters)
        old_stress = 0.0
        # The old stress is zero, hence forward-minus-reverse glide is exactly
        # zero and this first increment is the exact elastic Gate-A limit.
        expected_stress = self.parameters.shear_modulus_Pa * rate * dt
        self.assertAlmostEqual(
            advanced.ledger.elastic_energy_change_J_m3,
            expected_stress**2 / (2.0 * self.parameters.shear_modulus_Pa),
        )
        self.assertEqual(advanced.ledger.plastic_work_J_m3, 0.0)
        np.testing.assert_array_equal(
            advanced.state.signed.mobile_plus_m2,
            self.state.signed.mobile_plus_m2,
        )
        self.assertLess(abs(advanced.ledger.total_energy_residual_J_m3), 1.0e-12)

    def test_loaded_homogeneous_step_couples_flux_slip_fp_rotation_and_heat(self) -> None:
        loaded = IntegratedCDDState(
            self.state.signed, self.state.plastic_deformation_gradient,
            self.state.orientation, self.state.temperature_K,
            applied_shear=0.012,
        )
        advanced = integrated_cdd_step(loaded, 0.0, 1.0e-9, self.parameters)
        old_stress = self.parameters.shear_modulus_Pa * loaded.applied_shear
        weights = self.parameters.slip_dyads_crystal[:, 0, 2]
        expected_slip = np.empty(4)
        for family in range(4):
            velocity = self.parameters.glide.net_rate_s_inv(
                old_stress * weights[family], 900.0
            )
            expected_slip[family] = (
                self.b * 5.0e14 * velocity * advanced.accepted_dt_s
            )
        np.testing.assert_allclose(
            advanced.state.signed.face_slip,
            np.broadcast_to(expected_slip[:, None], (4, self.n)),
            rtol=2.0e-14, atol=1.0e-20,
        )
        self.assertGreater(float(np.max(np.abs(
            advanced.state.signed.face_slip - loaded.signed.face_slip
        ))), 0.0)
        self.assertGreater(float(np.max(np.abs(
            advanced.state.plastic_deformation_gradient
            - loaded.plastic_deformation_gradient
        ))), 0.0)
        self.assertGreaterEqual(advanced.ledger.heat_J_m3, 0.0)
        self.assertLess(
            float(np.max(np.abs(nye_compatibility_residual_m_inv(advanced.state.signed)))),
            1.0e-12,
        )
        self.assertEqual(advanced.state.physical_grain_count, 1)
        self.assertFalse(hasattr(advanced.state, "grain_labels"))

    def test_unloaded_compatible_gradient_releases_correlation_energy(self) -> None:
        x = np.arange(self.n)
        wave = 0.01 * 2.5e14 * np.sin(2.0 * np.pi * 3.0 * x / self.n)
        plus = self.state.signed.mobile_plus_m2.copy()
        minus = self.state.signed.mobile_minus_m2.copy()
        plus[0] += wave
        minus[0] -= wave
        kappa = plus - minus
        slip = np.zeros_like(plus)
        # Solve gamma_i-gamma_{i-1}=-b*dx*kappa_i for each zero-mean family.
        for family in range(4):
            for cell in range(1, self.n):
                slip[family, cell] = (
                    slip[family, cell - 1]
                    - self.b * self.dx * kappa[family, cell]
                )
            slip[family] -= np.mean(slip[family])
        signed = StaggeredSignedState(plus, minus, slip, self.b, self.dx)
        state = IntegratedCDDState(
            signed, self.state.plastic_deformation_gradient,
            self.state.orientation, self.state.temperature_K,
        )
        self.assertLess(
            float(np.max(np.abs(nye_compatibility_residual_m_inv(signed)))),
            2.0e-11,
        )
        advanced = integrated_cdd_step(state, 0.0, 1.0e-8, self.parameters)
        self.assertLess(advanced.ledger.correlation_energy_change_J_m3, 0.0)
        self.assertGreater(advanced.ledger.heat_J_m3, 0.0)

    def test_work_energy_and_population_ledgers_close_without_clipping(self) -> None:
        loaded = IntegratedCDDState(
            self.state.signed, self.state.plastic_deformation_gradient,
            self.state.orientation, self.state.temperature_K,
            applied_shear=0.01,
        )
        advanced = integrated_cdd_step(loaded, 1.0e3, 1.0e-9, self.parameters)
        ledger = advanced.ledger
        scale = max(abs(ledger.external_work_J_m3), abs(ledger.plastic_work_J_m3), 1.0)
        thermal_roundoff = (
            16.0 * np.finfo(float).eps
            * self.parameters.volumetric_heat_capacity_J_m3_K
            * float(np.max(advanced.state.temperature_K))
        )
        self.assertLessEqual(
            abs(ledger.total_energy_residual_J_m3),
            max(2.0e-11 * scale, thermal_roundoff),
        )
        self.assertLess(abs(ledger.work_projection_residual_J_m3), 2.0e-13 * scale)
        self.assertEqual(ledger.flux.clipping_added_m2, 0.0)
        self.assertLess(abs(ledger.flux.balance_residual_m2), 1.0)

    def test_complete_integrated_state_restarts_bitwise(self) -> None:
        loaded = IntegratedCDDState(
            self.state.signed, self.state.plastic_deformation_gradient,
            self.state.orientation, self.state.temperature_K,
            applied_shear=0.01,
        )
        first = integrated_cdd_step(loaded, 1.0e3, 1.0e-9, self.parameters).state
        continuous = integrated_cdd_step(first, 1.0e3, 1.0e-9, self.parameters).state
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "integrated.npz"
            save_integrated_cdd_checkpoint(checkpoint, first)
            restored = load_integrated_cdd_checkpoint(checkpoint)
        restarted = integrated_cdd_step(restored, 1.0e3, 1.0e-9, self.parameters).state
        for name in (
            "mobile_plus_m2", "mobile_minus_m2", "locked_plus_m2",
            "locked_minus_m2", "wall_plus_m2", "wall_minus_m2", "face_slip",
        ):
            self.assertTrue(np.array_equal(
                getattr(continuous.signed, name), getattr(restarted.signed, name)
            ))
        self.assertTrue(np.array_equal(
            continuous.plastic_deformation_gradient,
            restarted.plastic_deformation_gradient,
        ))
        self.assertTrue(np.array_equal(continuous.orientation, restarted.orientation))
        self.assertTrue(np.array_equal(continuous.temperature_K, restarted.temperature_K))
        self.assertEqual(continuous.applied_shear, restarted.applied_shear)
        self.assertEqual(continuous.time_s, restarted.time_s)
        self.assertEqual(continuous.accepted_steps, restarted.accepted_steps)

    def test_imex_removes_log_diffusion_cfl_without_filtering(self) -> None:
        x = np.arange(self.n)
        wave = 1.0 + 0.2 * np.cos(2.0 * np.pi * 11.0 * x / self.n)
        plus = self.state.signed.mobile_plus_m2 * wave[None, :]
        minus = self.state.signed.mobile_minus_m2 * wave[None, :]
        state = IntegratedCDDState(
            StaggeredSignedState(
                plus, minus, self.state.signed.face_slip, self.b, self.dx
            ),
            self.state.plastic_deformation_gradient, self.state.orientation,
            self.state.temperature_K,
        )
        explicit = integrated_cdd_step(
            state, 0.0, 0.1,
            replace(self.parameters, correlation_integration="explicit"),
        )
        imex = integrated_cdd_step(
            state, 0.0, 0.1,
            replace(self.parameters, correlation_integration="imex"),
        )
        self.assertGreater(explicit.halvings, 0)
        self.assertEqual(imex.halvings, 0)
        self.assertEqual(imex.ledger.flux.clipping_added_m2, 0.0)
        self.assertLess(imex.ledger.correlation_energy_change_J_m3, 0.0)

    def test_local_reactions_are_bounded_and_close_line_and_burgers_ledgers(self) -> None:
        reaction = ArrheniusMechanism(
            ExpFloorEnthalpy(
                0.05 * 1.602176634e-19, 1.0e9, 900.0, 0.5, 1.0, 2.0
            ),
            BoundedActivationEntropy(), 1.0e5,
            validity_temperature_K=(500.0, 1400.0),
            validity_stress_Pa=(0.0, 2.0e9),
        )
        parameters = replace(
            self.parameters, multiplication_per_slip_m2=2.0e13,
            annihilation=reaction, locking=reaction,
            unlocking=reaction, wall_capture=reaction,
        )
        seeded = StaggeredSignedState(
            self.state.signed.mobile_plus_m2,
            self.state.signed.mobile_minus_m2,
            self.state.signed.face_slip, self.b, self.dx,
            np.full((4, self.n), 2.0e13),
            np.full((4, self.n), 1.0e13),
        )
        advanced, ledger = apply_integrated_reactions(
            seeded, np.full((4, self.n), 0.002),
            self.state.temperature_K, 1.0e-5, parameters,
        )
        self.assertGreater(ledger.pair_generated_m2, 0.0)
        self.assertGreater(ledger.pair_annihilated_m2, 0.0)
        self.assertGreater(ledger.locked_transfer_m2, 0.0)
        self.assertGreater(ledger.unlocked_transfer_m2, 0.0)
        self.assertGreater(ledger.wall_capture_m2, 0.0)
        self.assertLess(
            abs(ledger.line_balance_residual_m2),
            3.0e-15 * ledger.line_content_before_m2,
        )
        self.assertLess(
            ledger.maximum_signed_burgers_residual_m2,
            3.0e-15 * ledger.line_content_before_m2,
        )
        self.assertGreaterEqual(float(np.min(advanced.mobile_plus_m2)), 0.0)

    def test_frozen_full_symbol_is_damped_through_nyquist(self) -> None:
        weights = self.parameters.slip_dyads_crystal[:, 0, 2]
        spectra = frozen_mode_eigenvalues_s_inv(
            32, 16.0e-6, 2.5e14, 0.45e9 * weights, 900.0,
            self.b, self.parameters,
        )
        maximum_real = max(
            float(np.max(values.real)) for values in spectra.values()
        )
        self.assertLessEqual(maximum_real, 1.0e-7)
        nyquist = spectra[16]
        self.assertLess(float(np.max(nyquist.real)), -1.0e-3)


if __name__ == "__main__":
    unittest.main()
