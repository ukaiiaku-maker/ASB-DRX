from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from asb_drx.bertin_bcc import (
    BertinBCCParameters,
    BertinBCCState,
    advance_bertin_bcc,
    cubic_family_angle_deg,
    evaluate_bertin_bcc,
    initial_orientation,
    load_bertin_checkpoint,
    save_bertin_checkpoint,
)


INITIAL_DENSITIES = np.asarray([4.8, 4.7, 4.9, 4.75]) * 1.0e14


def make_state(axis, perturbation_deg=0.1):
    return BertinBCCState(
        plastic_deformation_gradient=np.eye(3),
        initial_orientation=initial_orientation(
            axis,
            perturbation_axis_lab=(1.0, 0.0, 0.0),
            perturbation_deg=perturbation_deg,
        ),
        densities_m2=INITIAL_DENSITIES,
    )


class BertinBCCUnitTests(unittest.TestCase):
    def test_reference_parameters_and_temperature_envelope(self):
        parameters = BertinBCCParameters()
        self.assertEqual(parameters.velocity_exponent, 25.0)
        self.assertEqual(parameters.inactive_relaxation_s_inv, 5.0e8)
        self.assertEqual(parameters.elastic_constants_Pa(300.0), (261.12e9, 165.07e9, 55.28e9))
        with self.assertRaisesRegex(ValueError, "50 to 1000"):
            parameters.elastic_constants_Pa(1001.0)
        with self.assertRaisesRegex(ValueError, "outside the published"):
            from asb_drx.bertin_bcc import bertin_bcc_step

            bertin_bcc_step(make_state((0, 0, 1)), 1.0e5, 0.001, 300.0, parameters)

    def test_initial_orientation_and_plastic_incompressibility(self):
        state = make_state((4, 1, 9), perturbation_deg=0.0)
        response = evaluate_bertin_bcc(state, 300.0, BertinBCCParameters())
        np.testing.assert_allclose(
            response.loading_axis_crystal,
            np.asarray([4.0, 1.0, 9.0]) / np.linalg.norm([4.0, 1.0, 9.0]),
            atol=2.0e-15,
        )
        self.assertLess(abs(float(np.trace(response.plastic_velocity_gradient_s_inv))), 1.0e-6)
        np.testing.assert_allclose(
            response.plastic_spin_s_inv + response.plastic_spin_s_inv.T,
            0.0,
            atol=1.0e-8,
        )

    def test_complete_state_restarts_bitwise(self):
        parameters = BertinBCCParameters()
        initial = make_state((4, 1, 9))
        continuous, _ = advance_bertin_bcc(
            initial, 2.0e8, 0.2, 0.002, 300.0, parameters, sample_stride=25
        )
        first, _ = advance_bertin_bcc(
            initial, 2.0e8, 0.1, 0.002, 300.0, parameters, sample_stride=25
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "bertin.npz"
            save_bertin_checkpoint(checkpoint, first)
            restored = load_bertin_checkpoint(checkpoint)
        segmented, _ = advance_bertin_bcc(
            restored, 2.0e8, 0.2, 0.002, 300.0, parameters, sample_stride=25
        )
        self.assertTrue(
            np.array_equal(
                continuous.plastic_deformation_gradient,
                segmented.plastic_deformation_gradient,
            )
        )
        self.assertTrue(np.array_equal(continuous.densities_m2, segmented.densities_m2))
        self.assertEqual(continuous.axial_true_strain, segmented.axial_true_strain)
        self.assertEqual(continuous.time_s, segmented.time_s)
        self.assertEqual(continuous.accepted_steps, segmented.accepted_steps)


class BertinBCCGateATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parameters = BertinBCCParameters()
        cls.stable = {}
        for label, axis, rate, target, family in (
            ("001_compression", (0, 0, 1), -2.0e8, -1.0, "001"),
            ("111_compression", (1, 1, 1), -2.0e8, -1.0, "111"),
            ("101_tension", (1, 0, 1), 2.0e8, 1.0, "101"),
        ):
            initial = make_state(axis)
            final, history = advance_bertin_bcc(
                initial, rate, target, 0.001, 300.0, cls.parameters, sample_stride=100
            )
            response = evaluate_bertin_bcc(final, 300.0, cls.parameters)
            cls.stable[label] = (final, history, response, family)

        cls.initial_419 = make_state((4, 1, 9), perturbation_deg=1.0)
        cls.final_419_c, cls.history_419_c = advance_bertin_bcc(
            cls.initial_419, -2.0e8, -1.0, 0.001, 300.0, cls.parameters, sample_stride=100
        )
        cls.final_419_t, cls.history_419_t = advance_bertin_bcc(
            cls.initial_419, 2.0e8, 1.0, 0.001, 300.0, cls.parameters, sample_stride=100
        )

        cls.initial_111_t = make_state((1, 1, 1), perturbation_deg=1.0)
        cls.full_111_t, cls.full_history = advance_bertin_bcc(
            cls.initial_111_t, 2.0e8, 1.0, 0.001, 300.0, cls.parameters, sample_stride=50
        )
        cls.no_spin_111_t, _ = advance_bertin_bcc(
            cls.initial_111_t,
            2.0e8,
            1.0,
            0.001,
            300.0,
            replace(cls.parameters, plastic_spin_scale=0.0),
            sample_stride=25,
        )
        cls.no_relax_111_t, _ = advance_bertin_bcc(
            cls.initial_111_t,
            2.0e8,
            1.0,
            0.001,
            300.0,
            replace(cls.parameters, inactive_relaxation_scale=0.0),
            sample_stride=25,
        )

    def test_published_stable_orientations_remain_stable(self):
        for label, (_, _, response, family) in self.stable.items():
            with self.subTest(label=label):
                self.assertLess(cubic_family_angle_deg(response.loading_axis_crystal, family), 1.0)

    def test_419_rotates_toward_published_attractors(self):
        initial = evaluate_bertin_bcc(self.initial_419, 300.0, self.parameters)
        compression = evaluate_bertin_bcc(self.final_419_c, 300.0, self.parameters)
        tension = evaluate_bertin_bcc(self.final_419_t, 300.0, self.parameters)
        self.assertLess(
            cubic_family_angle_deg(compression.loading_axis_crystal, "111"),
            cubic_family_angle_deg(initial.loading_axis_crystal, "111") - 10.0,
        )
        self.assertLess(
            cubic_family_angle_deg(tension.loading_axis_crystal, "101"),
            cubic_family_angle_deg(initial.loading_axis_crystal, "101") - 5.0,
        )

    def test_111_tension_rotation_requires_plastic_spin(self):
        initial = evaluate_bertin_bcc(self.initial_111_t, 300.0, self.parameters)
        full = evaluate_bertin_bcc(self.full_111_t, 300.0, self.parameters)
        no_spin = evaluate_bertin_bcc(
            self.no_spin_111_t, 300.0, replace(self.parameters, plastic_spin_scale=0.0)
        )
        initial_angle = cubic_family_angle_deg(initial.loading_axis_crystal, "101")
        self.assertLess(cubic_family_angle_deg(full.loading_axis_crystal, "101"), initial_angle - 5.0)
        self.assertGreater(
            cubic_family_angle_deg(no_spin.loading_axis_crystal, "101"),
            cubic_family_angle_deg(full.loading_axis_crystal, "101") + 5.0,
        )

    def test_inactive_family_is_removed_and_active_families_redistribute(self):
        self.assertLess(self.full_111_t.densities_m2[0], 0.25 * INITIAL_DENSITIES[0])
        self.assertGreater(self.no_relax_111_t.densities_m2[0], 5.0 * self.full_111_t.densities_m2[0])
        self.assertGreater(float(np.max(self.full_111_t.densities_m2[1:])), 15.0 * INITIAL_DENSITIES[0])

    def test_orientation_and_density_both_change_stress(self):
        full = evaluate_bertin_bcc(self.full_111_t, 300.0, self.parameters)
        no_spin = evaluate_bertin_bcc(
            self.no_spin_111_t, 300.0, replace(self.parameters, plastic_spin_scale=0.0)
        )
        no_relax = evaluate_bertin_bcc(
            self.no_relax_111_t, 300.0, replace(self.parameters, inactive_relaxation_scale=0.0)
        )
        self.assertGreater(abs(full.cauchy_stress_Pa[2, 2] - no_spin.cauchy_stress_Pa[2, 2]), 10.0e6)
        self.assertGreater(abs(full.cauchy_stress_Pa[2, 2] - no_relax.cauchy_stress_Pa[2, 2]), 5.0e6)
        peak = max(item["axial_cauchy_stress_Pa"] for item in self.full_history)
        self.assertLess(full.cauchy_stress_Pa[2, 2], peak - 50.0e6)

    def test_final_timestep_refinement_is_below_five_percent(self):
        coarse, _ = advance_bertin_bcc(
            self.initial_419, -2.0e8, -0.5, 0.001, 300.0, self.parameters, sample_stride=100
        )
        fine, _ = advance_bertin_bcc(
            self.initial_419, -2.0e8, -0.5, 0.0005, 300.0, self.parameters, sample_stride=200
        )
        coarse_response = evaluate_bertin_bcc(coarse, 300.0, self.parameters)
        fine_response = evaluate_bertin_bcc(fine, 300.0, self.parameters)
        stress_change = abs(coarse_response.cauchy_stress_Pa[2, 2] - fine_response.cauchy_stress_Pa[2, 2]) / abs(fine_response.cauchy_stress_Pa[2, 2])
        density_change = abs(np.sum(coarse.densities_m2) - np.sum(fine.densities_m2)) / np.sum(fine.densities_m2)
        angle_change = abs(
            cubic_family_angle_deg(coarse_response.loading_axis_crystal, "111")
            - cubic_family_angle_deg(fine_response.loading_axis_crystal, "111")
        ) / max(cubic_family_angle_deg(fine_response.loading_axis_crystal, "111"), 1.0)
        self.assertLess(max(stress_change, density_change, angle_change), 0.05)


if __name__ == "__main__":
    unittest.main()
