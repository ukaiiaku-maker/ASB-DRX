import importlib.util
import math
from pathlib import Path
import unittest
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "full_model/production/arrhenius_kinetics.py"
SPEC = importlib.util.spec_from_file_location("full_arrhenius", MODULE_PATH)
KIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(KIN)


class FullModelArrheniusTest(unittest.TestCase):
    def test_zero_entropy_is_exact_legacy_rate(self):
        h = KIN.exp_floor_enthalpy_j(650e6, 1.9 * KIN.EV_J, 1.5e9, 2.2, 2.5, 0.1)
        process = KIN.ActivatedProcess("glide", 1.0e12, entropy_over_kB=0.0)
        expected = 1.0e12 * math.exp(-h / (KIN.KB_J_K * 1100.0))
        self.assertEqual(KIN.activated_rate_s(process, h, 1100.0), expected)

    def test_signed_entropy_enters_once(self):
        h = 1.2 * KIN.EV_J
        zero = KIN.activated_rate_s(KIN.ActivatedProcess("x", 1e9), h, 1000.0)
        positive = KIN.activated_rate_s(
            KIN.ActivatedProcess("x", 1e9, entropy_over_kB=2.0), h, 1000.0)
        negative = KIN.activated_rate_s(
            KIN.ActivatedProcess("x", 1e9, entropy_over_kB=-2.0), h, 1000.0)
        self.assertAlmostEqual(positive / zero, math.exp(2.0), places=12)
        self.assertAlmostEqual(negative / zero, math.exp(-2.0), places=12)

    def test_attempt_entropy_identifiable_product(self):
        p = KIN.ActivatedProcess("x", 2.5e7, entropy_over_kB=-1.25)
        self.assertEqual(p.identifiable_prefactor_s, 2.5e7 * math.exp(-1.25))

    def test_negative_barrier_has_explicit_drag_or_rejection(self):
        drag = KIN.ActivatedProcess("x", 1e9, entropy_over_kB=20.0, drag_rate_s=2e8)
        self.assertEqual(KIN.activated_rate_s(drag, 0.01 * KIN.EV_J, 1500.0), 2e8)
        reject = KIN.ActivatedProcess(
            "x", 1e9, entropy_over_kB=20.0, negative_barrier_mode="reject")
        with self.assertRaises(ValueError):
            KIN.activated_rate_s(reject, 0.01 * KIN.EV_J, 1500.0)

    def test_event_frequency_and_velocity_are_distinct(self):
        self.assertEqual(KIN.event_velocity_m_s(2.0e6, 2.5e-10), 5.0e-4)

    def test_production_exponent_restriction(self):
        with self.assertRaises(ValueError):
            KIN.exp_floor_enthalpy_j(1.0, 1.0, 1.0, 1.0, 0.5, 0.0)

    def test_exp_floor_accepts_spatial_driving_field(self):
        stress = np.array([0.0, 1e8, 2e8])
        value = KIN.exp_floor_enthalpy_j(stress, KIN.EV_J, 1e8, 1.0, 1.0, 0.05)
        self.assertEqual(value.shape, stress.shape)
        self.assertTrue(np.all(np.diff(value) < 0.0))

    def test_invalid_process_units_and_modes_are_rejected(self):
        with self.assertRaises(ValueError):
            KIN.ActivatedProcess("x", 0.0)
        with self.assertRaises(ValueError):
            KIN.ActivatedProcess("x", 1.0, negative_barrier_mode="clip")
        with self.assertRaises(ValueError):
            KIN.free_barrier_j(-1.0, 1000.0, 0.0)


class CandidateRestartSourceTest(unittest.TestCase):
    def test_checkpoint_writes_and_loads_every_candidate_array(self):
        source = (ROOT / "full_model/production/drx_full_v34_recovery.py").read_text()
        for name in (
            "nuc_cand_active", "nuc_cand_age",
            "nuc_cand_best_barrier", "nuc_cand_birth_step",
        ):
            self.assertIn(f"{name}={name}", source)
            self.assertIn(f"'{name}'", source)
        self.assertIn("legacy checkpoint omits candidate state", source)
        self.assertIn("_atomic_savez_compressed", source)
        self.assertIn("checkpoint collective activity memory shape mismatch", source)
        self.assertIn("_restart_step_offset", source)
        self.assertIn("_potential_checkpoint_state", source)
        self.assertIn("legacy checkpoint omits numerical Arrhenius-potential state", source)
        for name in ("nuc_raw_trigger_total", "nuc_raw_viable_trigger_total"):
            self.assertIn(f"{name}=np.array({name}", source)
            self.assertIn(f"'{name}'", source)
        self.assertIn("embryo_population_json=np.array(population_to_json(embryo_population))", source)
        self.assertIn("exact stateful-embryo restart requires embryo_population_json", source)
        self.assertIn("physical_grain_tracker_json=np.array(tracker_to_json(grain_tracker))", source)
        self.assertIn("exact stateful-embryo restart requires physical_grain_tracker_json", source)
        self.assertIn("exact area-integrated hazard restart requires complete hazard state", source)
        self.assertIn("rng_hazard_measure_state_json=np.array", source)
        self.assertIn("hazard_exposure_total", source)
        self.assertIn("use_expf_embryo_creation", source)
        self.assertIn("creation_enthalpy = exp_floor_enthalpy_j", source)
        self.assertIn("embryo_creation_route='precursor'", source)
        self.assertIn("classical_critical_R=best_R", source)
        self.assertIn("candidate_kinetic_barrier = (fields['creation_free_barrier']", source)

    def test_local_trajectory_fixture_is_exact(self):
        result = __import__("json").loads(
            (ROOT / "full_model/verification/patch_A_restart_local.json").read_text())
        self.assertTrue(result["zero_entropy_reference_comparison"]["bitwise_exact"])
        self.assertTrue(result["restart_comparison"]["bitwise_exact"])
        self.assertTrue(result["restart_comparison"]["atomic_checkpoint_publish"])


if __name__ == "__main__":
    unittest.main()
