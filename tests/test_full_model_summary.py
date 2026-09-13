import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from full_model.analysis.summarize_full_v34_case import summarize


class FullModelSummaryTest(unittest.TestCase):
    def test_zero_event_chain_stops_at_hazard_exposure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fields = [
                "step", "t_us", "eps_pct", "sigma_MPa", "T_max", "asb_T_range",
                "asb_gdot_top5_frac", "asb_qdot_top5_frac", "asb_band_anisotropy_T",
                "asb_rho_hot_over_cold", "nuc_candidates", "nuc_hazard_max",
                "nuc_Hmax", "nuc_candidate_new", "nuc_candidate_active",
                "nuc_candidate_promotable", "grain_hazard_births", "n_grains",
            ]
            with (root / "drx_v25_restart_asb_diagnostics.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(dict.fromkeys(fields, 0) | {
                    "step": 9, "t_us": 10, "eps_pct": 50, "sigma_MPa": 100,
                    "T_max": 1101, "asb_T_range": 1, "asb_gdot_top5_frac": 0.05,
                    "asb_qdot_top5_frac": 0.05, "asb_band_anisotropy_T": 1,
                    "asb_rho_hot_over_cold": 1, "nuc_candidates": 12,
                    "nuc_hazard_max": 0.1, "nuc_Hmax": 1e-6, "n_grains": 12,
                })
            np.savez_compressed(
                root / "drx_v25_restart_000009.npz",
                H_nuc=np.array([[1e-6]]), E_nuc=np.array([[0.1]]),
                nuc_cand_active=np.array([[False]]), nuc_cand_age=np.array([[0]]),
                nuc_cand_best_barrier=np.array([[np.inf]]), step=np.array(9),
                sim_time=np.array(1e-5),
            )
            result = summarize(root, "drx_isothermal")
            self.assertEqual(
                result["candidate_to_grain_chain"]["first_failing_stage"],
                "hazard_exposure_or_unrecorded_raw_trigger_requires_explicit_counter",
            )
        self.assertAlmostEqual(result["physical_horizon"]["final_strain"], 0.5)
        self.assertTrue(result["asb_classification"].startswith("NOT_EVALUATED"))

    def test_physical_grain_closes_chain_and_uses_persisted_event_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fields = [
                "step", "t_us", "eps_pct", "sigma_MPa", "T_max", "asb_T_range",
                "asb_gdot_top5_frac", "asb_qdot_top5_frac", "asb_band_anisotropy_T",
                "asb_rho_hot_over_cold", "nuc_candidates", "nuc_hazard_max",
                "nuc_Hmax", "nuc_candidate_new", "nuc_candidate_active",
                "nuc_candidate_promotable", "grain_hazard_births", "n_grains",
                "hazard_exposure_total", "nuc_raw_trigger_total",
                "nuc_raw_viable_trigger_total", "embryo_records_total",
                "embryo_active_total", "embryo_promotable_total",
                "atomic_promotion_attempt_total", "atomic_promotion_commit_total",
                "atomic_promotion_rollback_total", "physical_drx_grains",
                "physical_recrystallized_area_fraction",
            ]
            with (root / "drx_v25_restart_asb_diagnostics.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(dict.fromkeys(fields, 0) | {
                    "step": 9, "t_us": 10, "eps_pct": 50, "sigma_MPa": 100,
                    "T_max": 1101, "nuc_candidates": 2, "nuc_hazard_max": 1,
                    "nuc_Hmax": 1, "nuc_candidate_new": 1,
                    "grain_hazard_births": 1, "n_grains": 4,
                    "hazard_exposure_total": 1, "nuc_raw_trigger_total": 1,
                    "nuc_raw_viable_trigger_total": 1, "embryo_records_total": 1,
                    "atomic_promotion_attempt_total": 1,
                    "atomic_promotion_commit_total": 1, "physical_drx_grains": 1,
                    "physical_recrystallized_area_fraction": 0.2,
                })
            event = {"step": 7, "heat_released_J": 2.0e-14, "energy_closure_J": 0.0}
            grain = {"label": 3, "embryo_promoted": True, "status": "recrystallized"}
            np.savez_compressed(
                root / "drx_v25_restart_000009.npz",
                step=np.array(9), sim_time=np.array(1e-5),
                atomic_promotion_events_json=np.array(json.dumps([event])),
                physical_grain_tracker_json=np.array(json.dumps({"records": [grain]})),
            )
            result = summarize(root, "drx_isothermal")
        self.assertIsNone(result["candidate_to_grain_chain"]["first_failing_stage"])
        promotion = result["embryo_and_atomic_promotion"]
        self.assertEqual(promotion["committed_event_ledgers"], [event])
        self.assertEqual(promotion["promoted_grain_records"], [grain])


if __name__ == "__main__":
    unittest.main()
