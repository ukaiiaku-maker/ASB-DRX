import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FullModelRecoveryTest(unittest.TestCase):
    def test_reference_hashes(self):
        manifest = json.loads((ROOT / "full_model/provenance_manifest.json").read_text())
        for record in manifest["sources"].values():
            path = ROOT / "full_model" / record["path"]
            self.assertTrue(path.is_file(), path)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"])
        for record in manifest["archived_configurations"].values():
            path = ROOT / "full_model" / record["path"]
            self.assertTrue(path.is_file(), path)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"])

    def test_reference_drivers_compile(self):
        for path in [
            ROOT / "full_model/reference_sources/v32/drx_var_v32_gb_transmission_processzone_asb_sweep.py",
            ROOT / "full_model/reference_sources/v34/drx_var_v34_candidate_drx_asb_sweep.py",
        ]:
            compile(path.read_text(), str(path), "exec")

    def test_baseline_stage_is_not_reinterpreted(self):
        audit = json.loads((ROOT / "full_model/reference_baseline_audit.json").read_text())
        v34 = audit["v34"]
        self.assertGreater(v34["max_possible_sites"], 0)
        self.assertGreater(v34["max_hazard_rate_s-1"], 0.0)
        self.assertLess(v34["max_hazard_threshold_ratio"], 1.0)
        self.assertEqual(v34["raw_stochastic_event_attempts"], 0)
        self.assertEqual(v34["candidate_creations"], 0)
        self.assertEqual(v34["physical_grains_created"], 0)
        self.assertEqual(
            v34["first_failing_stage"],
            "hazard_integration_did_not_reach_stochastic_threshold",
        )


if __name__ == "__main__":
    unittest.main()
