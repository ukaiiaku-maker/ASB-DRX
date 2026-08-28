from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from asb_drx.collective_diagnostics import depin_count_diagnostics, native_audit_branching_diagnostics


class CollectiveDiagnosticsTests(unittest.TestCase):
    def test_depin_count_clustering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "counts.csv"
            path.write_text(
                "step,time_s,event,event_count\n"
                "1,0.1,depin_cross,1\n"
                "2,0.2,depin_cross,2\n"
                "10,1.0,depin_cross,1\n"
            )
            result = depin_count_diagnostics(path, cluster_windows=(0, 1))
            self.assertEqual(result["total_events"], 4)
            self.assertEqual(result["max_same_step_events"], 2)
            self.assertEqual(result["clusters_by_window_steps"]["1"]["cluster_count"], 2)

    def test_native_branching_response_and_hazard_crossing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl"
            rows = [
                self._row(1, 1, True, 10.0, 0.1, 0.9, 1.1, 1.0),
                self._row(1, 2, False, 10.0, 0.2, 0.0, 0.2, 1.0),
                self._row(1, 3, False, 10.0, 0.3, 0.0, 0.3, 1.0),
                self._row(2, 2, False, 20.0, 0.2, 0.0, 0.2, 1.0),
                self._row(2, 3, False, 10.0, 0.3, 0.0, 0.3, 1.0),
                self._row(3, 2, False, 20.0, 0.2, 0.0, 0.2, 1.0),
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            result = native_audit_branching_diagnostics(path)
            self.assertEqual(result["accepted_release_rows"], 1)
            self.assertEqual(result["threshold_crossing_valid_fraction"], 1.0)
            self.assertGreater(result["mean_centered_excess_probability_per_release"], 0.0)

    @staticmethod
    def _row(step, contact, accepted, rate, rdt, before, after, threshold):
        return {
            "step": step,
            "contact_id": contact,
            "accepted": accepted,
            "rate_s": rate,
            "Rdt": rdt,
            "accumulated_hazard_before": before,
            "accumulated_hazard_after": after,
            "threshold": threshold,
        }


if __name__ == "__main__":
    unittest.main()
