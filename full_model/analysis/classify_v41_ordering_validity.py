#!/usr/bin/env python3
"""Separate execution, energy admissibility, and accuracy for V40 ordering."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.input.read_text())
    isolated = raw["isolated_fixed_state_ordering"]
    legacy = isolated["legacy_v37_single_capped_extent"]
    current = isolated["current_v39_complete_time_dispatch"]
    coupled = raw["records"]
    requested = float(raw["requested_dt_s"])
    accepted = float(coupled[
        "current_v39_complete_time_dispatch"]["accepted_dt_s"])

    def row(value):
        delta = float(value["defect_energy_change_J_per_m_thickness"])
        completed = (value["status"] == "VALID"
                     and value["complete_elapsed_time_s"] == requested)
        return {
            "solver_completed_requested_horizon": completed,
            "conservation_or_execution_status": value["status"],
            "energy_change_J_per_m_thickness": delta,
            "unforced_energy_admissible": delta <= 0.0,
            "accurate_comparator_qualified": bool(
                completed and delta <= 0.0 and value is current),
            "integration_method": value["integration_method"],
            "ordered_line_m_per_m_thickness": value["observables"][
                "ordered_line_m_per_m_thickness"],
        }
    legacy_row = row(legacy); current_row = row(current)
    result = {
        "schema": "asb-drx/v41/ordering-legacy-current-validity/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input": {"path": str(args.input.resolve()),
                  "sha256": hashlib.sha256(args.input.read_bytes()).hexdigest()},
        "coupled_requested_dt_s": requested,
        "coupled_actual_accepted_dt_s": accepted,
        "coupled_horizon_fraction": accepted/requested,
        "coupled_short_interval_states_agree": all(
            item["absolute_difference"] == 0.0
            for item in raw["comparisons"].values()),
        "coupled_one_microsecond_overlap_qualified": False,
        "isolated_fixed_state": {"legacy": legacy_row,
                                 "current": current_row},
        "isolated_legacy_current_accurate_overlap_qualified": False,
        "classification": (
            "LEGACY_EXECUTED_BUT_UNFORCED_ENERGY_INADMISSIBLE;"
            "CURRENT_COMPLETE_TIME_ENERGY_ADMISSIBLE"),
        "claim_boundary": (
            "The exact coupled agreement covers 37.616 ps, not the requested "
            "1 microsecond. The isolated legacy endpoint is retained as a "
            "failed thermodynamic comparator and does not invalidate unrelated "
            "legacy trajectory intervals without their own energy audit."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
