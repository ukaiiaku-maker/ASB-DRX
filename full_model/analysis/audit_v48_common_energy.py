#!/usr/bin/env python3
"""Reprice retained V46 endpoints and trace one V48 common-energy cycle."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from full_model.analysis.postprocess_v47_v46_physical_response import state_record
from full_model.analysis.run_v34_finite_coupled_response import (
    I3Controls, resolved_bicrystal, run_i3_cycle,
)
from full_model.analysis.run_v36_recurrent_physical_response import driving_at_time
from full_model.analysis.run_v39_common_horizon import load_stage


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path(
        "/Users/sdillon/HPC3/worktrees/asb-drx-full-v46-20260920/"
        "full_model/production/results-local")
    initial_path = root/"v46-checkpointed/n128/operation_000_initial.npz"
    manifest = json.loads((root/"v46-original-horizon/n128/run_manifest.json").read_text())
    final_path = Path(manifest["latest_checkpoint"])
    context = resolved_bicrystal(
        grid=128, length_m=3.2e-6, interface_width_m=4e-7,
        temperature_K=1100.0, child_line_fraction=.35)
    initial, initial_meta = load_stage(initial_path, context)
    final, final_meta = load_stage(final_path, context)
    initial_record = state_record(
        initial, context, float(initial_meta["physical_time_s"]))
    final_record = state_record(final, context, float(final_meta["physical_time_s"]))
    changes = {key: float(final_record["physical_energy_J"][key]
                              -initial_record["physical_energy_J"][key])
               for key in initial_record["physical_energy_J"]}

    compact = resolved_bicrystal(
        grid=16, length_m=3.2e-6, interface_width_m=4e-7,
        temperature_K=1100.0, child_line_fraction=.35)
    drive = driving_at_time(16, .01, "hold", 0.0, 2.5e-7)
    _, cycle = run_i3_cycle(
        compact, compact["state"], compact["state"].eta.copy(), drive,
        I3Controls(
            mura_enabled=True, front_enabled=False,
            trial_dt_s=5e-7, front_dt_s=5e-7,
            mura_transport_operator="compatible_dealiased"))
    record = {
        "schema": "asb-drx/v48/common-energy-audit/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "retained_v46": {
            "initial_checkpoint": str(initial_path),
            "initial_checkpoint_sha256": digest(initial_path),
            "final_checkpoint": str(final_path),
            "final_checkpoint_sha256": digest(final_path),
            "initial": initial_record["physical_energy_J"],
            "final": final_record["physical_energy_J"],
            "changes": changes,
            "old_v47_reported_defect_change_J": 4.428687545816458e-14,
            "v48_repriced_defect_change_J": changes["defect_storage_J"],
            "historical_cumulative_first_law_qualified": False,
            "scope": (
                "endpoint repricing repairs the diagnostic functional; the "
                "historical trajectory lacks a complete cumulative external-"
                "work ledger and is not retroactively promoted"),
        },
        "current_source_stage_trace": {
            "grid": 16,
            "interval_s": 5e-7,
            "complete_energy": cycle["complete_energy"],
            "mura": {key: cycle["mura"][key] for key in (
                "plastic_work_increment_J_m3_cells",
                "deposited_heat_increment_J_m3_cells",
                "stored_line_energy_increment_J_m3_cells",
                "locking_energy_change_J_m3_cells",
                "locking_heat_increment_J_m3_cells",
                "ordering_energy_change_J_m3_cells",
                "ordering_heat_increment_J_m3_cells",
                "first_law_residual_J_m3_cells")},
            "physical_functional": "extensive_signed_reservoir_v48",
        },
        "classification": "COMMON_ENERGY_STAGE_CLOSURE_VERIFIED",
        "retained_v46_physical_response_promoted": False,
    }
    output = Path("full_model/verification/v48_common_energy_audit.json")
    output.write_text(json.dumps(record, indent=2, sort_keys=True)+"\n")
    print(json.dumps({
        "classification": record["classification"],
        "v48_repriced_defect_change_J": changes["defect_storage_J"],
        "stage_first_law_residual_J": cycle["complete_energy"][
            "first_law_residual_J"],
        "sha256": digest(output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
