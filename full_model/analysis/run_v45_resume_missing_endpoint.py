#!/usr/bin/env python3
"""Resume only the missing V44 n128 post-front Mura half with V45 ordering."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

from full_model.analysis.run_v34_finite_coupled_response import resolved_bicrystal
from full_model.analysis.run_v36_recurrent_physical_response import driving_at_time
from full_model.analysis.run_v39_common_horizon import load_stage, mura_to_time, save_stage


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source_path = Path(
        "full_model/production/results-local/v44-compatible/sub4/n128/after_front.npz")
    output_root = Path(
        "full_model/production/results-local/v45-resumed/sub4/n128")
    output_root.mkdir(parents=True, exist_ok=True)
    context = resolved_bicrystal(
        grid=128, length_m=3.2e-6, interface_width_m=4e-7,
        temperature_K=1100.0, child_line_fraction=.35)
    state, parent = load_stage(source_path, context)
    if parent["stage"] != "after_front":
        raise ValueError("V45 resume requires the retained after-front state")
    macro_dt = float(parent["macro_dt_s"])
    physical_time = float(parent["physical_time_s"])
    driving = driving_at_time(128, .01, "hold", 0.0, physical_time)
    started = time.perf_counter()
    state, audits, elapsed = mura_to_time(
        context, state, .5*macro_dt, driving,
        maximum_substep_s=.5*macro_dt/4,
        mura_transport_operator="compatible_dealiased")
    wall = time.perf_counter()-started
    output = output_root/"after_second_mura.npz"
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    metadata = {
        "source_sha": source, "stage": "after_second_mura",
        "grid": 128, "macro_dt_s": macro_dt,
        "completed_intervals": 1, "physical_time_s": physical_time,
        "v45_resume": {
            "parent_path": str(source_path.resolve()),
            "parent_sha256": digest(source_path),
            "parent_source_sha": parent["source_sha"],
            "front_repeated": False, "front_heat_redeposited": False,
            "operator_exposure_s": elapsed,
            "accepted_substeps_s": [float(row["accepted_dt_s"]) for row in audits],
            "ordering_methods": [row["ordering_integration_method"] for row in audits],
            "ordering_dispatches": [row["ordering_stiff_dispatch"] for row in audits],
            "ordering_active_degrees_of_freedom": [
                row.get("ordering_active_degrees_of_freedom") for row in audits],
            "ordering_dense_jacobian_bytes_avoided": [
                row.get("ordering_dense_jacobian_bytes_avoided") for row in audits],
            "ordering_linear_iterations": [
                row.get("ordering_linear_iterations") for row in audits],
            "ordering_nonlinear_iterations": [
                row.get("ordering_nonlinear_iterations") for row in audits],
            "wall_seconds": wall,
        },
        "records": [],
    }
    save_stage(output, state, context, metadata)
    record = {
        "schema": "asb-drx/v45/resumed-missing-endpoint/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": source, "output": str(output.resolve()),
        "output_sha256": digest(output), **metadata["v45_resume"],
    }
    report = Path("full_model/verification/v45_resumed_n128_endpoint.json")
    report.write_text(json.dumps(record, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"elapsed_s": elapsed, "wall_seconds": wall,
                      "sha256": digest(report)}, sort_keys=True))


if __name__ == "__main__":
    main()
