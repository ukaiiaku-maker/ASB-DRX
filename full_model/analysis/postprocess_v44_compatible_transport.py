#!/usr/bin/env python3
"""Decision record for V44 spatial and temporal compatible transport tests."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from full_model.analysis.run_v43_spatial_first_difference import (
    STAGES, compare_field, fields,
)


ROOT = Path("full_model/production/results-local/v44-compatible")
OUT = Path("full_model/verification/v44_compatible_transport_decision.json")


def cross_grid(mode):
    result = {}
    for stage in STAGES:
        a, _, _ = fields(ROOT/mode/"n128"/f"{stage}.npz", 128)
        b, _, _ = fields(ROOT/mode/"n192"/f"{stage}.npz", 192)
        result[stage] = {name: compare_field(a[name], b[name]) for name in a}
    return result


def same_grid_temporal(grid):
    result = {}
    for stage in STAGES:
        one, _, _ = fields(ROOT/"one"/f"n{grid}"/f"{stage}.npz", grid)
        four, _, _ = fields(ROOT/"sub4"/f"n{grid}"/f"{stage}.npz", grid)
        result[stage] = {name: compare_field(one[name], four[name]) for name in one}
    return result


def main():
    cross = {mode: cross_grid(mode) for mode in ("one", "sub4")}
    temporal = {str(n): same_grid_temporal(n) for n in (128, 192)}
    final = cross["sub4"]["after_second_mura"]["curl_nye"]
    time_final = {n: temporal[str(n)]["after_second_mura"]["curl_nye"]
                  for n in (128, 192)}
    spatial_pass = bool(
        final["complex_coefficient_relative_rms"] <= .05
        and final["rms_relative_difference"] <= .05)
    temporal_pass = bool(all(
        row["complex_coefficient_relative_rms"] <= .05
        and row["rms_relative_difference"] <= .05
        for row in time_final.values()))
    ordered = {str(n): {
        "one_m": float(fields(ROOT/"one"/f"n{n}"/"after_first_mura.npz", n)[0][
            "ordered_density"].sum()*(3.2e-6/n)**2),
        "sub4_m": float(fields(ROOT/"sub4"/f"n{n}"/"after_first_mura.npz", n)[0][
            "ordered_density"].sum()*(3.2e-6/n)**2),
    } for n in (128, 192)}
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    payload = {
        "schema": "asb-drx/v44/compatible-transport-decision/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha_at_postprocessing": source,
        "trajectory_source_sha": json.load(open(
            ROOT/"one"/"n128"/"stage_run.json"))["source_sha"],
        "operator": "v44_compatible_dealiased_mura_transport_and_capture",
        "cross_grid": cross, "same_grid_temporal": temporal,
        "ordered_line_first_generation": ordered,
        "provisional_threshold": .05,
        "spatial_threshold_passed": spatial_pass,
        "temporal_threshold_passed": temporal_pass,
        "classification": (
            "COMPATIBLE_TRANSPORT_SPATIAL_AND_TEMPORAL_THRESHOLDS_PASSED"
            if spatial_pass and temporal_pass else
            "COMPATIBLE_TRANSPORT_REMAINS_UNRESOLVED"),
        "ordered_residual_classification": (
            "SEPARATE_FIRST_OPERATION_NEAR_EXTINCTION_RESIDUAL_NOT_WALL_SIGNAL"),
        "drx_claimed": False, "strict_asb_claimed": False,
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"classification": payload["classification"],
                      "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()},
                     sort_keys=True))


if __name__ == "__main__":
    main()
