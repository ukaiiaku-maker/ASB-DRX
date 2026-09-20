#!/usr/bin/env python3
"""Validate and compare the completed V46 62.5 us atomic trajectories."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from full_model.analysis.postprocess_v46_checkpointed_accuracy import (
    common_band_strong_error,
)
from full_model.analysis.run_v43_spatial_first_difference import (
    compare_field, fields,
)


ROOT = Path("full_model/production/results-local/v46-original-horizon")
PARENT = Path("full_model/production/results-local/v46-checkpointed")
FINAL = "continuation_119_mura.npz"
EXPECTED_TIME_S = 62.5e-6


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(grid):
    root = ROOT/f"n{grid}"
    manifest = json.loads((root/"run_manifest.json").read_text())
    parent = json.loads((PARENT/f"n{grid}"/"run_manifest.json").read_text())
    if manifest["terminal_state"] != "COMPLETE":
        raise ValueError(f"n{grid} continuation is not complete")
    if manifest["completed_operation_count"] != 119:
        raise ValueError(f"n{grid} operation ledger is incomplete")
    if abs(manifest["physical_time_s"]-EXPECTED_TIME_S) > 1e-18:
        raise ValueError(f"n{grid} physical clock does not close")
    latest = root/FINAL
    if (Path(manifest["latest_checkpoint"]).resolve() != latest.resolve()
            or digest(latest) != manifest["latest_checkpoint_sha256"]):
        raise ValueError(f"n{grid} final checkpoint checksum mismatch")
    if digest(PARENT/f"n{grid}"/"operation_017_mura.npz") != manifest[
            "parent_checkpoint_sha256"]:
        raise ValueError(f"n{grid} parent checksum mismatch")
    operations = parent["operations"]+manifest["operations"]
    mura = [row for row in operations if row["kind"] == "mura"]
    front = [row for row in operations if row["kind"] == "front"]
    ordering = [row["ordering"] for row in mura]
    family_scales = [x for row in mura for x in row["family_event_scales"]]
    return manifest, {
        "total_operations": len(operations),
        "mura_operations": len(mura),
        "front_operations": len(front),
        "published_front_operations": sum(bool(row["published"]) for row in front),
        "minimum_event_scale": min(row["event_scale"] for row in mura),
        "minimum_family_event_scale": min(family_scales),
        "all_ordering_accessibility_checks_passed": all(
            row["asymptotic_state_accessibility_passed"] for row in ordering),
        "maximum_ordering_accessibility_violation": max(
            row["asymptotic_accessibility_maximum_violation"] for row in ordering),
        "maximum_ordering_projected_kkt_relative": max(
            row["asymptotic_projected_kkt_relative"] for row in ordering),
        "minimum_operator_wall_seconds": min(row["wall_seconds"] for row in operations),
        "maximum_operator_wall_seconds": max(row["wall_seconds"] for row in operations),
    }


def main():
    validated = {n: validate(n) for n in (128, 192)}
    manifests = {n: validated[n][0] for n in validated}
    sources = {manifests[n]["configuration"]["parent_scientific_source_sha"]
               for n in manifests}
    continuation_sources = {
        manifests[n]["configuration"]["continuation_source_sha"]
        for n in manifests}
    if len(sources) != 1 or len(continuation_sources) != 1:
        raise ValueError("matched grids do not share source provenance")
    values = {}
    metadata = {}
    for n in (128, 192):
        values[n], metadata[n], _ = fields(ROOT/f"n{n}"/FINAL, n)
    rows = {}
    for name in values[128]:
        rows[name] = {
            "selected_common_band": compare_field(
                values[128][name], values[192][name], 24),
            "common_band_strong_norm": common_band_strong_error(
                values[128][name], values[192][name], 63),
        }
    curl = rows["curl_nye"]
    selected = curl["selected_common_band"]
    classification = {
        "selected_band_passed": bool(
            selected["complex_coefficient_relative_rms"] <= .05),
        "whole_field_rms_amplitude_passed": bool(
            selected["rms_relative_difference"] <= .05),
        "common_band_63_strong_norm_passed": bool(
            curl["common_band_strong_norm"][
                "complex_difference_relative_rms"] <= .05),
    }
    classification["original_horizon_spatial_gate_passed"] = bool(
        classification["selected_band_passed"]
        and classification["whole_field_rms_amplitude_passed"])
    payload = {
        "schema": "asb-drx/v46/original-horizon-accuracy/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_source_sha": next(iter(sources)),
        "continuation_source_sha": next(iter(continuation_sources)),
        "postprocessor_source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "physical_time_s": EXPECTED_TIME_S,
        "threshold": .05,
        "selected_half_width": 24,
        "coordinate_shift_optimized": False,
        "members": {
            str(n): {
                "manifest": str((ROOT/f"n{n}"/"run_manifest.json").resolve()),
                "manifest_sha256": digest(ROOT/f"n{n}"/"run_manifest.json"),
                "latest_checkpoint_sha256": manifests[n]["latest_checkpoint_sha256"],
                "ledger_summary": validated[n][1],
            } for n in (128, 192)},
        "fields": rows,
        "curl_nye_band_sweep": {
            str(half): compare_field(values[128]["curl_nye"],
                                     values[192]["curl_nye"], half)
            for half in (8, 16, 24, 32, 48, 63)},
        "classification": classification,
        "drx_claimed": False,
        "lagb_claimed": False,
        "strict_asb_claimed": False,
        "material_calibration_claimed": False,
    }
    output = Path("full_model/verification/v46_original_horizon_accuracy.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps({
        "classification": classification,
        "selected_curl_error": selected["complex_coefficient_relative_rms"],
        "curl_rms_amplitude_error": selected["rms_relative_difference"],
        "common_band_63_curl_error": curl["common_band_strong_norm"][
            "complex_difference_relative_rms"],
        "sha256": digest(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
