#!/usr/bin/env python3
"""Matched-stage spatial accuracy for the atomic V46 trajectories."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from full_model.analysis.run_v43_spatial_first_difference import (
    coefficients, compare_field, fields, rms,
)


ROOT = Path("full_model/production/results-local/v46-checkpointed")
STAGES = {
    "initial": "operation_000_initial.npz",
    "after_first_mura": "operation_008_mura.npz",
    "after_front": "operation_009_front.npz",
    "after_second_mura": "operation_017_mura.npz",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def common_band_strong_error(left, right, half_width):
    a = coefficients(left, half_width)
    b = coefficients(right, half_width)
    scale = max(rms(a), rms(b), 1e-300)
    return {
        "half_width": half_width,
        "complex_difference_rms": rms(a-b),
        "complex_difference_relative_rms": rms(a-b)/scale,
        "n128_coefficient_rms": rms(a),
        "n192_coefficient_rms": rms(b),
        "semantics": (
            "strong L2 norm on the shared centered Fourier subspace; no "
            "interpolation or optimized shift"),
    }


def validate_manifest(grid):
    root = ROOT/f"n{grid}"
    manifest_path = root/"run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["terminal_state"] != "COMPLETE" or manifest[
            "completed_operation_count"] != 17:
        raise ValueError(f"n{grid} trajectory is incomplete")
    latest = Path(manifest["latest_checkpoint"])
    if digest(latest) != manifest["latest_checkpoint_sha256"]:
        raise ValueError(f"n{grid} latest checkpoint checksum mismatch")
    for path in STAGES.values():
        if not (root/path).is_file():
            raise FileNotFoundError(root/path)
    return manifest


def main():
    manifests = {n: validate_manifest(n) for n in (128, 192)}
    sources = {manifests[n]["configuration"]["source_sha"] for n in manifests}
    if len(sources) != 1:
        raise ValueError("matched grids do not share one scientific source")
    payload = {
        "schema": "asb-drx/v46/checkpointed-spatial-accuracy/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_source_sha": next(iter(sources)),
        "postprocessor_source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "domain_m": 3.2e-6, "selected_half_width": 24,
        "threshold": .05, "coordinate_shift_optimized": False,
        "members": {}, "stages": {},
    }
    for n, manifest in manifests.items():
        payload["members"][str(n)] = {
            "manifest": str((ROOT/f"n{n}"/"run_manifest.json").resolve()),
            "manifest_sha256": digest(ROOT/f"n{n}"/"run_manifest.json"),
            "terminal_state": manifest["terminal_state"],
            "completed_operation_count": manifest["completed_operation_count"],
            "physical_time_s": manifest["physical_time_s"],
            "latest_checkpoint_sha256": manifest["latest_checkpoint_sha256"],
        }
    retained = {}
    for stage, filename in STAGES.items():
        values = {}
        metadata = {}
        for n in (128, 192):
            values[n], metadata[n], _ = fields(ROOT/f"n{n}"/filename, n)
        rows = {}
        for name in values[128]:
            selected = compare_field(values[128][name], values[192][name], 24)
            rows[name] = {
                "selected_common_band": selected,
                "common_band_strong_norm": common_band_strong_error(
                    values[128][name], values[192][name], 63),
            }
        payload["stages"][stage] = {
            "physical_time_s": {str(n): metadata[n]["physical_time_s"]
                                for n in (128, 192)},
            "checkpoint_sha256": {str(n): digest(ROOT/f"n{n}"/filename)
                                  for n in (128, 192)},
            "fields": rows,
        }
        retained[stage] = values
    final = retained["after_second_mura"]
    payload["final_curl_nye_band_sweep"] = {
        str(half): compare_field(final[128]["curl_nye"],
                                 final[192]["curl_nye"], half)
        for half in (8, 16, 24, 32, 48, 63)}
    curl = payload["stages"]["after_second_mura"]["fields"]["curl_nye"]
    selected = curl["selected_common_band"]
    payload["classification"] = {
        "selected_band_passed": bool(
            selected["complex_coefficient_relative_rms"] <= .05),
        "whole_field_rms_amplitude_passed": bool(
            selected["rms_relative_difference"] <= .05),
        "common_band_63_strong_norm_passed": bool(
            curl["common_band_strong_norm"][
                "complex_difference_relative_rms"] <= .05),
        "selected_spatial_gate_passed": bool(
            selected["complex_coefficient_relative_rms"] <= .05
            and selected["rms_relative_difference"] <= .05),
        "original_62_5us_horizon_authorized": bool(
            selected["complex_coefficient_relative_rms"] <= .05
            and selected["rms_relative_difference"] <= .05),
    }
    payload["drx_claimed"] = False
    payload["strict_asb_claimed"] = False
    payload["material_calibration_claimed"] = False
    output = Path("full_model/verification/v46_checkpointed_spatial_accuracy.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps({
        "classification": payload["classification"],
        "selected_curl_error": selected["complex_coefficient_relative_rms"],
        "curl_rms_amplitude_error": selected["rms_relative_difference"],
        "common_band_63_curl_error": curl["common_band_strong_norm"][
            "complex_difference_relative_rms"],
        "sha256": digest(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
