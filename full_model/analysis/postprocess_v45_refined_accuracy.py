#!/usr/bin/env python3
"""Current-source four/eight temporal and selected spatial V45 decision."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from full_model.analysis.postprocess_v45_matched_accuracy import member, pair


ROOT = Path("full_model/production/results-local/v45-current-final")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    members = {}
    for substeps in (4, 8):
        for grid in (128, 192):
            path = ROOT/f"sub{substeps}/n{grid}/after_second_mura.npz"
            if not path.is_file():
                raise FileNotFoundError(f"required refined endpoint missing: {path}")
            members[(substeps, grid)] = member(
                f"current_sub{substeps}_n{grid}", path, grid,
                "compatible_dealiased", substeps)
    comparisons = [
        pair("n128_sub4_vs_sub8", members[(4, 128)], members[(8, 128)],
             same_grid=True,
             interpretation="current-source temporal refinement"),
        pair("n192_sub4_vs_sub8", members[(4, 192)], members[(8, 192)],
             same_grid=True,
             interpretation="current-source temporal refinement"),
        pair("selected_sub8_n128_vs_n192", members[(8, 128)], members[(8, 192)],
             same_grid=False,
             interpretation="current-source selected spatial refinement"),
    ]
    threshold = .05
    decisions = {}
    for row in comparisons:
        curl = row["fields"]["curl_nye"]
        decisions[row["comparison"]] = {
            "complex_fourier_passed": bool(
                curl["complex_coefficient_relative_rms"] <= threshold),
            "whole_field_rms_passed": bool(
                curl["whole_field_rms_relative_difference"] <= threshold),
            "passed": bool(
                curl["complex_coefficient_relative_rms"] <= threshold and
                curl["whole_field_rms_relative_difference"] <= threshold),
        }
    payload = {
        "schema": "asb-drx/v45/refined-current-accuracy/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha_at_postprocessing": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "threshold": threshold, "comparisons": comparisons,
        "decisions": decisions,
        "temporal_accuracy_passed": bool(
            decisions["n128_sub4_vs_sub8"]["passed"] and
            decisions["n192_sub4_vs_sub8"]["passed"]),
        "spatial_accuracy_passed": decisions[
            "selected_sub8_n128_vs_n192"]["passed"],
        "selected_substeps_per_half": 8,
        "full_current_source_trajectory_regenerated": True,
        "drx_claimed": False, "strict_asb_claimed": False,
        "material_calibration_claimed": False,
    }
    payload["classification"] = (
        "CURRENT_SOURCE_TEMPORAL_AND_SPATIAL_THRESHOLDS_PASSED"
        if payload["temporal_accuracy_passed"] and payload[
            "spatial_accuracy_passed"] else
        "CURRENT_SOURCE_TEMPORAL_PASSED_SPATIAL_FAILED"
        if payload["temporal_accuracy_passed"] else
        "CURRENT_SOURCE_TEMPORAL_REFINEMENT_FAILED")
    output = Path("full_model/verification/v45_refined_current_accuracy.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps({
        "classification": payload["classification"],
        "sha256": digest(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
