#!/usr/bin/env python3
"""Stage/source-indexed V45 comparisons without ambiguous member labels."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from full_model.analysis.run_v43_spatial_first_difference import compare_field, fields


ROOT = Path("full_model/production/results-local")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def member(name, path, grid, operator, substeps):
    values, metadata, context = fields(path, grid)
    return values, {
        "member": name, "path": str(Path(path).resolve()),
        "sha256": digest(path), "source_sha": metadata["source_sha"],
        "operator": operator, "grid": grid,
        "domain_length_m": float(context["spacing_m"])*grid,
        "spacing_m": float(context["spacing_m"]),
        "interface_width_m": 4e-7,
        "macro_index": 1, "substage": "after_second_mura",
        "physical_time_s": float(metadata["physical_time_s"]),
        "macro_dt_s": float(metadata["macro_dt_s"]),
        "requested_substeps_per_half": substeps,
        "accepted_second_half_substeps_s": (
            metadata.get("v45_resume", {}).get("accepted_substeps_s")
            or metadata.get("accepted_dt_s")),
        "start_checkpoint": metadata.get("v45_resume", {}).get("parent_path"),
    }


def spectral(a, b, name_a, name_b):
    raw = compare_field(a, b)
    return {
        "fourier_mask": {
            "fft_normalization": "complex coefficient divided by n^2",
            "centered_common_mode_half_width": 24,
            "mode_count_per_axis": 49,
            "active_phase_threshold_relative_to_pair_max": 1e-8,
        },
        "complex_coefficient_relative_rms": raw[
            "complex_coefficient_relative_rms"],
        "power_relative_rms": raw["power_relative_rms"],
        "active_coefficient_phase_rms_rad": raw[
            "active_coefficient_phase_rms_rad"],
        f"{name_a}_whole_field_rms": raw["n128_rms"],
        f"{name_b}_whole_field_rms": raw["n192_rms"],
        "whole_field_rms_relative_difference": raw[
            "rms_relative_difference"],
    }


def pair(name, left, right, *, same_grid, interpretation):
    a, ma = left; b, mb = right
    comparisons = {}
    for field in a:
        row = spectral(a[field], b[field], ma["member"], mb["member"])
        if same_grid:
            difference = np.asarray(a[field])-np.asarray(b[field])
            scale = max(float(np.sqrt(np.mean(np.asarray(a[field])**2))),
                        float(np.sqrt(np.mean(np.asarray(b[field])**2))), 1e-300)
            row["strong_norms"] = {
                "difference_rms": float(np.sqrt(np.mean(difference**2))),
                "difference_rms_relative": float(
                    np.sqrt(np.mean(difference**2))/scale),
                "difference_maximum_absolute": float(np.max(np.abs(difference))),
                "member_a_minimum": float(np.min(a[field])),
                "member_a_maximum": float(np.max(a[field])),
                "member_b_minimum": float(np.min(b[field])),
                "member_b_maximum": float(np.max(b[field])),
                "integral_difference_grid_units": float(np.sum(difference)),
            }
        comparisons[field] = row
    return {
        "comparison": name, "member_a": ma, "member_b": mb,
        "same_grid": same_grid, "interpretation": interpretation,
        "fields": comparisons,
    }


def main():
    compatible = ROOT/"v44-compatible"
    one128 = member(
        "v44_one_n128", compatible/"one/n128/after_second_mura.npz", 128,
        "compatible_dealiased", 1)
    one192 = member(
        "v44_one_n192", compatible/"one/n192/after_second_mura.npz", 192,
        "compatible_dealiased", 1)
    sub192 = member(
        "v44_sub4_n192", compatible/"sub4/n192/after_second_mura.npz", 192,
        "compatible_dealiased", 4)
    resumed128 = member(
        "v45_resumed_sub4_n128",
        ROOT/"v45-resumed/sub4/n128/after_second_mura.npz", 128,
        "compatible_dealiased", 4)
    v43 = json.loads(Path(
        "full_model/verification/v43_spatial_first_difference_and_repair.json"
    ).read_text())
    v44 = json.loads(Path(
        "full_model/verification/v44_compatible_transport_decision.json"
    ).read_text())
    payload = {
        "schema": "asb-drx/v45/matched-accuracy/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "postprocessor_source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "claim_scope": (
            "COMPATIBLE_TRANSPORT_IMPLEMENTED_MATCHED_SPATIAL_IMPROVEMENT_"
            "UNESTABLISHED"),
        "matched_v43_v44_first_macro": {
            stage: {
                "v43": v43["first_macro_baseline"][stage]["curl_nye"],
                "v44": v44["cross_grid"]["one"][stage]["curl_nye"],
            } for stage in ("after_first_mura", "after_second_mura")
        },
        "comparisons": [
            pair("n128_same_grid_one_vs_four_substeps", one128, resumed128,
                 same_grid=True,
                 interpretation=(
                     "temporal comparison; four-substep endpoint is an "
                     "attributable retained-state continuation under V45")),
            pair("n192_same_grid_one_vs_four_substeps", one192, sub192,
                 same_grid=True,
                 interpretation="V44 source-frozen temporal comparison"),
            pair("sub4_n128_vs_n192_final", resumed128, sub192,
                 same_grid=False,
                 interpretation=(
                     "spatial comparison completing the retained V44 cohort; "
                     "not a full analytic-start current-source trajectory")),
        ],
        "selected_threshold": 0.05,
        "full_current_source_trajectory_regenerated": False,
        "drx_claimed": False, "material_calibration_claimed": False,
    }
    temporal = payload["comparisons"][0]["fields"]["curl_nye"]
    spatial = payload["comparisons"][2]["fields"]["curl_nye"]
    payload["classification"] = {
        "retained_endpoint_complete": True,
        "n128_temporal_threshold_passed": bool(
            temporal["complex_coefficient_relative_rms"] <= .05 and
            temporal["whole_field_rms_relative_difference"] <= .05),
        "hybrid_source_spatial_threshold_passed": bool(
            spatial["complex_coefficient_relative_rms"] <= .05 and
            spatial["whole_field_rms_relative_difference"] <= .05),
        "current_source_solution_accuracy_promoted": False,
    }
    output = Path("full_model/verification/v45_matched_accuracy.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps({
        "sha256": digest(output), **payload["classification"]}, sort_keys=True))


if __name__ == "__main__":
    main()
