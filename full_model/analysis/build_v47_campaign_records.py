#!/usr/bin/env python3
"""Build checksum-addressed V47 controller and campaign manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evidence(path):
    path = Path(path)
    return {"path": str(path), "sha256": digest(path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-passed", type=int, required=True)
    parser.add_argument("--regression-wall-seconds", type=float, required=True)
    args = parser.parse_args()
    generated = datetime.now(timezone.utc).isoformat()
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    short = json.loads(Path(
        "full_model/verification/v47_local_bundle_summary.json").read_text())
    geometry = json.loads(Path(
        "full_model/verification/v47_geometry_force.json").read_text())
    first = json.loads(Path(
        "full_model/verification/v47_ordering_production_dispatch_first_nonzero.json").read_text())
    late = json.loads(Path(
        "full_model/verification/v47_ordering_production_dispatch_late_macro.json").read_text())
    physical = json.loads(Path(
        "full_model/verification/v47_v46_physical_response_n128.json").read_text())
    pair = json.loads(Path(
        "full_model/verification/v47_current_source_physical_pair.json").read_text())
    evidence_paths = {
        "raw_spectral_union": "full_model/verification/v47_v46_raw_spectral_union.json",
        "geometry_force": "full_model/verification/v47_geometry_force.json",
        "ordering_first_nonzero": "full_model/verification/v47_ordering_first_nonzero.json",
        "ordering_post_front": "full_model/verification/v47_ordering_post_front.json",
        "ordering_late": "full_model/verification/v47_ordering_late_macro.json",
        "ordering_production_first": "full_model/verification/v47_ordering_production_dispatch_first_nonzero.json",
        "ordering_production_late": "full_model/verification/v47_ordering_production_dispatch_late_macro.json",
        "v46_physical_response": "full_model/verification/v47_v46_physical_response_n128.json",
        "current_source_physical_pair": "full_model/verification/v47_current_source_physical_pair.json",
        "local_bundle_summary": "full_model/verification/v47_local_bundle_summary.json",
    }
    regression = {
        "command": "PYTHONPATH=src:. pytest -q tests",
        "passed": args.tests_passed,
        "wall_seconds": args.regression_wall_seconds,
    }
    controller = {
        "schema": "asb-drx/v47/campaign-controller/v1",
        "generated_utc": generated, "source_sha": source,
        "execution_location": "local", "fixture_passed": True,
        "scientific_gate_passed": False,
        "classification": (
            "KINEMATICS_REFINED_ORDERING_REPAIRED_GEOMETRY_RATE_UNQUALIFIED"),
        "branches": {
            "raw_union": {
                "state": "PASSED_ACCOUNTING",
                "fine_tail_squared_norm_fraction": 0.021388190627261848,
                "union_relative_difference": 0.1591270290410718,
            },
            "fixed_scale_refinement": {
                "state": "MIXED_SCOPED",
                "classification": short["short_refinement"]["classification"],
                "curl_n192_n256_half63": short["short_refinement"][
                    "comparisons"]["n192_n256"]["half_width_63"]["curl_nye"],
                "ordered_n192_n256_half63": short["short_refinement"][
                    "comparisons"]["n192_n256"]["half_width_63"]["ordered_density"],
                "full_horizon_n256_completed": False,
            },
            "geometry_affinity": {
                "state": "FAILED_SCIENTIFIC",
                "classification": geometry["classification"],
                "force_fractional_extent_spread": geometry[
                    "n64_directional_force_fractional_extent_spread"],
                "rigid_translation_zero_force_passed": geometry[
                    "n64_complete_loop_rigid_translation"]["zero_self_force_passed"],
                "physical_chemical_work_enabled": True,
                "rate_qualified": geometry["rate_qualified"],
            },
            "ordering_finite_time": {
                "state": "PASSED_STATE_QUALIFIED",
                "first_dispatch": first["ordering_ledger"][
                    "ordering_stiff_dispatch"],
                "first_endpoint_distance_relative": first["ordering_ledger"][
                    "ordering_asymptotic_endpoint_distance_relative"],
                "late_dispatch": late["ordering_ledger"][
                    "ordering_stiff_dispatch"],
                "late_endpoint_distance_relative": late["ordering_ledger"][
                    "ordering_asymptotic_endpoint_distance_relative"],
            },
            "temporal_refinement": {
                "state": ("PASSED" if "PASSED" in short[
                    "temporal_refinement"]["classification"] else "FAILED_SCIENTIFIC"),
                "classification": short["temporal_refinement"]["classification"],
                "relative_differences": short["temporal_refinement"][
                    "relative_differences"],
            },
            "physical_response": {
                "state": "VALID_SCOPED",
                "v46_classification": physical["classification"],
                "contour_displacement_m": physical["front_response"][
                    "maximum_abs_component_contour_displacement_m"],
                "displacement_over_interface_width": physical[
                    "front_response"]["maximum_displacement_over_interface_width"],
                "pair_classification": pair["classification"],
            },
        },
        "evidence": {key: evidence(path) for key, path in evidence_paths.items()},
        "regression": regression,
        "drx_claimed": False, "lagb_claimed": False,
        "strict_asb_claimed": False, "material_calibration_claimed": False,
        "next_exact_task": (
            "derive an affinity-coupled geometry EXP-floor rate and resolve "
            "the n64-to-n128 active-increment sign change without changing "
            "the fixed 400 nm representation"),
    }
    controller_path = Path("full_model/verification/v47_campaign_controller.json")
    controller_path.write_text(json.dumps(controller, indent=2, sort_keys=True)+"\n")
    manifest = {
        "schema": "asb-drx/v47/campaign-manifest/v1",
        "generated_utc": generated, "repository": subprocess.check_output(
            ["git", "remote", "get-url", "origin"], text=True).strip(),
        "worktree": str(Path.cwd()),
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], text=True).strip(),
        "source_roles": {
            "v46_closing_source": "ea519f0284e2cfdecf8d14ab2d802d6279975657",
            "v46_scientific_source": "096159de1478bb259e445c15c11765d9149d7fe0",
            "v47_geometry_and_analysis_source": "bca1d9288bcdd9acfba688084a213b7def99b4c3",
            "v47_ordering_dispatch_source": "eb739b3faf00b37b2ddb5b5c5c9b2e62c6fa5699",
            "v47_evidence_source": source,
        },
        "configuration": {
            "domain_m": 3.2e-6, "representation_length_m": 4e-7,
            "geometry_fixture_section_thickness_m": 3.2e-6,
            "common_state_represented_thickness_m": 4.96e-10,
            "temperature_K": 1100.0,
            "short_refinement_grids": [128, 192, 256],
        },
        "evidence": {key: evidence(path) for key, path in evidence_paths.items()},
        "controller": evidence(controller_path),
        "local_checkpoint_records": short["checkpoint_records"],
        "regression": regression,
        "execution_policy": (
            "local_only; no HPC3 jobs submitted; unrelated DDD workers preserved"),
    }
    Path("full_model/verification/v47_campaign_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"controller_sha256": digest(controller_path),
                      "manifest_sha256": digest(
                          "full_model/verification/v47_campaign_manifest.json")},
                     sort_keys=True))


if __name__ == "__main__":
    main()
