#!/usr/bin/env python3
"""Assemble the compact V41 controller and case manifest from decisions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: Path) -> dict:
    return {"path": str(path), "sha256": digest(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verification-root", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--tests-passed", type=int, required=True)
    parser.add_argument("--test-wall-seconds", type=float, required=True)
    parser.add_argument("--one-grain-job-id", required=True)
    parser.add_argument("--one-grain-source-sha", required=True)
    parser.add_argument("--one-grain-run-root", required=True)
    parser.add_argument("--thermal-source-sha", required=True)
    args = parser.parse_args()
    root = args.verification_root
    paths = {
        "front": root/"v41_front_direction_energy_summary.json",
        "front_continuation": root/"v41_front_continuation_decision.json",
        "gradient": root/"v41_gradient_transfer_decision.json",
        "gradient_fields": root/"v41_gradient_field_audit.json",
        "ordering": root/"v41_ordering_legacy_current_validity.json",
        "one_grain": root/"v41_one_grain_decision.json",
        "one_grain_topology": root/"v41_one_grain_topology_pair.json",
        "thermal": root/"v41_thermal_response_decision.json",
    }
    data = {key: load(path) for key, path in paths.items()}
    now = datetime.now(timezone.utc).isoformat()
    one = data["one_grain"]
    thermal = data["thermal"]
    controller = {
        "schema": "asb-drx/v41/campaign-controller/v1",
        "generated_utc": now,
        "campaign": (
            "Direction-energy consistency, gradient transfer, and physical "
            "response campaign V41"),
        "canonical_branch": "exp/full-v34-recovery-v1",
        "final_source_sha": args.source_sha,
        "branches": {
            "front_direction_energy": {
                "state": "PASSED_OPERATOR",
                "classification": data["front"]["classification"],
                "zero_applied_front_work": True,
                "artifact": str(paths["front"]),
            },
            "repaired_front_continuation": {
                "state": "PHYSICAL_TERMINAL",
                "classification": data["front_continuation"]["classification"],
                "physical_time_s": data["front_continuation"]["physical_time_s"],
                "repaired_published_intervals": data["front_continuation"][
                    "repaired_published_intervals"],
                "artifact": str(paths["front_continuation"]),
            },
            "gradient_transfer": {
                "state": "PASSED_OPERATOR",
                "classification": data["gradient"]["classification"],
                "declared_identity_passed": data["gradient"][
                    "front_transfer_identity_passed"],
                "artifact": str(paths["gradient"]),
            },
            "wall_scale_spatial_accuracy": {
                "state": "FAILED_NUMERICAL",
                "classification": "SECOND_MURA_STRONG_NORM_REFINEMENT_UNRESOLVED",
                "cross_grid_strong_relative_difference": data["gradient"][
                    "second_mura_cross_grid_curl_nye_relative_difference"],
                "common_mode_relative_difference": data["gradient_fields"][
                    "common_mode_comparisons"]["post_second_mura"]["curl"][
                        "coefficient_relative_difference"],
                "artifact": str(paths["gradient_fields"]),
            },
            "ordering_comparator": {
                "state": "PASSED_OPERATOR",
                "classification": data["ordering"]["classification"],
                "artifact": str(paths["ordering"]),
            },
            "current_one_grain_organization": {
                "state": (
                    "PASSED_SCIENTIFIC" if one["scientific_gate_passed"]
                    else "FAILED_SCIENTIFIC"),
                "classification": one["classification"],
                "job_id": args.one_grain_job_id,
                "final_strain": one["final_strain"],
                "post_endpoint_diagnostic": one[
                    "post_endpoint_diagnostic"]["classification"],
                "artifact": str(paths["one_grain"]),
            },
            "matched_thermal_response": {
                "state": "FAILED_SCIENTIFIC",
                "classification": thermal["classification"],
                "same_geometry_and_transport": thermal[
                    "same_geometry_and_transport"],
                "strict_asb_claimed": False,
                "artifact": str(paths["thermal"]),
            },
            "regression": {
                "state": "PASSED_OPERATOR",
                "configured_tests_passed": args.tests_passed,
                "wall_seconds": args.test_wall_seconds,
                "command": "PYTHONPATH=src:. python -m pytest -q tests",
            },
        },
        "controller_has_executed_next_actions": True,
        "prepared_only": False,
        "campaign_terminal": True,
        "drx_claimed": False,
        "persistent_lagb_claimed": bool(one["scientific_gate_passed"]),
        "strict_asb_claimed": False,
        "material_calibration_claimed": False,
        "unrelated_jobs_untouched": ["56132213"],
    }
    manifest = {
        "schema": "asb-drx/v41/case-manifest/v1",
        "generated_utc": now,
        "canonical_branch": "exp/full-v34-recovery-v1",
        "final_source_sha": args.source_sha,
        "artifacts": {key: artifact(path) for key, path in paths.items()},
        "local_cases": [
            {
                "id": "matched_directional_trials",
                "status": "COMPLETED",
                "source_commit": "69c1fcadcaefd4d26e75aeaf7fe175b2b5e12688",
                "classification": data["front"]["classification"],
            },
            {
                "id": "repaired_zero_work_front_continuation",
                "status": "PHYSICAL_TERMINAL",
                "source_commit": data["front_continuation"]["source_sha"],
                "completion_worktree_head": data["front_continuation"][
                    "completion_worktree_head"],
                "source_scope_equivalence": (
                    "intervening commit changed analysis and documentation "
                    "only; production operator files are identical"),
                "physical_time_s": data["front_continuation"]["physical_time_s"],
                "classification": data["front_continuation"]["classification"],
            },
            {
                "id": "matched_gradient_transfer_n128_n192",
                "status": "COMPLETED",
                "source_commit": data["gradient"]["source_sha"],
                "classification": data["gradient"]["classification"],
                "strong_refinement_passed": data["gradient"][
                    "second_mura_five_percent_passed"],
            },
            {
                "id": "matched_frozen_flow_thermal_control",
                "status": "COMPLETED",
                "source_commit": args.thermal_source_sha,
                "classification": thermal["classification"],
            },
            {
                "id": "one_grain_exact_topology_disabling_pair",
                "status": "COMPLETED",
                "source_commit": args.source_sha,
                "final_strain": 0.0501,
                "classification": one["post_endpoint_diagnostic"][
                    "classification"],
                "all_hard_invariants_passed": one[
                    "post_endpoint_diagnostic"][
                        "all_hard_invariants_passed"],
            },
        ],
        "hpc3_runs": [{
            "run_id": "v40-mura-resume2-20260918-a23f8c7-attempt2",
            "job_id": args.one_grain_job_id,
            "status": "COMPLETED",
            "source_commit": args.one_grain_source_sha,
            "remote_root": args.one_grain_run_root,
            "archive_sha256": one["verified_archive"]["sha256"],
            "checksum_verified": one["verified_archive"]["checksum_verified"],
            "classification": one["classification"],
        }],
        "verification": {
            "configured_regression_tests_passed": args.tests_passed,
            "regression_wall_seconds": args.test_wall_seconds,
            "drx_claimed": False,
            "persistent_lagb_claimed": bool(one["scientific_gate_passed"]),
            "strict_asb_claimed": False,
        },
        "unrelated_jobs_observed_and_not_touched": ["56132213"],
    }
    (root/"v41_campaign_controller.json").write_text(
        json.dumps(controller, indent=2, sort_keys=True) + "\n")
    (root/"v41_case_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
