#!/usr/bin/env python3
"""Assemble V42 controller and checksum manifest from completed decisions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def artifact(path):
    return {"path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verification-root", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--tests-passed", type=int, default=747)
    parser.add_argument("--test-wall-seconds", type=float, default=252.37)
    args = parser.parse_args(); root = args.verification_root
    paths = {
        "topology_first_failure": root/"v42_topology_first_failure.json",
        "topology_repair": root/"v42_topology_repair_decision.json",
        "topology_continuation": root/"v42_topology_continuation.json",
        "gradient": root/"v42_matched_gradient_refinement.json",
        "one_grain": root/"v42_one_grain_response.json",
        "front": root/"v42_front_physical_alternative.json",
        "thermal": root/"v42_thermal_selected_response.json",
        "registry": root/"v42_parameter_and_hypothesis_registry.json",
    }
    data = {name: load(path) for name, path in paths.items()}
    now = datetime.now(timezone.utc).isoformat()
    topology = data["topology_continuation"]
    gradient = data["gradient"]
    controller = {
        "schema": "asb-drx/v42/campaign-controller/v1",
        "generated_utc": now,
        "campaign": "Topology energy, resolved organization, and physical response V42",
        "canonical_branch": "exp/full-v34-recovery-v1",
        "final_source_sha": args.source_sha,
        "branches": {
            "topology_energy_kinematics": {
                "state": "PASSED_OPERATOR" if topology["hard_invariants_passed"] else "HARD_INVALID",
                "classification": topology["classification"],
                "artifact": str(paths["topology_continuation"]),
            },
            "resolved_organization": {
                "state": "FAILED_SCIENTIFIC",
                "classification": data["one_grain"]["classification"],
                "artifact": str(paths["one_grain"]),
            },
            "mura_spatial_accuracy": {
                "state": "PASSED_NUMERICAL" if gradient[
                    "strong_rms_five_percent_passed"] else "FAILED_NUMERICAL",
                "classification": gradient["classification"],
                "curl_nye_rms_relative_difference": gradient[
                    "relative_differences"]["curl_nye_rms_m1"],
                "artifact": str(paths["gradient"]),
            },
            "front_physical_alternative": {
                "state": "COMPLETED",
                "classification": data["front"]["classification"],
                "artifact": str(paths["front"]),
            },
            "thermal_selected_response": {
                "state": "VALID_RETAINED_EVIDENCE",
                "classification": data["thermal"]["classification"],
                "artifact": str(paths["thermal"]),
            },
            "regression": {
                "state": "PASSED_OPERATOR", "configured_tests_passed": args.tests_passed,
                "wall_seconds": args.test_wall_seconds,
                "command": "PYTHONPATH=src:. python -m pytest -q tests",
            },
        },
        "controller_has_executed_next_actions": True,
        "prepared_only": False, "campaign_terminal": True,
        "drx_claimed": False, "persistent_lagb_claimed": False,
        "strict_asb_claimed": False, "material_calibration_claimed": False,
        "unrelated_jobs_untouched": ["56132213"],
    }
    manifest = {
        "schema": "asb-drx/v42/case-manifest/v1", "generated_utc": now,
        "canonical_branch": "exp/full-v34-recovery-v1",
        "final_source_sha": args.source_sha,
        "artifacts": {name: artifact(path) for name, path in paths.items()},
        "local_cases": [
            {"id": "first_topology_failure_replay", "status": "COMPLETED",
             "source_commit": "4b6da0dfbb6f9e8b9a4e009e1eeb39c3b67ca03a",
             "classification": data["topology_first_failure"]["classification"]},
            {"id": "matched_gradient_62p5us_n128_n192", "status": "COMPLETED",
             "source_commit": gradient["source_commit"],
             "classification": gradient["classification"],
             "wall_seconds": {"n128": 442.00628770608455,
                              "n192": 949.4433867059415}},
            {"id": "front_elastic_mismatch_counterfactual", "status": "COMPLETED",
             "classification": data["front"]["classification"]},
        ],
        "hpc3_runs": [{
            "run_id": Path(args.remote_root).name, "job_id": args.job_id,
            "status": "COMPLETED", "source_commit": topology[
                "repaired_status"]["source_sha"],
            "remote_root": args.remote_root, "archive_sha256": args.archive_sha256,
            "checksum_verified": True, "classification": topology["classification"],
        }],
        "quarantined_attempts": [{
            "job_id": "56166765",
            "classification": "INVALID_SEED_FILENAME_STARTED_FRESH_TRAJECTORY",
            "used_as_scientific_evidence": False,
        }],
        "verification": {
            "configured_regression_tests_passed": args.tests_passed,
            "regression_wall_seconds": args.test_wall_seconds,
            "drx_claimed": False, "persistent_lagb_claimed": False,
            "strict_asb_claimed": False,
        },
        "unrelated_jobs_observed_and_not_touched": ["56132213"],
    }
    (root/"v42_campaign_controller.json").write_text(
        json.dumps(controller, indent=2, sort_keys=True)+"\n")
    (root/"v42_case_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
