#!/usr/bin/env python3
"""Audit the completed V65 campaign before making a completion claim."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from full_model.analysis.build_v65_delivery import verify


EXPECTED_PRODUCTION_SOURCE = "6debbe0813ca0c365abcf473e5106cb90dbdd75a"
EXPECTED_V64_CHECKPOINT = (
    "2bbf987b6e6dbbf603c518686b90812a91c821715874bd580a0367dbeb1b1667"
)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def evaluate_records(records: dict) -> dict[str, bool]:
    manifest = records.get("manifest") or {}
    result = records.get("result") or {}
    finalizer = records.get("finalizer") or {}
    closure = records.get("closure") or {}
    decision = records.get("decision") or {}
    identity = records.get("identity_monitor") or {}
    repository_sync = records.get("repository_sync") or {}
    target = float(result.get("target_time_s", float("nan")))
    final = float(result.get("final_time_s", float("nan")))
    exit_codes = finalizer.get("step_exit_codes", {})
    return {
        "v64_terminal_lineage_exact": manifest.get("lineage", {}).get(
            "terminal_checkpoint_sha256"
        ) == EXPECTED_V64_CHECKPOINT,
        "production_source_exact": result.get("source_commit")
        == EXPECTED_PRODUCTION_SOURCE,
        "production_target_reached": result.get("execution_status")
        == "TARGET_REACHED" and abs(final-target) <= 1e-18,
        "finalizer_complete": finalizer.get("status") == "COMPLETE"
        and finalizer.get("all_required_artifacts_present") is True,
        "postprocess_and_tests_passed": bool(exit_codes)
        and all(int(code) == 0 for code in exit_codes.values()),
        "closure_complete": closure.get("status") == "COMPLETE",
        "scoped_experiment_complete": decision.get(
            "scoped_experiment_complete"
        ) is True,
        "claim_boundary_preserved": decision.get(
            "full_scientific_objective_complete"
        ) is False,
        "component_identity_monitor_complete": identity.get("status")
        == "COMPLETE",
        "terminal_repository_sync_complete": repository_sync.get("status")
        == "COMPLETE",
    }


def git_value(worktree: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=worktree, text=True
    ).strip()


def audit(campaign: Path, worktree: Path, branch: str) -> dict:
    analysis = campaign/"analysis"
    run = campaign/"runs/production_1p035_to1p25us_6debbe0"
    archive = campaign/"v65_definitive_delivery.zip"
    records = {
        "manifest": load(worktree/"full_model/verification/v65_campaign_manifest.json"),
        "result": load(run/"result.json"),
        "finalizer": load(campaign/"finalizer_status.json"),
        "closure": load(campaign/"closure_status.json"),
        "decision": load(analysis/"v65_final_decision.json"),
        "identity_monitor": load(
            analysis/"live_monitor/component_identity_monitor_status.json"
        ),
        "repository_sync": load(campaign/"repository_sync_status.json"),
    }
    checks = evaluate_records(records)
    archive_error = None
    if archive.is_file():
        try:
            verify(archive)
        except Exception as error:  # retained verbatim in the audit artifact
            archive_error = f"{type(error).__name__}: {error}"
    checks["archive_inventory_and_payload_hashes_verified"] = bool(
        archive.is_file() and archive_error is None
    )
    closure = records["closure"] or {}
    checks["archive_outer_hash_matches_closure"] = bool(
        archive.is_file()
        and closure.get("archive_sha256") == digest(archive)
    )
    local_head = git_value(worktree, "rev-parse", "HEAD")
    remote_line = git_value(
        worktree, "ls-remote", "--heads", "origin", branch
    )
    remote_head = remote_line.split()[0] if remote_line else None
    porcelain = git_value(worktree, "status", "--porcelain")
    checks["worktree_clean"] = porcelain == ""
    checks["remote_branch_matches_local_head"] = remote_head == local_head
    return {
        "schema": "asb-drx-v65-completion-audit-v1",
        "status": "COMPLETE" if all(checks.values()) else "INCOMPLETE",
        "checks": checks,
        "failed_or_pending_checks": [
            name for name, passed in checks.items() if not passed
        ],
        "archive_error": archive_error,
        "local_head": local_head,
        "remote_head": remote_head,
        "archive": str(archive.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--worktree", required=True, type=Path)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = audit(
        args.campaign.resolve(), args.worktree.resolve(), args.branch
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name+".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["status"] == "COMPLETE" else 1)


if __name__ == "__main__":
    main()
