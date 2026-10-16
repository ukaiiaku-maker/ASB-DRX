#!/usr/bin/env python3
"""Audit physical-source equivalence across matched V67 result commits."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
PHYSICAL_PREFIXES = (
    "full_model/production/",
    "full_model/reference_configs/",
    "full_model/reference_sources/",
)
PHYSICAL_ANALYSIS_DRIVERS = {
    "full_model/analysis/run_v58_three_grain_production.py",
    "full_model/analysis/run_v66_coupled_rx_asb.py",
    "full_model/analysis/run_v67_drx_asb_overlap.py",
    "full_model/analysis/run_v67_interface_structural_bridge.py",
}


def physical_changes(paths: list[str]) -> list[str]:
    return sorted(path for path in paths if (
        path in PHYSICAL_ANALYSIS_DRIVERS
        or any(path.startswith(prefix) for prefix in PHYSICAL_PREFIXES)))


def _blob(commit: str, path: str) -> str:
    data = subprocess.check_output(
        ("git", "show", f"{commit}:{path}"), cwd=ROOT)
    return hashlib.sha256(data).hexdigest()


def audit(first_result: Path, second_result: Path) -> dict:
    first = json.loads(first_result.read_text())
    second = json.loads(second_result.read_text())
    first_commit = first["source_commit"]
    second_commit = second["source_commit"]
    paths = subprocess.check_output(
        ("git", "diff", "--name-only", first_commit, second_commit),
        cwd=ROOT, text=True).splitlines()
    physical = physical_changes(paths)
    blobs = {
        path: {
            first_commit: _blob(first_commit, path),
            second_commit: _blob(second_commit, path),
        } for path in sorted(PHYSICAL_ANALYSIS_DRIVERS)
    }
    return {
        "schema": "v67-matched-physical-source-equivalence-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "results": [str(first_result.resolve()), str(second_result.resolve())],
        "source_commits": [first_commit, second_commit],
        "all_changed_paths": paths,
        "physical_changed_paths": physical,
        "physical_driver_blob_sha256": blobs,
        "decision": {
            "physical_source_equivalent": not physical,
            "classification": (
                "MATCHED_PHYSICAL_SOURCE_EQUIVALENT"
                if not physical else "PHYSICAL_SOURCE_MISMATCH"),
        },
        "scope": (
            "Equivalence covers the production package, reference physics "
            "inputs, and the four analysis drivers that construct and advance "
            "this V67 common-state trajectory. Documentation, classifiers, "
            "tests, and previously generated evidence may differ."),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("first_result", type=Path)
    parser.add_argument("second_result", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    payload = audit(args.first_result, args.second_result)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
