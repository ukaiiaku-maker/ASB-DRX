#!/usr/bin/env python3
"""Build the compact V36 source/input/restart/output provenance index."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local_cases(roots):
    records = []
    for root in roots:
        for result in sorted(Path(root).glob("*/result.json")):
            data = json.loads(result.read_text())
            checkpoints = sorted(result.parent.glob("checkpoint_*.npz"))
            records.append({
                "case": result.parent.name,
                "source_commit": data.get("source_commit"),
                "configuration": data.get("configuration"),
                "result_path": str(result.resolve()),
                "result_sha256": digest(result),
                "terminal": True,
                "completed_intervals": data.get("completed_intervals"),
                "physical_time_s": data.get("physical_time_s"),
                "latest_checkpoint": (
                    None if not checkpoints else str(checkpoints[-1].resolve())),
                "latest_checkpoint_sha256": (
                    None if not checkpoints else digest(checkpoints[-1])),
            })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-root", action="append", type=Path, default=[])
    parser.add_argument("--hpc-run-record", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    artifacts = {}
    for relative in (
        "full_model/verification/v36_front_channel_kinetics.json",
        "full_model/verification/v36_mura_rate_limit.json",
        "full_model/verification/v36_thermal_matched_causality.json",
        "full_model/verification/v36_recurrent_physical_response.json",
    ):
        path = root/relative
        artifacts[relative] = ({"sha256": digest(path)} if path.exists()
                               else {"status": "PENDING"})
    hpc = None
    if args.hpc_run_record and args.hpc_run_record.exists():
        hpc = json.loads(args.hpc_run_record.read_text())
        # Avoid duplicating the very large file/output checksum maps in the
        # compact campaign index; their authoritative record remains linked.
        hpc = {
            key: hpc.get(key) for key in (
                "run_id", "job_id", "git_commit", "remote_path",
                "local_result_path", "state", "fetch_status",
                "input_bundle_checksum", "last_status_raw")
        }
        hpc["run_record"] = str(args.hpc_run_record.resolve())
        hpc["run_record_sha256"] = digest(args.hpc_run_record)
    result = {
        "schema": "asb-drx/v36/case-manifest/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "branch_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "production_source_commit": (
            "890cb8906a9772d8bd5c5eb43164ecd44ad2720f"),
        "production_source_semantics": (
            "exact V36 recurrent runner and physics; later commits add only "
            "job configuration, postprocessing, evidence, and reports"),
        "local_cases": local_cases(args.local_root),
        "hpc_n192": hpc,
        "artifacts": artifacts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(len(result["local_cases"]))


if __name__ == "__main__":
    main()
