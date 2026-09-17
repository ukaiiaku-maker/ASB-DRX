#!/usr/bin/env python3
"""Durable local monitor/fetch controller for the two V37 HPC bundles."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


TERMINAL = {"RETRIEVED", "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT",
            "OUT_OF_MEMORY", "NODE_FAIL"}
ACTIVE_MARKERS = ("|RUNNING|", "|PENDING|", "|CONFIGURING|")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
    temporary.replace(path)


def read_record(project: Path, run_id: str) -> dict:
    path = project/".hpc3"/"runs"/f"{run_id}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--poll-s", type=float, default=300.0)
    args = parser.parse_args()
    iteration = 0
    while True:
        iteration += 1; statuses = {}
        for run_id in args.run_id:
            process = subprocess.run(
                [str(args.runner), "status", run_id], cwd=args.project,
                text=True, capture_output=True, timeout=60)
            record = read_record(args.project, run_id)
            output = process.stdout+process.stderr
            statuses[run_id] = {
                "returncode": process.returncode,
                "status_output": output[-8000:],
                "scheduler_active": any(marker in output for marker in ACTIVE_MARKERS),
                "record_state": record.get("state"),
                "job_id": record.get("job_id"),
                "fetch_status": record.get("fetch_status"),
            }
        # Reconciliation contacts every historical run in the project and can be
        # slow.  It is unnecessary while either scoped scheduler job is active.
        if not any(item["scheduler_active"] for item in statuses.values()):
            subprocess.run(
                [str(args.runner), "reconcile", "--fetch-completed"],
                cwd=args.project, text=True, capture_output=True, timeout=300)
            # Re-read after reconciliation because it can atomically fetch results.
            for run_id in args.run_id:
                record = read_record(args.project, run_id)
                statuses[run_id]["record_state"] = record.get("state")
                statuses[run_id]["fetch_status"] = record.get("fetch_status")
        done = all(item["record_state"] in TERMINAL for item in statuses.values())
        atomic_json(args.state_file, {
            "schema": "asb-drx/v37/hpc-manager-state/v1",
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "iteration": iteration, "terminal": done, "runs": statuses,
        })
        if done:
            return
        time.sleep(max(args.poll_s, 120.0))


if __name__ == "__main__":
    main()
