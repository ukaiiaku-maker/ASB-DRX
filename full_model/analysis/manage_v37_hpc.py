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


def parse_submission(output: str) -> tuple[str | None, str | None]:
    run_id = None
    job_id = None
    for line in output.splitlines():
        if line.startswith("Run ID:"):
            run_id = line.split(":", 1)[1].strip()
        elif line.startswith("Job ID:"):
            job_id = line.split(":", 1)[1].strip()
    return run_id, job_id


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
    temporary.replace(path)


def read_record(project: Path, run_id: str) -> dict:
    path = project/".hpc3"/"runs"/f"{run_id}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def run_postprocessor(command: list[str], output: Path) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(command, text=True, capture_output=True, timeout=900)
    return {
        "returncode": process.returncode,
        "output": (process.stdout+process.stderr)[-8000:],
        "artifact": str(output),
        "artifact_exists": output.exists(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--poll-s", type=float, default=300.0)
    parser.add_argument("--followup-config", type=Path)
    parser.add_argument("--followup-trigger-run-id")
    parser.add_argument("--conduction-run-id")
    parser.add_argument("--mura-run-id")
    parser.add_argument("--postprocess-root", type=Path)
    args = parser.parse_args()
    if bool(args.followup_config) != bool(args.followup_trigger_run_id):
        parser.error("follow-up config and trigger run id must be supplied together")
    run_ids = list(dict.fromkeys(args.run_id))
    followup = {"state": "NOT_REQUESTED", "run_id": None, "job_id": None}
    postprocessing = {}
    if args.followup_config:
        followup["state"] = "WAITING_FOR_TRIGGER"
    if args.state_file.exists():
        previous = json.loads(args.state_file.read_text())
        saved = previous.get("followup", {})
        if saved.get("run_id"):
            followup = saved
            run_ids.append(saved["run_id"])
        postprocessing = previous.get("postprocessing", {})
    iteration = 0
    while True:
        iteration += 1; statuses = {}
        for run_id in run_ids:
            try:
                process = subprocess.run(
                    [str(args.runner), "status", run_id], cwd=args.project,
                    text=True, capture_output=True, timeout=60)
                output = process.stdout+process.stderr
                returncode = process.returncode
            except subprocess.TimeoutExpired as error:
                output = ((error.stdout or "")+(error.stderr or "")
                          +"\nSTATUS_TIMEOUT")
                returncode = 124
            record = read_record(args.project, run_id)
            statuses[run_id] = {
                "returncode": returncode,
                "status_output": output[-8000:],
                "scheduler_active": any(marker in output for marker in ACTIVE_MARKERS),
                "record_state": record.get("state"),
                "job_id": record.get("job_id"),
                "fetch_status": record.get("fetch_status"),
            }
        # Fetch each finished scoped run independently so a long companion job
        # never delays postprocessing or causes historical project-wide scans.
        for run_id, item in statuses.items():
            if item["returncode"] == 0 and not item["scheduler_active"]:
                try:
                    fetched = subprocess.run(
                        [str(args.runner), "fetch", run_id], cwd=args.project,
                        text=True, capture_output=True, timeout=300)
                    item["fetch_returncode"] = fetched.returncode
                    item["fetch_output"] = (fetched.stdout+fetched.stderr)[-8000:]
                except subprocess.TimeoutExpired:
                    item["fetch_returncode"] = 124
                    item["fetch_output"] = "FETCH_TIMEOUT"
                record = read_record(args.project, run_id)
                item["record_state"] = record.get("state")
                item["fetch_status"] = record.get("fetch_status")

        if args.postprocess_root:
            project_results = (args.project/"hpc3-results"/args.project.name)
            for run_id, item in statuses.items():
                if (item["record_state"] != "RETRIEVED"
                        or postprocessing.get(run_id, {}).get("returncode") == 0):
                    continue
                fetched_root = project_results/run_id
                artifact_dir = args.postprocess_root/run_id
                try:
                    if run_id == args.conduction_run_id:
                        command = [
                            "python", "full_model/analysis/postprocess_v37_conduction_results.py",
                            "--root", str(fetched_root), "--output",
                            str(artifact_dir/"v37_conduction_results.json"),
                            "--figure", str(artifact_dir/"v37_conduction_results.png")]
                        output = artifact_dir/"v37_conduction_results.json"
                    elif run_id == args.mura_run_id:
                        configs = list(fetched_root.rglob("case_config.json"))
                        if len(configs) != 1:
                            raise ValueError("expected one fetched Mura case")
                        command = [
                            "python", "full_model/analysis/postprocess_v37_mura_long.py",
                            "--case-dir", str(configs[0].parent), "--output",
                            str(artifact_dir/"v37_mura_long.json"), "--figure",
                            str(artifact_dir/"v37_mura_long.png")]
                        output = artifact_dir/"v37_mura_long.json"
                    elif run_id == followup.get("run_id"):
                        command = [
                            "python", "full_model/analysis/postprocess_v37_front_response.py",
                            "--root", str(fetched_root), "--output",
                            str(artifact_dir/"v37_front_response.json"), "--figure",
                            str(artifact_dir/"v37_front_response.png")]
                        output = artifact_dir/"v37_front_response.json"
                    else:
                        continue
                    postprocessing[run_id] = run_postprocessor(command, output)
                except (OSError, ValueError, subprocess.TimeoutExpired) as error:
                    postprocessing[run_id] = {
                        "returncode": 125, "error": str(error)}

        trigger = statuses.get(args.followup_trigger_run_id, {})
        if (args.followup_config and followup["state"] == "WAITING_FOR_TRIGGER"
                and trigger.get("returncode") == 0
                and not trigger.get("scheduler_active", True)):
            submitted = subprocess.run(
                [str(args.runner), "submit", str(args.followup_config)],
                cwd=args.project, text=True, capture_output=True, timeout=600)
            output = submitted.stdout+submitted.stderr
            new_run, new_job = parse_submission(output)
            if submitted.returncode == 0 and new_run and new_job:
                followup = {"state": "SUBMITTED", "run_id": new_run,
                            "job_id": new_job, "output": output[-8000:]}
                run_ids.append(new_run)
            else:
                followup = {"state": "WAITING_FOR_TRIGGER", "run_id": None,
                            "job_id": None,
                            "last_submission_returncode": submitted.returncode,
                            "last_submission_output": output[-8000:]}
        done = bool(statuses) and all(
            item["record_state"] in TERMINAL for item in statuses.values())
        if args.followup_config:
            done = (done and followup["state"] == "SUBMITTED"
                    and followup["run_id"] in statuses)
        atomic_json(args.state_file, {
            "schema": "asb-drx/v37/hpc-manager-state/v1",
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "iteration": iteration, "terminal": done, "runs": statuses,
            "followup": followup,
            "postprocessing": postprocessing,
        })
        if done:
            return
        time.sleep(max(args.poll_s, 120.0))


if __name__ == "__main__":
    main()
