#!/usr/bin/env python3
"""Durable one-heavy-solver controller for V50 loading then matched hold."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    temporary.replace(path)


def read_json(path):
    for _ in range(5):
        try:
            return json.loads(Path(path).read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(1)
    raise RuntimeError(f"could not read stable JSON: {path}")


def valid_full_duration_prefix(manifest):
    valid = 0
    for record in manifest.get("records", []):
        for segment in record.get("segments", []):
            requested = float(segment["requested_duration_s"])
            accepted = float(segment["accepted_duration_s"])
            tolerance = 64*2.220446049250313e-16*max(requested, 1e-300)
            if accepted < requested-tolerance:
                return valid
        valid += 1
    return valid


def manager_payload(source_sha, state, **updates):
    return {
        "schema": "asb-drx/v50/completion-manager/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "manager_pid": os.getpid(), "source_sha": source_sha,
        "state": state, **updates,
    }


def postprocess_pair(loading, hold, output):
    if int(loading["completed_intervals"]) != int(hold["completed_intervals"]):
        raise RuntimeError("loading/hold common horizon mismatch")
    if loading["source_checkpoint_sha256"] != hold["source_checkpoint_sha256"]:
        raise RuntimeError("loading/hold initial state mismatch")
    lr = loading["records"]; hr = hold["records"]
    if abs(float(lr[-1]["load_elapsed_time_s"])
           -float(hr[-1]["load_elapsed_time_s"])) > 1e-18:
        raise RuntimeError("loading/hold physical time mismatch")
    keys = tuple(lr[-1]["endpoint_observables"])
    difference = {key: (float(lr[-1]["endpoint_observables"][key])
                        -float(hr[-1]["endpoint_observables"][key]))
                  for key in keys}
    payload = {
        "schema": "asb-drx/v50/common-horizon-physical-pair/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "loading_source_sha": loading["source_sha"],
        "hold_source_sha": hold["source_sha"],
        "shared_initial_checkpoint_sha256": loading[
            "source_checkpoint_sha256"],
        "completed_intervals": loading["completed_intervals"],
        "common_physical_time_s": lr[-1]["load_elapsed_time_s"],
        "loading_endpoint": lr[-1]["endpoint_observables"],
        "hold_endpoint": hr[-1]["endpoint_observables"],
        "loading_minus_hold": difference,
        "loading_cumulative_first_law_residual_J": lr[-1][
            "cumulative_first_law_residual_J"],
        "hold_cumulative_first_law_residual_J": hr[-1][
            "cumulative_first_law_residual_J"],
        "loading_history": [r["endpoint_observables"] for r in lr],
        "hold_history": [r["endpoint_observables"] for r in hr],
        "front_enabled": False,
        "classification": "COMMON_HORIZON_LOADING_HOLD_PAIR_COMPLETE",
        "drx_claimed": False, "lagb_claimed": False,
        "strict_asb_claimed": False,
    }
    atomic_json(output, payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loading-manifest", type=Path, required=True)
    parser.add_argument("--hold-seed", type=Path, required=True)
    parser.add_argument("--hold-output", type=Path, required=True)
    parser.add_argument("--manager-record", type=Path, required=True)
    parser.add_argument("--pair-output", type=Path, required=True)
    parser.add_argument("--target-intervals", type=int, default=52)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    args = parser.parse_args()
    source_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.run(["git", "diff", "--quiet"]).returncode != 0:
        raise RuntimeError("manager requires immutable tracked source")
    initial_source_sha = source_sha
    while True:
        loading = read_json(args.loading_manifest)
        valid = valid_full_duration_prefix(loading)
        atomic_json(args.manager_record, manager_payload(
            source_sha, "WAITING_FOR_RETAINED_LOADING",
            loading_status=loading["status"],
            loading_published_intervals=len(loading["records"]),
            loading_valid_full_duration_prefix=valid,
            target_intervals=args.target_intervals,
            next_action="poll retained loading without launching a second solver"))
        if valid != len(loading["records"]):
            atomic_json(args.manager_record, manager_payload(
                source_sha, "FAILED_SCIENTIFIC_SHORTENED_V49_PREFIX",
                loading_valid_full_duration_prefix=valid,
                next_action="preserve valid prefix; do not start hold"))
            return 2
        if (loading["status"] == "COMPLETE"
                and int(loading["completed_intervals"]) == args.target_intervals):
            break
        time.sleep(args.poll_seconds)
    if subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip() != initial_source_sha:
        raise RuntimeError("V50 source changed while manager waited")
    latest = Path(loading["latest_checkpoint"])
    if digest(latest) != loading["latest_checkpoint_sha256"]:
        raise RuntimeError("completed loading checkpoint checksum mismatch")
    args.hold_output.mkdir(parents=True, exist_ok=True)
    existing = args.hold_output/"run_manifest.json"
    restart = args.hold_seed
    if existing.exists():
        prior = read_json(existing)
        if prior["status"] == "COMPLETE":
            hold = prior
            pair = postprocess_pair(loading, hold, args.pair_output)
            atomic_json(args.manager_record, manager_payload(
                source_sha, "COMPLETE", pair_output=str(args.pair_output),
                pair_classification=pair["classification"]))
            return 0
        restart = Path(prior["latest_checkpoint"])
    command = [
        sys.executable, "-m",
        "full_model.analysis.run_v49_physical_continuation",
        "--restart", str(restart), "--output-dir", str(args.hold_output),
        "--grid", "128", "--intervals", str(args.target_intervals),
        "--dt-s", "4.8828125e-7", "--protocol", "hold",
    ]
    log_path = args.hold_output/"manager_child.log"
    with log_path.open("a") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        while child.poll() is None:
            progress = (read_json(existing) if existing.exists() else {})
            atomic_json(args.manager_record, manager_payload(
                source_sha, "RUNNING_MATCHED_HOLD", child_pid=child.pid,
                hold_published_intervals=len(progress.get("records", [])),
                target_intervals=args.target_intervals,
                loading_checkpoint_sha256=loading["latest_checkpoint_sha256"],
                source_transition={
                    "from_source_sha": loading["source_sha"],
                    "to_source_sha": source_sha,
                    "hold_restart_checkpoint": str(Path(restart).resolve()),
                },
                next_action="continue hold; then postprocess common horizon"))
            time.sleep(args.poll_seconds)
    if child.returncode != 0:
        atomic_json(args.manager_record, manager_payload(
            source_sha, "FAILED_HOLD_PROCESS", returncode=child.returncode,
            child_log=str(log_path.resolve()),
            next_action="diagnose from last durable hold checkpoint"))
        return child.returncode
    hold = read_json(existing)
    pair = postprocess_pair(loading, hold, args.pair_output)
    atomic_json(args.manager_record, manager_payload(
        source_sha, "COMPLETE", pair_output=str(args.pair_output.resolve()),
        pair_output_sha256=digest(args.pair_output),
        pair_classification=pair["classification"],
        next_action="scientifically classify common-horizon response"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
