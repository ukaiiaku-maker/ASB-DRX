#!/usr/bin/env python3
"""Resumable V53 local-only near-flow and companion-grid controller."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from full_model.analysis.run_v52_completion_manager import (
    LogicalLock, atomic_json, digest, inspect_manifest, process_identity,
    run_short, utc, wait_or_run_continuation,
)


SCHEMA = "asb-drx/v53/bulk-manager/v1"
N128_104_SHA = "8a6ccdfdbe3e2c8c7abbd952c78e75bbde53930c97117b8c23aaf683bd59c6ba"
N192_032_SHA = "e89784719aace964e358c0b367d72d386b6abfdfa5f0d512d49821acfd7c5b2f"


def persist(path, state, status=None):
    if status is not None:
        state["state"] = status
    state["updated_utc"] = utc()
    atomic_json(path, state)


def load_state(path, root, n128_output, n192_output):
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if path.exists():
        state = json.loads(path.read_text())
        if state.get("schema") != SCHEMA:
            raise RuntimeError("V53 manager journal schema mismatch")
        required = {
            "source_worktree": str(root),
            "n128_output": str(n128_output),
            "n192_output": str(n192_output),
        }
        if any(state.get(key) != value for key, value in required.items()):
            raise RuntimeError("V53 manager journal ownership mismatch")
        if state.get("source_sha") != source:
            raise RuntimeError("worker source changed after journal creation")
        state["resume_count"] = int(state.get("resume_count", 0))+1
        state["resumed_utc"] = utc()
    else:
        state = {
            "schema": SCHEMA,
            "created_utc": utc(),
            "source_sha": source,
            "source_worktree": str(root),
            "n128_output": str(n128_output),
            "n192_output": str(n192_output),
            "execution_location": "local",
            "hpc3_jobs_submitted": 0,
            "resume_count": 0,
            "stages": {},
            "history": [],
            "state": "STARTING",
        }
    state["manager"] = process_identity(os.getpid())
    return state


def segment_wall_seconds(manifest, after_interval=0):
    return float(sum(
        float(segment.get("wall_seconds", 0.0))
        for row in manifest.get("records", [])
        if int(row.get("interval", 0)) > int(after_interval)
        for segment in row.get("segments", [])))


def recent_segment_seconds(manifest, count=6):
    values = [
        float(segment.get("wall_seconds", 0.0))
        for row in manifest.get("records", [])
        for segment in row.get("segments", [])
        if float(segment.get("wall_seconds", 0.0)) > 0.0]
    return (sum(values[-count:])/len(values[-count:])) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n128-checkpoint", type=Path, required=True)
    parser.add_argument("--n192-checkpoint", type=Path, required=True)
    parser.add_argument("--retained-n128-root", type=Path, required=True)
    parser.add_argument("--initialization-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manager-state", type=Path, required=True)
    parser.add_argument("--n192-targets", default="52,81,104")
    parser.add_argument("--n192-wall-budget-s", type=float, default=8*3600.0)
    parser.add_argument("--poll-s", type=float, default=30.0)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    output_root = args.output_root.resolve()
    n128_output = output_root/"n128/loading"
    n192_output = output_root/"n192/loading"
    verification = output_root/"verification"
    verification.mkdir(parents=True, exist_ok=True)
    log = verification/"v53_bulk_manager.log"
    if digest(args.n128_checkpoint) != N128_104_SHA:
        raise RuntimeError("retained n128 interval-104 anchor checksum mismatch")
    if digest(args.n192_checkpoint) != N192_032_SHA:
        raise RuntimeError("retained n192 interval-32 anchor checksum mismatch")
    state_path = args.manager_state.resolve()
    lock = LogicalLock(output_root/".v53-bulk-manager.lock")
    lock.acquire()
    state = load_state(state_path, root, n128_output, n192_output)
    persist(state_path, state)
    try:
        temporal_dir = output_root/"near-flow-temporal"
        temporal = temporal_dir/"near_flow_temporal.json"
        if not temporal.exists():
            elapsed = run_short([
                sys.executable,
                "full_model/analysis/run_v53_near_flow_temporal.py",
                "--checkpoint", str(args.n128_checkpoint.resolve()),
                "--output-dir", str(temporal_dir)], root, log)
            state["history"].append({
                "stage": "N128_NEAR_FLOW_TEMPORAL", "wall_s": elapsed})
        temporal_payload = json.loads(temporal.read_text())
        if temporal_payload["source_checkpoint_sha256"] != N128_104_SHA:
            raise RuntimeError("near-flow result has wrong parent")
        state["stages"]["N128_NEAR_FLOW_TEMPORAL"] = {
            "classification": "VALID_COMPLETED",
            "result": str(temporal), "sha256": digest(temporal),
            "stress_increment_sign_resolved": temporal_payload[
                "comparison"]["stress_increment_sign_resolved_by_subdivision"],
            "quarter_step_branch_required": temporal_payload[
                "quarter_step_branch_required"],
        }
        persist(state_path, state, "N128_NEAR_FLOW_TEMPORAL_COMPLETE")

        command = [
            sys.executable,
            "full_model/analysis/run_v49_physical_continuation.py",
            "--restart", str(args.n128_checkpoint.resolve()),
            "--output-dir", str(n128_output), "--grid", "128",
            "--intervals", "128", "--dt-s", "4.8828125e-7",
            "--protocol", "continued_deformation", "--strain-rate-s", "100",
        ]
        n128 = wait_or_run_continuation(
            output=n128_output, target=128, command=command, cwd=root,
            state=state, state_path=state_path, stage="N128_TO_128",
            log=log, poll_s=args.poll_s)
        n128_manifest = Path(n128["manifest_path"])
        continuation = verification/"v53_n128_continuation.json"
        plot = verification/"v53_n128_continuation.png"
        if not continuation.exists():
            elapsed = run_short([
                sys.executable,
                "full_model/analysis/run_v52_continuation_analysis.py",
                "--manifest", str(n128_manifest), "--output", str(continuation),
                "--plot", str(plot)], root, log)
            state["history"].append({
                "stage": "N128_TO_128_POSTPROCESS", "wall_s": elapsed})
        state["stages"]["N128_TO_128_POSTPROCESS"] = {
            "classification": "VALID_COMPLETED",
            "result": str(continuation), "sha256": digest(continuation),
            "plot": str(plot), "plot_sha256": digest(plot),
        }
        persist(state_path, state, "N128_TO_128_COMPLETE")

        targets = sorted(set(int(value) for value in args.n192_targets.split(",")))
        if any(target <= 32 for target in targets):
            raise ValueError("V53 n192 targets must extend the retained interval 32")
        maximum = 32
        for target in targets:
            manifest_path = n192_output/"run_manifest.json"
            if manifest_path.exists():
                current = json.loads(manifest_path.read_text())
                maximum = int(current["completed_intervals"])
                # V52's accepted interval-32 prefix is inherited scientific
                # history, not wall time spent from the V53 extension budget.
                spent = segment_wall_seconds(current, after_interval=32)
                recent = recent_segment_seconds(current)
                projected = (None if recent is None else
                             recent*max(target-maximum, 0))
                if maximum < target and (spent >= args.n192_wall_budget_s or
                        projected is not None
                        and spent+projected > args.n192_wall_budget_s):
                    state["history"].append({
                        "stage": "N192_BUDGET_TERMINAL",
                        "maximum_completed_interval": maximum,
                        "next_target_not_started": target,
                        "cumulative_segment_wall_s": spent,
                        "recent_mean_segment_wall_s": recent,
                        "projected_increment_wall_s": projected,
                        "budget_s": args.n192_wall_budget_s,
                    })
                    break
            command = [
                sys.executable,
                "full_model/analysis/run_v49_physical_continuation.py",
                "--restart", str(args.n192_checkpoint.resolve()),
                "--output-dir", str(n192_output), "--grid", "192",
                "--intervals", str(target), "--dt-s", "4.8828125e-7",
                "--protocol", "continued_deformation", "--strain-rate-s", "100",
            ]
            result = wait_or_run_continuation(
                output=n192_output, target=target, command=command, cwd=root,
                state=state, state_path=state_path,
                stage=f"N192_TO_{target:03d}", log=log, poll_s=args.poll_s)
            maximum = int(result["manifest"]["completed_intervals"])
            coarse_checkpoint = (args.retained_n128_root/
                                 f"checkpoint_{maximum:06d}.npz")
            if not coarse_checkpoint.exists():
                raise RuntimeError(f"missing retained n128 interval {maximum}")
            comparison = verification/f"v53_spatial_comparison_{maximum:03d}.json"
            run_short([
                sys.executable,
                "full_model/analysis/run_v53_spatial_comparison.py",
                "--n128-checkpoint", str(coarse_checkpoint),
                "--n128-manifest", str(args.retained_n128_root/"run_manifest.json"),
                "--n192-checkpoint", result["manifest"]["latest_checkpoint"],
                "--n192-manifest", result["manifest_path"],
                "--initialization-audit", str(args.initialization_audit),
                "--output", str(comparison)], root, log)
            state["stages"][f"N192_COMPARE_{maximum:03d}"] = {
                "classification": "VALID_COMPLETED",
                "result": str(comparison), "sha256": digest(comparison),
            }
            persist(state_path, state, f"N192_{maximum:03d}_QUALIFIED")

        state["maximum_n192_completed_interval"] = maximum
        state["completed_utc"] = utc()
        state["state"] = "BULK_COMPLETE_AT_ACHIEVED_SCOPE"
        state.pop("failure", None); state.pop("failed_utc", None)
        persist(state_path, state)
    except Exception as error:
        state["state"] = "FAILED_CONTROLLER"
        state["failure"] = f"{type(error).__name__}: {error}"
        state["failed_utc"] = utc()
        persist(state_path, state)
        raise
    finally:
        lock.release()


if __name__ == "__main__":
    main()
