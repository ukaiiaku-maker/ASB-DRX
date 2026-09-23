#!/usr/bin/env python3
"""Durable sequential handoff from the frozen V53 bulk worker to closure."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from full_model.analysis.run_v52_completion_manager import (
    LogicalLock, atomic_json, digest, process_identity, same_identity, utc,
)


SCHEMA = "asb-drx/v53/handoff-manager/v1"


def persist(path, state, status=None):
    if status is not None:
        state["state"] = status
    state["updated_utc"] = utc(); atomic_json(path, state)


def run_stage(state, state_path, name, command, cwd, log, output=None):
    if output is not None and Path(output).exists():
        state["stages"][name] = {
            "classification": "VALID_EXISTING_OUTPUT",
            "output": str(Path(output).resolve()),
            "sha256": digest(output),
        }
        persist(state_path, state, "COMPLETED_"+name)
        return
    persist(state_path, state, "RUNNING_"+name)
    started = time.perf_counter()
    with Path(log).open("a") as stream:
        stream.write("COMMAND "+json.dumps(command)+"\n"); stream.flush()
        result = subprocess.run(
            command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
            text=True, env={**os.environ, "PYTHONPATH": "src:."})
    record = {
        "command": command, "cwd": str(Path(cwd).resolve()),
        "returncode": result.returncode,
        "wall_seconds": time.perf_counter()-started,
    }
    if output is not None and Path(output).exists():
        record.update({"output": str(Path(output).resolve()),
                       "sha256": digest(output)})
    state["stages"][name] = record
    persist(state_path, state)
    if result.returncode:
        raise RuntimeError(f"{name} failed with return code {result.returncode}")
    if output is not None and not Path(output).exists():
        raise RuntimeError(f"{name} did not create its declared output")
    state["stages"][name]["classification"] = "VALID_COMPLETED"
    persist(state_path, state, "COMPLETED_"+name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predecessor-state", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bulk-source-worktree", type=Path, required=True)
    parser.add_argument("--retained-n128-root", type=Path, required=True)
    parser.add_argument("--retained-n192-checkpoint", type=Path, required=True)
    parser.add_argument("--initialization-audit", type=Path, required=True)
    parser.add_argument("--manager-state", type=Path, required=True)
    parser.add_argument("--poll-s", type=float, default=30.0)
    args = parser.parse_args()
    root = Path.cwd().resolve(); output = args.output_root.resolve()
    verification = output/"verification"; verification.mkdir(parents=True, exist_ok=True)
    log = verification/"v53_handoff_manager.log"
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    state_path = args.manager_state.resolve()
    lock = LogicalLock(output/".v53-handoff-manager.lock"); lock.acquire()
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("schema") != SCHEMA or state.get("source_sha") != source:
            raise RuntimeError("handoff journal source or schema mismatch")
        state["resume_count"] = int(state.get("resume_count", 0))+1
    else:
        state = {
            "schema": SCHEMA, "created_utc": utc(), "source_sha": source,
            "source_worktree": str(root), "output_root": str(output),
            "execution_location": "local", "hpc3_jobs_submitted": 0,
            "resume_count": 0, "state": "STARTING", "stages": {},
        }
    state["manager"] = process_identity(os.getpid()); persist(state_path, state)
    try:
        while True:
            predecessor = json.loads(args.predecessor_state.read_text())
            identity = predecessor.get("manager", {})
            live = process_identity(identity.get("pid", -1))
            if same_identity(identity, live):
                state["predecessor"] = {
                    "state": predecessor.get("state"), "live": True,
                    "manager": identity}
                persist(state_path, state, "WAITING_FOR_FROZEN_N128_WORKER")
                time.sleep(args.poll_s); continue
            state["predecessor"] = {
                "state": predecessor.get("state"), "live": False,
                "manager": identity}
            if predecessor.get("state") != "BULK_COMPLETE_AT_ACHIEVED_SCOPE":
                raise RuntimeError(
                    f"predecessor ended without valid terminal: {predecessor.get('state')}")
            break
        persist(state_path, state, "PREDECESSOR_COMPLETE")

        parent104 = args.retained_n128_root/"checkpoint_000104.npz"
        temporal104 = output/"near-flow-temporal/near_flow_temporal.json"
        quarter104 = output/"near-flow-temporal/near_flow_quarter.json"
        run_stage(state, state_path, "N128_104_QUARTER", [
            sys.executable, "full_model/analysis/run_v53_near_flow_quarter.py",
            "--checkpoint", str(parent104), "--prior-result", str(temporal104),
            "--output-dir", str(output/"near-flow-temporal")], root, log,
            quarter104)

        parent106 = output/"n128/loading/checkpoint_000106.npz"
        peak_dir = output/"peak-temporal-106"
        peak_temporal = peak_dir/"near_flow_temporal.json"
        run_stage(state, state_path, "N128_PEAK_106_FULL_HALF", [
            sys.executable, "full_model/analysis/run_v53_near_flow_temporal.py",
            "--checkpoint", str(parent106), "--expected-interval", "106",
            "--output-dir", str(peak_dir)], root, log, peak_temporal)
        peak = json.loads(peak_temporal.read_text())
        if peak["quarter_step_branch_required"]:
            run_stage(state, state_path, "N128_PEAK_106_QUARTER", [
                sys.executable,
                "full_model/analysis/run_v53_near_flow_quarter.py",
                "--checkpoint", str(parent106), "--prior-result",
                str(peak_temporal), "--output-dir", str(peak_dir)], root, log,
                peak_dir/"near_flow_quarter.json")
        else:
            state["stages"]["N128_PEAK_106_QUARTER"] = {
                "classification": "NOT_REQUIRED_SIGN_RESOLVED_BY_FULL_HALF"}
            persist(state_path, state)

        bulk_state = verification/"v53_bulk_manager_handoff.json"
        bulk_command = [
            sys.executable,
            "full_model/analysis/run_v53_bulk_manager.py",
            "--n128-checkpoint", str(parent104),
            "--n192-checkpoint", str(args.retained_n192_checkpoint),
            "--retained-n128-root", str(args.retained_n128_root),
            "--initialization-audit", str(args.initialization_audit),
            "--output-root", str(output), "--manager-state", str(bulk_state),
            "--n192-targets", "52,81,104", "--n192-wall-budget-s", "28800",
            "--poll-s", str(args.poll_s)]
        bulk_terminal = False
        if bulk_state.exists():
            while True:
                bulk_payload = json.loads(bulk_state.read_text())
                bulk_terminal = bulk_payload.get("state") == (
                    "BULK_COMPLETE_AT_ACHIEVED_SCOPE")
                if bulk_terminal:
                    break
                identity = bulk_payload.get("manager", {})
                if not same_identity(
                        identity, process_identity(identity.get("pid", -1))):
                    break
                state["stages"]["N192_EXTENSION"] = {
                    "classification": "ATTACHED_TO_HEALTHY_EXISTING_MANAGER",
                    "manager": identity}
                persist(state_path, state, "WAITING_FOR_N192_EXTENSION")
                time.sleep(args.poll_s)
        if bulk_terminal:
            state["stages"]["N192_EXTENSION"] = {
                "classification": "VALID_EXISTING_TERMINAL",
                "output": str(bulk_state), "sha256": digest(bulk_state)}
            persist(state_path, state, "COMPLETED_N192_EXTENSION")
        else:
            run_stage(state, state_path, "N192_EXTENSION", bulk_command,
                      args.bulk_source_worktree, log, None)
            if (not bulk_state.exists() or json.loads(
                    bulk_state.read_text()).get("state") !=
                    "BULK_COMPLETE_AT_ACHIEVED_SCOPE"):
                raise RuntimeError("n192 extension manager lacks a valid terminal")
            state["stages"]["N192_EXTENSION"].update({
                "output": str(bulk_state), "sha256": digest(bulk_state)})
            persist(state_path, state, "COMPLETED_N192_EXTENSION")

        geometry = verification/"v53_conjugate_geometry.json"
        geometry_checkpoint = output/"geometry/common_clock_half.npz"
        run_stage(state, state_path, "CONJUGATE_GEOMETRY", [
            sys.executable,
            "full_model/analysis/run_v53_conjugate_geometry.py",
            "--output", str(geometry), "--checkpoint",
            str(geometry_checkpoint), "--duration-s", "2e-9"], root, log,
            geometry)

        n192_manifest = json.loads((output/"n192/loading/run_manifest.json").read_text())
        n192_interval = int(n192_manifest["completed_intervals"])
        diagnostics = verification/"v53_checkpoint_diagnostics.json"
        specifications = [
            f"n128_i032={args.retained_n128_root/'checkpoint_000032.npz'}",
            f"n128_i052={args.retained_n128_root/'checkpoint_000052.npz'}",
            f"n128_i081={args.retained_n128_root/'checkpoint_000081.npz'}",
            f"n128_i104={args.retained_n128_root/'checkpoint_000104.npz'}",
            f"n128_i128={output/'n128/loading/checkpoint_000128.npz'}",
            f"n192_i{n192_interval:03d}={output/'n192/loading'/f'checkpoint_{n192_interval:06d}.npz'}",
        ]
        diagnostic_command = [
            sys.executable,
            "full_model/analysis/run_v53_checkpoint_diagnostics.py"]
        for specification in specifications:
            diagnostic_command.extend(["--checkpoint", specification])
        diagnostic_command.extend(["--output", str(diagnostics)])
        run_stage(state, state_path, "CHECKPOINT_DIAGNOSTICS",
                  diagnostic_command, root, log, diagnostics)

        persist(state_path, state, "RUNNING_CANONICAL_REGRESSION")
        started = time.perf_counter()
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "tests"], cwd=root,
            capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": "src:."})
        with Path(log).open("a") as stream:
            stream.write(result.stdout); stream.write(result.stderr)
        matches = re.findall(r"(\d+) passed", result.stdout)
        state["canonical_regression"] = {
            "returncode": result.returncode,
            "wall_seconds": time.perf_counter()-started,
            "tests_passed": int(matches[-1]) if matches else None,
        }
        if result.returncode:
            raise RuntimeError("canonical regression failed")
        state["state"] = "COMPLETE_AT_ACHIEVED_SCOPE"
        state["completed_utc"] = utc(); state.pop("failure", None)
        persist(state_path, state)
    except Exception as error:
        state["state"] = "FAILED_CONTROLLER"
        state["failure"] = f"{type(error).__name__}: {error}"
        state["failed_utc"] = utc(); persist(state_path, state)
        raise
    finally:
        lock.release()


if __name__ == "__main__":
    main()
