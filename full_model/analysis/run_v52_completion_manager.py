#!/usr/bin/env python3
"""Durably sequence V52 continuation, companion grid, analysis, and tests."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    os.replace(temporary, path)


def run(command, cwd, log):
    started = time.perf_counter()
    with Path(log).open("a") as stream:
        stream.write("COMMAND "+json.dumps(command)+"\n"); stream.flush()
        result = subprocess.run(command, cwd=cwd, stdout=stream,
                                stderr=subprocess.STDOUT, text=True,
                                env={**os.environ, "PYTHONPATH": "src:."})
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {command}")
    return time.perf_counter()-started


def matching_primary_process(output_root):
    listing = subprocess.run(["ps", "-axo", "pid=,command="],
                             capture_output=True, text=True, check=True).stdout
    needle = str(Path(output_root).resolve())
    return [line.strip() for line in listing.splitlines()
            if "run_v49_physical_continuation.py" in line and needle in line]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-output", type=Path, required=True)
    parser.add_argument("--manager-state", type=Path, required=True)
    parser.add_argument("--poll-s", type=float, default=60.0)
    args = parser.parse_args()
    root = Path.cwd(); verification = root/"full_model/verification"
    log = verification/"v52_completion_manager.log"
    lock = args.manager_state.with_suffix(".lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError("V52 completion manager lock already exists") from error
    os.write(descriptor, f"pid={os.getpid()}\n".encode()); os.close(descriptor)
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    state = {
        "schema": "asb-drx/v52/completion-manager/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "manager_pid": os.getpid(), "manager_source_sha": source,
        "state": "WAITING_FOR_N128_CONTINUATION", "history": [],
        "primary_output": str(args.primary_output.resolve()),
    }
    atomic_json(args.manager_state, state)
    try:
        manifest_path = args.primary_output/"run_manifest.json"
        while True:
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                checkpoint = Path(manifest["latest_checkpoint"])
                checksum_ok = (checkpoint.exists() and digest(checkpoint)
                               ==manifest["latest_checkpoint_sha256"])
                state.update({
                    "primary_status": manifest["status"],
                    "primary_completed_intervals": manifest[
                        "completed_intervals"],
                    "primary_checkpoint_checksum_verified": checksum_ok,
                    "primary_process_matches": matching_primary_process(
                        args.primary_output),
                    "last_poll_utc": datetime.now(timezone.utc).isoformat(),
                })
                atomic_json(args.manager_state, state)
                if manifest["status"] == "COMPLETE":
                    if manifest["completed_intervals"] != 104 or not checksum_ok:
                        raise RuntimeError("primary terminal manifest is invalid")
                    break
            time.sleep(args.poll_s)

        state["state"] = "RUNNING_N192_COMMON_INITIAL_MACRO"
        atomic_json(args.manager_state, state)
        retained_initial = Path(
            "/Users/sdillon/HPC3/worktrees/asb-drx-full-v49-20260921/"
            "full_model/production/results-local/v49-physical/n128/initial.npz")
        spatial_root = root/"full_model/production/results-local/v52-spatial/n192"
        initial = spatial_root/"initial.npz"
        init_evidence = verification/"v52_spatial_initialization.json"
        elapsed = run([
            "python", "full_model/analysis/run_v52_analytic_spatial_initialization.py",
            "--retained-n128", str(retained_initial),
            "--companion-checkpoint", str(initial),
            "--evidence", str(init_evidence),
        ], root, log)
        state["history"].append({"stage": "ANALYTIC_N192_INITIAL", "wall_s": elapsed})
        elapsed = run([
            "python", "full_model/analysis/run_v49_physical_continuation.py",
            "--initial-checkpoint", str(initial),
            "--output-dir", str(spatial_root/"loading"),
            "--grid", "192", "--intervals", "1", "--dt-s", "4.8828125e-7",
            "--protocol", "continued_deformation", "--strain-rate-s", "100",
        ], root, log)
        state["history"].append({"stage": "N192_FIRST_MACRO", "wall_s": elapsed})
        state["state"] = "POSTPROCESSING"
        atomic_json(args.manager_state, state)
        n128_root = Path(
            "/Users/sdillon/HPC3/worktrees/asb-drx-full-v49-20260921/"
            "full_model/production/results-local/v49-physical/n128/loading")
        elapsed = run([
            "python", "full_model/analysis/run_v52_spatial_comparison.py",
            "--n128-checkpoint", str(n128_root/"checkpoint_000001.npz"),
            "--n128-manifest", str(n128_root/"run_manifest.json"),
            "--n192-checkpoint", str(spatial_root/"loading/checkpoint_000001.npz"),
            "--n192-manifest", str(spatial_root/"loading/run_manifest.json"),
            "--initialization-audit", str(init_evidence),
            "--output", str(verification/"v52_spatial_comparison.json"),
        ], root, log)
        state["history"].append({"stage": "SPATIAL_ANALYSIS", "wall_s": elapsed})
        elapsed = run([
            "python", "full_model/analysis/run_v52_continuation_analysis.py",
            "--manifest", str(manifest_path),
            "--output", str(verification/"v52_continuation.json"),
            "--plot", str(verification/"v52_continuation.png"),
        ], root, log)
        state["history"].append({"stage": "CONTINUATION_ANALYSIS", "wall_s": elapsed})
        state["state"] = "RUNNING_CANONICAL_REGRESSION"
        atomic_json(args.manager_state, state)
        elapsed = run(["python", "-m", "pytest", "-q", "tests"], root, log)
        state["history"].append({"stage": "CANONICAL_REGRESSION", "wall_s": elapsed})
        state["state"] = "COMPLETE"
        state["completed_utc"] = datetime.now(timezone.utc).isoformat()
        state["outputs"] = {
            name: {"path": str(path.resolve()), "sha256": digest(path)}
            for name, path in {
                "primary_manifest": manifest_path,
                "n192_manifest": spatial_root/"loading/run_manifest.json",
                "spatial_initialization": init_evidence,
                "spatial_comparison": verification/"v52_spatial_comparison.json",
                "continuation": verification/"v52_continuation.json",
                "continuation_plot": verification/"v52_continuation.png",
            }.items()
        }
        atomic_json(args.manager_state, state)
    except Exception as error:
        state["state"] = "FAILED_INFRASTRUCTURE_OR_NUMERICAL"
        state["failure"] = f"{type(error).__name__}: {error}"
        state["failed_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(args.manager_state, state)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
