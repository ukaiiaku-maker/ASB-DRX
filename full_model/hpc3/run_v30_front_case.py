#!/usr/bin/env python3
"""Fault-isolated, restartable executor for one V30 front case."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = ROOT/"full_model"/"production"
DEFAULT_MANIFEST = Path(__file__).with_name("v30_front_case_manifest.json")
_child = None
_signal_received = None


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_identity():
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT, text=True).strip()
    return sha, bool(dirty)


def select_case(manifest, case_id, case_index):
    cases = manifest["cases"]
    if case_id is not None:
        match = [case for case in cases if case["id"] == case_id]
        if len(match) != 1:
            raise ValueError(f"case id {case_id!r} is not unique")
        return match[0], cases.index(match[0])
    if case_index is None:
        value = os.environ.get("V30_CASE_INDEX", os.environ.get(
            "SLURM_ARRAY_TASK_ID"))
        if value is None:
            raise ValueError("case id/index or scheduler array index is required")
        case_index = int(value)
    return cases[int(case_index)], int(case_index)


def checkpoint_records(case_root):
    records = []
    for path in case_root.glob("segments/segment-*/attempt-*/*.npz"):
        try:
            with np.load(path, allow_pickle=True) as data:
                if "step" not in data.files:
                    continue
                step = int(data["step"])
                coupled = "coupled_front_metadata_json" in data.files
            if coupled:
                records.append((step, path.stat().st_mtime_ns, path, coupled))
        except Exception:
            # Interrupted atomic checkpoint publication or filesystem damage;
            # retain the file as evidence but never select it for restart.
            continue
    return sorted(records)


def segment_for_step(case, next_step):
    start = 0
    for index, segment in enumerate(case["segments"]):
        end = start+int(segment["steps"])
        if next_step < end:
            return index, start, end, segment
        start = end
    return None


def build_parameters(manifest, case, segment, remaining_steps, restart):
    result = dict(manifest["base_parameters"])
    grid = int(case["grid"])
    delta = case.get("relative_delta")
    if delta is None:
        rho_a = float(case["rho_a_m2"])
        rho_b = float(case["rho_b_m2"])
    else:
        base = 2.5e17
        rho_a = base*(1.0+0.5*float(delta))
        rho_b = base*(1.0-0.5*float(delta))
    result.update(
        Nx=grid, Ny=grid, nSteps=int(remaining_steps),
        T0=float(case.get("temperature_K", 1100.0)),
        sibm_clean_parent_density_m2=rho_a,
        sibm_clean_child_density_m2=rho_b,
        sibm_mobility_multiplier=float(case["mobility"]),
        sibm_parent_label_override=int(case.get("parent_label", 0)),
        sibm_child_label_override=int(case.get("child_label", 1)),
        sibm_applied_pressure_Pa=float(segment["applied_pressure_Pa"]),
        restart_file=(str(restart) if restart is not None else None),
        restart_reset_clock=(restart is None))
    return result


def _forward_signal(signum, _frame):
    global _signal_received
    _signal_received = int(signum)
    if _child is not None and _child.poll() is None:
        _child.send_signal(signal.SIGTERM)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--case-id")
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--output-root", type=Path,
                        default=ROOT/"full_model"/"production"/"output"/
                        "v30_front")
    parser.add_argument("--allow-unverified-source", action="store_true")
    parser.add_argument("--allow-blocked", action="store_true")
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    case, case_index = select_case(manifest, args.case_id, args.case_index)
    if (case["dependency_state"] == "BLOCKED_BY_DEPENDENCY"
            and not args.allow_blocked
            and os.environ.get("V30_A2_AUTHORIZED") != "1"):
        raise SystemExit("case is BLOCKED_BY_DEPENDENCY; A1 promotion is required")

    source_sha, source_dirty = git_identity()
    expected_sha = os.environ.get("V30_EXPECTED_SOURCE_SHA")
    expected_remote_sha = os.environ.get("V30_EXPECTED_REMOTE_SHA")
    if not args.allow_unverified_source:
        if not expected_sha or not expected_remote_sha:
            raise SystemExit(
                "V30_EXPECTED_SOURCE_SHA and V30_EXPECTED_REMOTE_SHA are required")
        if (source_dirty or source_sha != expected_sha
                or source_sha != expected_remote_sha):
            raise SystemExit(
                f"source identity failure: head={source_sha}, expected={expected_sha}, "
                f"remote={expected_remote_sha}, dirty={source_dirty}")

    case_root = (args.output_root/case["id"]).resolve()
    case_root.mkdir(parents=True, exist_ok=True)
    status_path = case_root/"case_status.json"
    provenance = {
        "schema": "asb-drx/v30-front-case-status/v1",
        "case": case["id"], "case_index": case_index,
        "tier": case["tier"], "source_sha": source_sha,
        "source_dirty": source_dirty,
        "expected_source_sha": expected_sha,
        "expected_remote_sha": expected_remote_sha,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "host": os.uname().nodename,
        "start_utc": utc_now(), "state": "RUNNING", "runs": [],
    }
    if status_path.exists():
        try:
            previous = json.loads(status_path.read_text())
            provenance["runs"] = previous.get("runs", [])
            provenance["original_start_utc"] = previous.get(
                "original_start_utc", previous.get("start_utc"))
        except (OSError, ValueError):
            provenance["previous_status_unreadable"] = True
    provenance.setdefault("original_start_utc", provenance["start_utc"])
    atomic_json(status_path, provenance)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, _forward_signal)
    end_epoch = int(os.environ.get("SLURM_JOB_END_TIME", "0") or 0)

    while True:
        checkpoints = checkpoint_records(case_root)
        latest = checkpoints[-1] if checkpoints else None
        next_step = latest[0]+1 if latest else 0
        if _signal_received is not None:
            provenance.update(
                state="PARTIAL_RESTARTABLE", end_utc=utc_now(),
                next_step=next_step, reason=f"received_signal_{_signal_received}",
                latest_checkpoint=str(latest[2]) if latest else None)
            atomic_json(status_path, provenance)
            return 0
        selection = segment_for_step(case, next_step)
        if selection is None:
            provenance.update(state="PASSED", end_utc=utc_now(),
                              final_step=next_step-1,
                              latest_checkpoint=str(latest[2]) if latest else None)
            atomic_json(status_path, provenance)
            return 0
        segment_index, segment_start, segment_end, segment = selection
        if end_epoch and end_epoch-time.time() < 20*60:
            provenance.update(
                state="PARTIAL_RESTARTABLE", end_utc=utc_now(),
                next_step=next_step, reason="less_than_20_minutes_allocation_remaining",
                latest_checkpoint=str(latest[2]) if latest else None)
            atomic_json(status_path, provenance)
            return 0
        attempt_root = case_root/"segments"/f"segment-{segment_index:02d}"
        attempt_index = len(list(attempt_root.glob("attempt-*")))
        run_root = attempt_root/f"attempt-{attempt_index:03d}"
        run_root.mkdir(parents=True, exist_ok=False)
        remaining = segment_end-next_step
        config = build_parameters(
            manifest, case, segment, remaining, latest[2] if latest else None)
        config_path = run_root/"v30_params.json"
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True)+"\n")
        run_record = {
            "segment_index": segment_index, "segment_start": segment_start,
            "segment_end": segment_end, "start_step": next_step,
            "requested_steps": remaining, "start_utc": utc_now(),
            "output": str(run_root), "config": str(config_path),
            "config_sha256": sha256(config_path),
            "restart": str(latest[2]) if latest else None,
            "restart_sha256": sha256(latest[2]) if latest else None,
            "state": "RUNNING",
        }
        provenance["runs"].append(run_record)
        atomic_json(status_path, provenance)
        environment = dict(
            os.environ, DRX_OUTDIR=str(run_root), MPLBACKEND="Agg",
            OMP_NUM_THREADS=os.environ.get("SLURM_CPUS_PER_TASK", "1"),
            DRX_PARAMS=json.dumps(config, separators=(",", ":")))
        with (run_root/"stdout.log").open("w") as stdout, \
                (run_root/"stderr.log").open("w") as stderr:
            global _child
            _child = subprocess.Popen(
                [sys.executable, "drx_full_v34_recovery.py"], cwd=PRODUCTION,
                env=environment, stdout=stdout, stderr=stderr, text=True)
            returncode = _child.wait()
            _child = None
        run_record.update(end_utc=utc_now(), returncode=returncode,
                          signal_received=_signal_received)
        new_checkpoints = checkpoint_records(case_root)
        newest = new_checkpoints[-1] if new_checkpoints else latest
        run_record["latest_checkpoint"] = str(newest[2]) if newest else None
        run_record["latest_step"] = newest[0] if newest else None
        terminal_paths = sorted(run_root.glob("sibm_terminal_event.json"))
        if terminal_paths:
            run_record["terminal_event"] = json.loads(terminal_paths[-1].read_text())
            run_record["state"] = "PASSED_TERMINAL"
            provenance.update(
                state="PASSED", terminal_event=run_record["terminal_event"],
                end_utc=utc_now(), final_step=run_record["latest_step"])
            atomic_json(status_path, provenance)
            return 0
        if returncode != 0:
            run_record["state"] = (
                "PARTIAL_RESTARTABLE" if _signal_received else "FAILED_SCIENTIFIC")
            provenance.update(
                state=run_record["state"], end_utc=utc_now(),
                returncode=returncode,
                latest_checkpoint=str(newest[2]) if newest else None)
            atomic_json(status_path, provenance)
            return 0 if _signal_received else returncode
        if newest is None or newest[0] < segment_end-1:
            run_record["state"] = "FAILED_INFRASTRUCTURE"
            provenance.update(
                state="FAILED_INFRASTRUCTURE", end_utc=utc_now(),
                reason="successful process omitted required terminal checkpoint")
            atomic_json(status_path, provenance)
            return 3
        run_record["state"] = "PASSED"
        atomic_json(status_path, provenance)


if __name__ == "__main__":
    raise SystemExit(main())
