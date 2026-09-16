#!/usr/bin/env python3
"""Restart one selected V31 front continuation without touching V30 evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = ROOT/"full_model"/"production"
DEFAULT_MANIFEST = Path(__file__).with_name(
    "v31_front_continuation_manifest.json")


def _utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
    os.replace(temporary, path)


def _git_identity():
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT, text=True).strip())
    return sha, dirty


def _checkpoint(path):
    with np.load(path, allow_pickle=True) as data:
        if "coupled_front_metadata_json" not in data.files:
            raise ValueError(f"checkpoint lacks coupled-front state: {path}")
        return int(data["step"])


def _latest_local(case_root):
    candidates = []
    for path in case_root.glob("attempt-*/*.npz"):
        try:
            candidates.append((_checkpoint(path), path.stat().st_mtime_ns, path))
        except (OSError, ValueError):
            continue
    return sorted(candidates)[-1] if candidates else None


def _selected_case(manifest, case_id):
    matches = [case for case in manifest["cases"] if case["id"] == case_id]
    if len(matches) != 1:
        raise ValueError(f"case {case_id!r} is not uniquely selected")
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--expected-source-sha")
    parser.add_argument("--allow-dirty-source", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.resolve().read_text())
    case = _selected_case(manifest, args.case_id)
    if "equal" in case["id"] and "delta" not in case["id"]:
        raise ValueError("completed equal/off controls cannot enter V31 continuation")

    archive = Path(manifest["frozen_archive_root"]).resolve()
    original_checkpoint = archive/case["checkpoint"]
    parameters_path = archive/case["parameters"]
    if (_sha256(original_checkpoint) != case["checkpoint_sha256"]
            or _sha256(parameters_path) != case["parameters_sha256"]):
        raise ValueError("frozen V30 input checksum mismatch")
    original_status = json.loads(
        (archive/case["id"]/"case_status.json").read_text())
    if original_status.get("state") != "FAILED_SCIENTIFIC":
        raise ValueError("continuation input was not a failed V30 case")

    source_sha, source_dirty = _git_identity()
    if not args.allow_dirty_source:
        if not args.expected_source_sha:
            raise ValueError("long continuation requires an expected source SHA")
        if source_sha != args.expected_source_sha:
            raise ValueError("continuation source SHA differs from requested SHA")
        if source_dirty:
            raise ValueError("continuation requires a clean source tree")
        remote_refs = subprocess.check_output(
            ["git", "branch", "-r", "--contains", source_sha], cwd=ROOT,
            text=True).strip()
        if not remote_refs:
            raise ValueError("continuation source SHA is not on a fetched remote ref")

    case_root = args.output_root.resolve()/case["id"]
    case_root.mkdir(parents=True, exist_ok=True)
    local = _latest_local(case_root)
    restart = local[2] if local else original_checkpoint
    start_step = _checkpoint(restart)+1
    target_step = int(case["target_step"])
    if start_step > target_step:
        return 0
    stop_step = target_step
    if args.max_steps is not None:
        if args.max_steps <= 0:
            raise ValueError("max-steps must be positive")
        stop_step = min(stop_step, start_step+args.max_steps-1)
    steps = stop_step-start_step+1
    attempt = len(list(case_root.glob("attempt-*")))
    run_root = case_root/f"attempt-{attempt:03d}"
    run_root.mkdir(parents=False, exist_ok=False)

    parameters = json.loads(parameters_path.read_text())
    parameters.update(
        nSteps=steps, restart_file=str(restart), restart_reset_clock=False,
        restart_prefix="v31_front_restart", restart_interval=min(100, steps),
        restart_wallclock_interval_s=840.0, save_interval=100000,
        plot_interval=100000, save_main_panels=False,
        save_signed_panels=False, write_field_npz=False)
    config_path = run_root/"v31_params.json"
    config_path.write_text(json.dumps(parameters, indent=2, sort_keys=True)+"\n")
    status = {
        "schema": "asb-drx/v31-front-continuation-status/v1",
        "case": case["id"], "grid": case["grid"],
        "comparison": case["comparison"], "source_sha": source_sha,
        "source_dirty": source_dirty, "archive_source_sha": manifest[
            "frozen_archive_source_sha"],
        "archive_checkpoint": str(original_checkpoint),
        "archive_checkpoint_sha256": case["checkpoint_sha256"],
        "restart": str(restart), "restart_sha256": _sha256(restart),
        "start_step": start_step, "requested_stop_step": stop_step,
        "target_step": target_step, "start_utc": _utc_now(),
        "state": "RUNNING", "attempt": attempt,
    }
    _atomic_json(case_root/"case_status.json", status)
    environment = dict(
        os.environ, DRX_OUTDIR=str(run_root), MPLBACKEND="Agg",
        OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "1"),
        DRX_PARAMS=json.dumps(parameters, separators=(",", ":")))
    with (run_root/"stdout.log").open("w") as stdout, \
            (run_root/"stderr.log").open("w") as stderr:
        completed = subprocess.run(
            [sys.executable, "drx_full_v34_recovery.py"], cwd=PRODUCTION,
            env=environment, stdout=stdout, stderr=stderr, check=False)
    newest = _latest_local(case_root)
    final_step = newest[0] if newest else None
    terminals = list(run_root.glob("sibm_terminal_event.json"))
    status.update(
        end_utc=_utc_now(), returncode=completed.returncode,
        final_step=final_step,
        latest_checkpoint=(str(newest[2]) if newest else None))
    if terminals:
        status["terminal_event"] = json.loads(terminals[0].read_text())
        status["state"] = "PASSED_TERMINAL"
    elif completed.returncode != 0:
        status["state"] = "FAILED_EXECUTION"
    elif final_step != stop_step:
        status["state"] = "FAILED_CHECKPOINT"
    elif stop_step < target_step:
        status["state"] = "PREFLIGHT_PASSED"
    else:
        status["state"] = "PASSED"
    _atomic_json(case_root/"case_status.json", status)
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0 if status["state"] in {
        "PREFLIGHT_PASSED", "PASSED", "PASSED_TERMINAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
