#!/usr/bin/env python3
"""Fail-closed, substep-checkpointed V46 common-state trajectory runner."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from full_model.analysis.run_v34_finite_coupled_response import (
    I3Controls, resolved_bicrystal, run_i3_cycle,
)
from full_model.analysis.run_v36_recurrent_physical_response import driving_at_time
from full_model.analysis.run_v39_common_horizon import (
    front_stage, load_stage, save_stage,
)


SCHEMA = "asb-drx/v46/checkpointed-horizon/v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
    temporary.replace(path)


def configuration(grid, macro_dt_s, substeps_per_half, source_sha):
    return {
        "grid": int(grid), "domain_m": 3.2e-6,
        "interface_width_m": 4e-7, "temperature_K": 1100.0,
        "child_line_fraction": .35, "macro_dt_s": float(macro_dt_s),
        "substeps_per_half": int(substeps_per_half),
        "mura_transport_operator": "compatible_dealiased",
        "source_sha": source_sha,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid", type=int, choices=(128, 192), required=True)
    parser.add_argument("--macro-dt-s", type=float, default=7.8125e-6)
    parser.add_argument("--substeps-per-half", type=int, default=8)
    args = parser.parse_args()
    if args.substeps_per_half <= 0 or args.macro_dt_s <= 0.0:
        raise ValueError("positive horizon and substep count required")
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip()
    config = configuration(
        args.grid, args.macro_dt_s, args.substeps_per_half, source)
    context = resolved_bicrystal(
        grid=args.grid, length_m=config["domain_m"],
        interface_width_m=config["interface_width_m"],
        temperature_K=config["temperature_K"],
        child_line_fraction=config["child_line_fraction"])
    manifest_path = output/"run_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != SCHEMA:
            raise ValueError("unsupported V46 trajectory manifest")
        if manifest.get("configuration") != config:
            raise ValueError("restart configuration or source mismatch")
        completed = int(manifest["completed_operation_count"])
        checkpoint = Path(manifest["latest_checkpoint"])
        if not checkpoint.is_file() or digest(checkpoint) != manifest[
                "latest_checkpoint_sha256"]:
            raise ValueError("restart checkpoint path or checksum mismatch")
        state, metadata = load_stage(checkpoint, context)
        if int(metadata.get("completed_operation_count", -1)) != completed:
            raise ValueError("restart checkpoint operation mismatch")
    else:
        state = context["state"]
        completed = 0
        checkpoint = output/"operation_000_initial.npz"
        save_stage(checkpoint, state, context, {
            "source_sha": source, "stage": "initial", "grid": args.grid,
            "physical_time_s": 0.0, "completed_operation_count": 0,
            "records": [],
        })
        manifest = {
            "schema": SCHEMA,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "configuration": config, "completed_operation_count": 0,
            "physical_time_s": 0.0, "latest_checkpoint": str(checkpoint),
            "latest_checkpoint_sha256": digest(checkpoint), "operations": [],
            "terminal_state": "RUNNING",
        }
        atomic_json(manifest_path, manifest)

    total_mura = 2*args.substeps_per_half
    substep_dt = .5*args.macro_dt_s/args.substeps_per_half
    total_operations = total_mura+1  # one front between the two halves
    while completed < total_operations:
        operation = completed+1
        if operation == args.substeps_per_half+1:
            physical_time = .5*args.macro_dt_s
            driving = driving_at_time(
                args.grid, .01, "hold", 0.0, physical_time)
            started = time.perf_counter()
            state, ledger = front_stage(
                context, state, args.macro_dt_s, driving, .0625, 1.0, 1)
            elapsed = time.perf_counter()-started
            record = {
                "operation": operation, "kind": "front",
                "physical_time_s": physical_time,
                "operator_exposure_s": args.macro_dt_s,
                "wall_seconds": elapsed,
                "published": bool(ledger["candidate_sweep_published"]),
                "classification": ledger["front_decision"]["classification"],
            }
        else:
            mura_index = operation if operation <= args.substeps_per_half else operation-1
            physical_time = (mura_index-.5)*substep_dt
            driving = driving_at_time(
                args.grid, .01, "hold", 0.0, physical_time)
            started = time.perf_counter()
            state, audit = run_i3_cycle(
                context, state, state.eta.copy(), driving,
                I3Controls(
                    mura_enabled=True, front_enabled=False,
                    trial_dt_s=substep_dt, front_dt_s=substep_dt,
                    mura_transport_operator="compatible_dealiased"))
            elapsed = time.perf_counter()-started
            ledger = audit["mura"]
            accepted = float(ledger["accepted_dt_s"])
            tolerance = 64*np.finfo(float).eps*substep_dt
            if accepted <= tolerance or abs(accepted-substep_dt) > tolerance:
                raise RuntimeError("Mura operation failed exact clock closure")
            record = {
                "operation": operation, "kind": "mura",
                "mura_index": mura_index,
                "physical_time_s": mura_index*substep_dt,
                "operator_exposure_s": accepted,
                "wall_seconds": elapsed,
                "event_scale": float(ledger["event_scale"]),
                "family_event_scales": [float(x) for x in
                                         ledger["family_event_scales"]],
                "ordering": {
                    "stiff_dispatch": ledger.get("ordering_stiff_dispatch"),
                    "integration_method": ledger.get(
                        "ordering_integration_method"),
                    "active_degrees_of_freedom": ledger.get(
                        "ordering_active_degrees_of_freedom"),
                    "maximum_attempt_exposure": ledger.get(
                        "ordering_maximum_attempt_exposure"),
                    "asymptotic_state_accessibility_passed": ledger.get(
                        "ordering_asymptotic_state_accessibility_passed"),
                    "asymptotic_accessibility_maximum_violation": ledger.get(
                        "ordering_asymptotic_accessibility_maximum_violation"),
                    "asymptotic_trial_normalized_remainder": ledger.get(
                        "ordering_asymptotic_trial_normalized_remainder"),
                    "asymptotic_trial_projected_change": ledger.get(
                        "ordering_asymptotic_trial_projected_change"),
                },
            }
        checkpoint = output/f"operation_{operation:03d}_{record['kind']}.npz"
        save_stage(checkpoint, state, context, {
            "source_sha": source, "stage": record["kind"],
            "grid": args.grid, "physical_time_s": record["physical_time_s"],
            "completed_operation_count": operation,
            "records": [record],
        })
        manifest["operations"].append(record)
        manifest["completed_operation_count"] = operation
        manifest["physical_time_s"] = record["physical_time_s"]
        manifest["latest_checkpoint"] = str(checkpoint)
        manifest["latest_checkpoint_sha256"] = digest(checkpoint)
        manifest["terminal_state"] = (
            "COMPLETE" if operation == total_operations else "RUNNING")
        atomic_json(manifest_path, manifest)
        completed = operation
        print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
