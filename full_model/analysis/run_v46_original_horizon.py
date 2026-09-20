#!/usr/bin/env python3
"""Atomic continuation from one V46 macro through the original 62.5 us horizon."""

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


SCHEMA = "asb-drx/v46/original-horizon/v1"
PARENT_SOURCE = "096159de1478bb259e445c15c11765d9149d7fe0"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary = Path(path).with_suffix(Path(path).suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
    temporary.replace(path)


def plan(macro_dt_s, substeps, total_macros):
    dt = .5*macro_dt_s/substeps
    rows = []
    global_mura = 16
    for macro in range(2, total_macros+1):
        start = (macro-1)*macro_dt_s
        for local in range(1, substeps+1):
            global_mura += 1
            rows.append({"kind": "mura", "macro": macro,
                         "global_mura_index": global_mura,
                         "driving_time_s": start+(local-.5)*dt,
                         "physical_time_s": start+local*dt})
        rows.append({"kind": "front", "macro": macro,
                     "driving_time_s": start+.5*macro_dt_s,
                     "physical_time_s": start+.5*macro_dt_s})
        for local in range(substeps+1, 2*substeps+1):
            global_mura += 1
            rows.append({"kind": "mura", "macro": macro,
                         "global_mura_index": global_mura,
                         "driving_time_s": start+(local-.5)*dt,
                         "physical_time_s": start+local*dt})
    return rows, dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, choices=(128, 192), required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--macro-dt-s", type=float, default=7.8125e-6)
    parser.add_argument("--substeps-per-half", type=int, default=8)
    parser.add_argument("--total-macros", type=int, default=8)
    args = parser.parse_args()
    if args.total_macros < 2 or args.substeps_per_half <= 0:
        raise ValueError("continuation requires at least two macros and positive substeps")
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip()
    configuration = {
        "grid": args.grid, "domain_m": 3.2e-6,
        "interface_width_m": 4e-7, "temperature_K": 1100.0,
        "child_line_fraction": .35, "macro_dt_s": args.macro_dt_s,
        "substeps_per_half": args.substeps_per_half,
        "total_macros": args.total_macros,
        "mura_transport_operator": "compatible_dealiased",
        "continuation_source_sha": source,
        "parent_scientific_source_sha": PARENT_SOURCE,
    }
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    context = resolved_bicrystal(
        grid=args.grid, length_m=configuration["domain_m"],
        interface_width_m=configuration["interface_width_m"],
        temperature_K=configuration["temperature_K"],
        child_line_fraction=configuration["child_line_fraction"])
    operations, dt = plan(
        args.macro_dt_s, args.substeps_per_half, args.total_macros)
    manifest_path = output/"run_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != SCHEMA or manifest.get(
                "configuration") != configuration:
            raise ValueError("continuation restart schema/config/source mismatch")
        completed = int(manifest["completed_operation_count"])
        checkpoint = Path(manifest["latest_checkpoint"])
        if not checkpoint.is_file() or digest(checkpoint) != manifest[
                "latest_checkpoint_sha256"]:
            raise ValueError("continuation restart checksum mismatch")
        state, metadata = load_stage(checkpoint, context)
        if int(metadata.get("completed_operation_count", -1)) != completed:
            raise ValueError("continuation restart operation mismatch")
    else:
        parent = args.parent.resolve()
        state, parent_metadata = load_stage(parent, context)
        if parent_metadata.get("source_sha") != PARENT_SOURCE:
            raise ValueError("parent scientific source mismatch")
        if int(parent_metadata.get("completed_operation_count", -1)) != 17:
            raise ValueError("parent is not the completed first macro")
        if abs(float(parent_metadata["physical_time_s"])-args.macro_dt_s) > 1e-18:
            raise ValueError("parent physical time mismatch")
        completed = 0
        checkpoint = output/"continuation_000_parent.npz"
        save_stage(checkpoint, state, context, {
            "source_sha": source, "parent_source_sha": PARENT_SOURCE,
            "stage": "continuation_parent", "grid": args.grid,
            "physical_time_s": args.macro_dt_s,
            "completed_operation_count": 0, "records": [],
        })
        manifest = {
            "schema": SCHEMA,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "configuration": configuration,
            "parent_checkpoint": str(parent),
            "parent_checkpoint_sha256": digest(parent),
            "completed_operation_count": 0,
            "physical_time_s": args.macro_dt_s,
            "latest_checkpoint": str(checkpoint),
            "latest_checkpoint_sha256": digest(checkpoint),
            "operations": [], "terminal_state": "RUNNING",
        }
        atomic_json(manifest_path, manifest)

    for index in range(completed, len(operations)):
        item = operations[index]
        driving = driving_at_time(
            args.grid, .01, "hold", 0.0, item["driving_time_s"])
        started = time.perf_counter()
        if item["kind"] == "front":
            state, ledger = front_stage(
                context, state, args.macro_dt_s, driving, .0625, 1.0, 1)
            record = {**item, "operation": index+1,
                      "operator_exposure_s": args.macro_dt_s,
                      "wall_seconds": time.perf_counter()-started,
                      "published": bool(ledger["candidate_sweep_published"]),
                      "classification": ledger[
                          "front_decision"]["classification"]}
        else:
            state, audit = run_i3_cycle(
                context, state, state.eta.copy(), driving,
                I3Controls(
                    mura_enabled=True, front_enabled=False,
                    trial_dt_s=dt, front_dt_s=dt,
                    mura_transport_operator="compatible_dealiased"))
            ledger = audit["mura"]
            accepted = float(ledger["accepted_dt_s"])
            tolerance = 64*np.finfo(float).eps*dt
            if accepted <= tolerance or abs(accepted-dt) > tolerance:
                raise RuntimeError("continued Mura operation failed clock closure")
            record = {
                **item, "operation": index+1,
                "operator_exposure_s": accepted,
                "wall_seconds": time.perf_counter()-started,
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
                    "asymptotic_projected_kkt_relative": ledger.get(
                        "ordering_asymptotic_projected_kkt_relative"),
                }}
        checkpoint = output/f"continuation_{index+1:03d}_{item['kind']}.npz"
        save_stage(checkpoint, state, context, {
            "source_sha": source, "parent_source_sha": PARENT_SOURCE,
            "stage": item["kind"], "grid": args.grid,
            "physical_time_s": item["physical_time_s"],
            "completed_operation_count": index+1, "records": [record],
        })
        manifest["operations"].append(record)
        manifest["completed_operation_count"] = index+1
        manifest["physical_time_s"] = item["physical_time_s"]
        manifest["latest_checkpoint"] = str(checkpoint)
        manifest["latest_checkpoint_sha256"] = digest(checkpoint)
        manifest["terminal_state"] = (
            "COMPLETE" if index+1 == len(operations) else "RUNNING")
        atomic_json(manifest_path, manifest)
        print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
