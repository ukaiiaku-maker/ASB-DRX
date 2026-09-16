#!/usr/bin/env python3
"""Classify the bounded V31 topology-aware front continuation."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT/"full_model"/"hpc3"/"v31_front_continuation_manifest.json"
NAMED_TERMINALS = {
    "FRONT_COMPONENT_SPLIT", "FRONT_COMPONENT_MERGE", "GRAIN_CONSUMED",
    "INTERFACE_PAIR_ANNIHILATED", "PAIR_LEFT_ACTIVE_WINDOW",
    "PAIR_IDENTITY_LOST", "PAIR_ENTERED_ACTIVE_WINDOW"}


def _record(case, output_root):
    case_root = output_root/case["id"]
    status_path = case_root/"case_status.json"
    if not status_path.exists():
        return {"case": case["id"], "grid": case["grid"],
                "comparison": case["comparison"], "state": "NOT_STARTED"}
    status = json.loads(status_path.read_text())
    result = {key: status.get(key) for key in (
        "case", "grid", "comparison", "state", "source_sha", "source_dirty",
        "start_step", "final_step", "target_step", "latest_checkpoint")}
    if status.get("start_utc") and status.get("end_utc"):
        start = datetime.fromisoformat(status["start_utc"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(status["end_utc"].replace("Z", "+00:00"))
        completed_steps = max(
            int(status.get("final_step", -1))-int(status["start_step"])+1, 1)
        result["elapsed_s"] = (end-start).total_seconds()
        result["seconds_per_step"] = result["elapsed_s"]/completed_steps
    terminal = status.get("terminal_event") or {}
    result["terminal_classification"] = terminal.get("classification")
    checkpoint = status.get("latest_checkpoint")
    if not checkpoint:
        return result
    with np.load(checkpoint, allow_pickle=True) as data:
        coupled = json.loads(str(data["coupled_front_metadata_json"].item()))
        experiment = json.loads(str(data["sibm_experiment_json"].item()))
    ledger = coupled["ledger"]
    result.update(
        component_count=len(coupled.get("topology", {}).get("components", [])),
        topology_event_count=int(coupled.get("topology_event_count", 0)),
        attempts=int(ledger["attempts"]), accepted=int(ledger["accepted"]),
        rejected_direction=int(ledger["rejected_direction"]),
        signed_volume_m3=float(ledger["a_to_b_swept_volume_m3"]
                               -ledger["b_to_a_swept_volume_m3"]),
        heat_J=float(ledger["heat_J"]),
        maximum_abs_line_closure_m=float(ledger[
            "maximum_abs_line_closure_m"]),
        maximum_abs_signed_closure_m2=float(ledger[
            "maximum_abs_signed_closure_m2"]),
        legacy_afterburner_calls=int(experiment.get(
            "legacy_afterburner_calls", 0)))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    records = [_record(case, args.output_root.resolve())
               for case in manifest["cases"]]
    failures = []
    completed = [item for item in records if item["state"] != "NOT_STARTED"]
    for item in completed:
        if item["state"] not in {"PREFLIGHT_PASSED", "PASSED", "PASSED_TERMINAL"}:
            failures.append(f"{item['case']}:execution")
        terminal = item.get("terminal_classification")
        if terminal and terminal not in NAMED_TERMINALS:
            failures.append(f"{item['case']}:unidentified_terminal")
        if (item.get("legacy_afterburner_calls", 1) != 0
                or item.get("maximum_abs_line_closure_m", np.inf) > 1e-15
                or item.get("maximum_abs_signed_closure_m2", np.inf) > 1.0):
            failures.append(f"{item['case']}:ledger_or_operator")
    by_key = {(item["grid"], item["comparison"]): item for item in completed}
    for grid in (64, 128):
        favorable = by_key.get((grid, "favorable"))
        reversed_case = by_key.get((grid, "reversed"))
        swapped = by_key.get((grid, "label_swap_favorable"))
        plus = by_key.get((grid, "near_equal_plus"))
        minus = by_key.get((grid, "near_equal_minus"))
        if favorable and favorable.get("signed_volume_m3", 0.0) <= 0.0:
            failures.append(f"n{grid}:favorable_sign")
        if reversed_case and reversed_case.get("signed_volume_m3", 0.0) >= 0.0:
            failures.append(f"n{grid}:reversed_sign")
        if swapped and swapped.get("signed_volume_m3", 0.0) <= 0.0:
            failures.append(f"n{grid}:label_swap_sign")
        if plus and minus:
            p = plus.get("signed_volume_m3", 0.0)
            m = minus.get("signed_volume_m3", 0.0)
            if not (p > 0.0 and m < 0.0):
                failures.append(f"n{grid}:near_equal_odd_sign")
    all_started = len(completed) == len(records)
    hard_gate = all_started and not failures
    remaining_case_hours = {
        item["case"]: (max(int(item["target_step"])-int(item["final_step"]), 0)
                       *float(item["seconds_per_step"])/3600.0)
        for item in completed if item.get("seconds_per_step") is not None}
    result = {
        "schema": "asb-drx/v31-front-continuation-decision/v1",
        "records": records, "all_selected_cases_started": all_started,
        "preflight_passed": hard_gate,
        "long_continuation_authorized": hard_gate,
        "a2_authorized": False,
        "estimated_remaining_case_hours": remaining_case_hours,
        "estimated_serial_remaining_hours": sum(remaining_case_hours.values()),
        "estimated_ideal_concurrent_wall_hours": max(
            remaining_case_hours.values(), default=0.0),
        "failures": failures,
        "classification": (
            "TOPOLOGY_AWARE_FRONT_CONTINUATION_PREFLIGHT_PASSED"
            if hard_gate else "V31_FRONT_CONTINUATION_INCOMPLETE"
            if not all_started else "V31_FRONT_CONTINUATION_FAILED"),
        "promotion_decision": (
            "PROMOTE_SELECTED_A1_CASES_TO_RESTARTABLE_LONG_CONTINUATION"
            if hard_gate else "DO_NOT_PROMOTE"),
        "note": "A2 remains blocked until selected cases reach their full horizons."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({key: result[key] for key in (
        "classification", "preflight_passed", "long_continuation_authorized",
        "a2_authorized", "failures")}, indent=2))
    return 0 if hard_gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
