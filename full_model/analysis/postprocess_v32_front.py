#!/usr/bin/env python3
"""Close the V31 continuation and assemble V32 front decision evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def _json(path):
    return json.loads(Path(path).read_text())


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _step(path):
    with np.load(path, allow_pickle=True) as data:
        return int(data["step"])


def _latest(case_root):
    candidates = []
    for path in Path(case_root).glob("attempt-*/*.npz"):
        try:
            candidates.append((_step(path), path.stat().st_mtime_ns, path))
        except (OSError, ValueError, KeyError):
            pass
    return max(candidates) if candidates else None


def _case_record(root, case):
    case_root = root/case["id"]
    status_path = case_root/"case_status.json"
    if not status_path.exists():
        return {"case": case["id"], "grid": case["grid"],
                "state": "NOT_STARTED", "classification": "RESTARTABLE"}
    status = _json(status_path)
    latest = _latest(case_root)
    state = status["state"]
    if state == "PASSED":
        classification = "COMPLETE"
    elif state == "PASSED_TERMINAL":
        classification = "UNSUPPORTED_TOPOLOGY_EVENT"
    elif state in {"RUNNING", "PREFLIGHT_PASSED"}:
        classification = "RESTARTABLE"
    else:
        classification = "EXECUTION_OR_CHECKPOINT_FAILURE"
    result = {
        "case": case["id"], "grid": case["grid"],
        "comparison": case["comparison"], "state": state,
        "classification": classification, "source_sha": status.get("source_sha"),
        "attempt": status.get("attempt"), "final_step": status.get("final_step"),
        "target_step": case["target_step"],
    }
    if latest:
        result.update(latest_checkpoint=str(latest[2]), latest_step=latest[0],
                      latest_checkpoint_sha256=_sha256(latest[2]))
    if status.get("terminal_event"):
        terminal = status["terminal_event"]
        result["terminal_classification"] = terminal.get("classification")
        result["maximum_abs_line_closure_m"] = terminal[
            "coupled_front_ledger"]["maximum_abs_line_closure_m"]
        result["maximum_abs_signed_closure_m2"] = terminal[
            "coupled_front_ledger"]["maximum_abs_signed_closure_m2"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--continuation-root", type=Path, required=True)
    parser.add_argument("--original-terminal", type=Path, required=True)
    parser.add_argument("--unlimited-replay-terminal", type=Path, required=True)
    parser.add_argument("--backtrack-checkpoint", type=Path, required=True)
    parser.add_argument("--backtrack-terminal", type=Path, required=True)
    parser.add_argument("--n128-replay-root", type=Path, required=True)
    parser.add_argument("--kinetics-screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--v32-source-commit", required=True)
    args = parser.parse_args()
    manifest = _json(args.manifest)
    records = [_case_record(args.continuation_root, case)
               for case in manifest["cases"]]
    closed = all(row["state"] in {"PASSED", "PASSED_TERMINAL"}
                 for row in records)
    original = _json(args.original_terminal)
    original_event = original["front_decision"]["topology_event"]
    satellite = min(original_event["new_components"],
                    key=lambda item: item["interface_length_cells"])
    unlimited = _json(args.unlimited_replay_terminal)
    backtrack_terminal = _json(args.backtrack_terminal)
    with np.load(args.backtrack_checkpoint, allow_pickle=True) as data:
        experiment = json.loads(str(data["sibm_experiment_json"].item()))
        coupled = json.loads(str(data["coupled_front_metadata_json"].item()))
    accepted = experiment["front_last_decision"]
    screen = _json(args.kinetics_screen)
    n128_replays = []
    for case_id in ("a1_n128_delta_p1e-6", "a1_n128_delta_m1e-6"):
        raw_paths = sorted((args.continuation_root/case_id).glob(
            "attempt-*/sibm_terminal_event.json"))
        replay_path = args.n128_replay_root/case_id/"sibm_terminal_event.json"
        raw = _json(raw_paths[-1]); replay = _json(replay_path)
        raw_event = raw["front_decision"]["topology_event"]
        loops = [item for item in raw_event["new_components"] if item["closed"]]
        n128_replays.append({
            "case": case_id,
            "raw_v31_classification": raw["classification"],
            "raw_v31_step": raw["step"],
            "raw_ray_crossings_before": raw["front_decision"][
                "ray_crossing_count_before"],
            "raw_ray_crossings_after": raw["front_decision"][
                "ray_crossing_count_after"],
            "closed_loop_area_cells2": loops[0]["enclosed_area_cells2"],
            "closed_loop_length_cells": loops[0]["interface_length_cells"],
            "v32_replay_classification": replay["classification"],
            "v32_replay_step": replay["step"],
            "v32_filtered_subcell_components": replay["front_decision"][
                "filtered_subcell_components_after"],
            "scientific_classification": "RESOLVED_OSCILLATORY_PHASE_ISLAND",
            "raw_sha256": _sha256(raw_paths[-1]),
            "replay_sha256": _sha256(replay_path),
        })
    launch_path = args.continuation_root/"launch_status.json"
    launch = _json(launch_path) if launch_path.exists() else None
    result = {
        "schema": "asb-drx/v32-front-decision/v1",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "authority": {
            "v31_execution_source_commit": (
                records[0].get("source_sha") if records else None),
            "v30_frozen_archive_source_commit": manifest[
                "frozen_archive_source_sha"],
            "v32_front_source_commit": args.v32_source_commit,
        },
        "manager_audit": {
            "launch_status": launch,
            "launch_status_was_premature": True,
            "reason": ("launch_status reported COMPLETED_WITH_TERMINALS while "
                       "case processes remained RUNNING and two cases had not started"),
            "all_ten_cases_closed": closed,
        },
        "v31_cases": records,
        "step135_terminal": {
            "classification": original["classification"],
            "step": original["step"],
            "satellite_enclosed_area_cells2": satellite["enclosed_area_cells2"],
            "satellite_interface_length_cells": satellite[
                "interface_length_cells"],
            "scientific_classification": "UNRESOLVED_SUBCELL_PHASE_ISLAND",
            "source_sha256": _sha256(args.original_terminal),
        },
        "production_replay": {
            "ordinary_split_supported": True,
            "unlimited_replay_terminal": unlimited["classification"],
            "unlimited_replay_step": unlimited["step"],
            "ray_crossings_before": unlimited["front_decision"][
                "ray_crossing_count_before"],
            "ray_crossings_after": unlimited["front_decision"][
                "ray_crossing_count_after"],
            "topology_limited_step": int(_step(args.backtrack_checkpoint)),
            "accepted_backtrack_fraction": accepted[
                "topology_backtrack_fraction"],
            "accepted_maximum_abs_line_closure_m": coupled["ledger"][
                "maximum_abs_line_closure_m"],
            "accepted_maximum_abs_signed_closure_m2": coupled["ledger"][
                "maximum_abs_signed_closure_m2"],
            "following_terminal": backtrack_terminal["classification"],
            "following_terminal_step": backtrack_terminal["step"],
            "classification": "UNRESOLVED_DIFFUSE_FRAGMENTATION",
        },
        "n128_terminal_replays": n128_replays,
        "kinetics_screen": {
            "sha256": _sha256(args.kinetics_screen),
            "robust_candidates": screen["robust_candidates"],
            "selected_candidate": screen["selected_candidate"],
            "promotion_decision": screen["promotion_decision"],
            "kinetics_discriminated": screen[
                "kinetics_discriminated_by_three_step_screen"],
        },
        "common_front_integration": {
            "requested_mode": "v32_existing_boundary_common_state",
            "adapter_active": False,
            "production_behavior": "FAIL_FAST_UNSUPPORTED_COMMON_FRONT_DUAL_OWNER",
            "overlapping_owners": [
                "signed_forest_wall", "beta_nye_alignment",
                "junction_state", "sparse_front_reservoirs"],
            "label_allocation_allowed": False,
            "integrated_run_authorized": False,
        },
        "fixture_passed": True,
        "scientific_gate_passed": False,
        "long_front_run_authorized": False,
        "classification": (
            "V31_CLOSED_FRONT_FRAGMENTATION_NOT_PROMOTED" if closed
            else "V31_CONTINUATION_ACTIVE_FRONT_FRAGMENTATION_NOT_PROMOTED"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"closed": closed, "classification": result[
        "classification"]}, indent=2))
    return 0 if closed else 3


if __name__ == "__main__":
    raise SystemExit(main())
