#!/usr/bin/env python3
"""Postprocess complete and partial V30 Tier-B1 Mura/Nye cases."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_history(path):
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def summarize(case_dir):
    status_path = case_dir/"status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {
        "status": "FAILED_INFRASTRUCTURE", "applied_strain": 0.0,
        "step": 0}
    history = read_history(case_dir/"history.jsonl")
    checkpoints = sorted(case_dir.glob("checkpoint_step_*.npz"))
    maximum_dual = max((row["dual_nye_relative_rms"] for row in history),
                       default=status.get("latest_metrics", {}).get(
                           "dual_nye_relative_rms"))
    maximum_continuity = max((row["normalized_line_continuity_residual"]
                              for row in history),
                             default=status.get("latest_metrics", {}).get(
                                 "normalized_line_continuity_residual"))
    invariant = all(row["accepted_step_hard_invariant_passed"]
                    and not row["post_step_projection_used"]
                    for row in history)
    reached_10 = float(status.get("applied_strain", 0.0)) >= .1
    reached_20 = float(status.get("applied_strain", 0.0)) >= .2
    metrics_available = maximum_dual is not None and maximum_continuity is not None
    passed_so_far = bool(metrics_available and invariant
                         and maximum_dual < .05 and maximum_continuity < .05)
    complete = status.get("status") == "COMPLETED" and reached_20
    if complete and passed_so_far:
        classification = "PRODUCTION_MURA_NYE_LONG_HORIZON_QUALIFIED"
    elif not passed_so_far and metrics_available:
        classification = "MURA_NYE_LONG_HORIZON_SCIENTIFIC_FAILURE"
    elif status.get("status", "").startswith("PARTIAL") and passed_so_far:
        classification = "PARTIAL_VALID_RESTARTABLE_EVIDENCE"
    else:
        classification = "INSUFFICIENT_OR_FAILED_INFRASTRUCTURE_EVIDENCE"
    return {
        "case": case_dir.name,
        "grid": status.get("grid"),
        "condition": status.get("condition"),
        "seed": status.get("seed"),
        "source_sha": status.get("source_sha"),
        "status": status.get("status"),
        "step": status.get("step", 0),
        "physical_time_s": status.get("physical_time_s", 0.0),
        "applied_strain": status.get("applied_strain", 0.0),
        "reached_10_percent": reached_10,
        "reached_20_percent": reached_20,
        "checkpoint_count": len(checkpoints),
        "checkpoint_sha256": ({path.name: sha256(path) for path in checkpoints}
                              if checkpoints else {}),
        "history_record_count": len(history),
        "maximum_dual_nye_relative_rms": maximum_dual,
        "maximum_normalized_line_continuity_residual": maximum_continuity,
        "all_recorded_hard_invariants_passed": invariant,
        "valid_so_far": passed_so_far,
        "complete": complete,
        "classification": classification,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index-csv", type=Path, required=True)
    args = parser.parse_args()
    cases = [summarize(path) for path in sorted(args.cases.iterdir())
             if path.is_dir()]
    complete = [case for case in cases if case["complete"]]
    valid = [case for case in cases if case["valid_so_far"]]
    expected = {(64, name) for name in (
        "homogeneous", "broadband_noise", "mechanical_heterogeneity")}
    expected |= {(128, name) for name in (
        "homogeneous", "broadband_noise", "mechanical_heterogeneity")}
    present = {(case["grid"], case["condition"]) for case in cases}
    scientific_pass = bool(expected <= present and len(complete) == 6
                           and len(valid) == 6)
    if scientific_pass:
        classification = "TIER_B1_RESOLVED_ANCHOR_QUALIFIED"
    elif any(case["classification"] ==
             "MURA_NYE_LONG_HORIZON_SCIENTIFIC_FAILURE" for case in cases):
        classification = "TIER_B1_MURA_NYE_SCIENTIFIC_FAILURE"
    elif cases and all(case["valid_so_far"] for case in cases):
        classification = "TIER_B1_PARTIAL_VALID_RESTARTABLE_EVIDENCE"
    else:
        classification = "TIER_B1_INCOMPLETE_OR_INFRASTRUCTURE_FAILURE"
    result = {
        "schema": "asb-drx/v30-mura-tier-b1-decision/v1",
        "cases": cases,
        "expected_case_count": 6,
        "fixture_passed": bool(cases and all(case["checkpoint_count"] > 0
                                              for case in cases)),
        "scientific_gate_passed": scientific_pass,
        "long_run_complete": len(complete) == 6,
        "classification": classification,
        "claims_remaining_false": [
            "CURRENT_CAPACITY_CONE_QUALIFIED",
            "PERSISTENT_PHYSICAL_LAGB_PRECURSOR_OBSERVED",
            "FULL_MODEL_DRX_MECHANISM_QUALIFIED",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    fields = ("case", "grid", "condition", "seed", "status", "step",
              "physical_time_s", "applied_strain", "checkpoint_count",
              "maximum_dual_nye_relative_rms",
              "maximum_normalized_line_continuity_residual", "classification")
    with args.index_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            writer.writerow({name: case.get(name) for name in fields})
    print(json.dumps({"classification": classification,
                      "case_count": len(cases)}, sort_keys=True))


if __name__ == "__main__":
    main()
