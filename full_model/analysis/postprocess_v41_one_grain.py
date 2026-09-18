#!/usr/bin/env python3
"""Promote a verified V40 compact continuation into the V41 decision schema."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--checksum", type=Path, required=True)
    parser.add_argument("--attempts-log", type=Path, required=True)
    parser.add_argument("--diagnostic-decision", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    expected = args.checksum.read_text().split()[0]
    actual = sha256(args.archive)
    if actual != expected:
        raise ValueError(f"archive checksum mismatch: {actual} != {expected}")
    source = json.loads(args.decision.read_text())
    cases = source["cases"]
    promoted = max(cases, key=lambda row: row["summary"]["final_strain"])
    frank = promoted["independent_frank_bilby"]
    candidate = bool(frank["candidate_wall_present"])
    relative_closure_applicable = bool(
        candidate
        and abs(frank["orientation_closure_vector"][0])
        + abs(frank["orientation_closure_vector"][1])
        + abs(frank["orientation_closure_vector"][2]) > 0.0)
    lagb = bool(source["scientific_gate_passed"] and relative_closure_applicable)
    diagnostic = None
    if args.diagnostic_decision is not None:
        raw_diagnostic = json.loads(args.diagnostic_decision.read_text())
        compact_cases = []
        for row in raw_diagnostic["cases"]:
            compact_cases.append({
                "condition": row["summary"]["condition"],
                "topology_route_enabled": row["summary"][
                    "topology_route_enabled"],
                "final_strain": row["summary"]["final_strain"],
                "hard_invariants_passed": row["summary"][
                    "hard_invariants_passed"],
                "orientation_span_deg": row["orientation"]["span_deg"],
                "nye_rms_m1": row["nye"]["rms_m1"],
                "ordered_line_m2_cells": row["summary"][
                    "ordered_line_m2_cells"],
                "candidate_wall_present": row["independent_frank_bilby"][
                    "candidate_wall_present"],
                "persistent_last_three_records": row[
                    "persistent_last_three_records"],
            })
        diagnostic = {
            "path": str(args.diagnostic_decision.resolve()),
            "sha256": sha256(args.diagnostic_decision),
            "exact_disabling_controls": True,
            "cases": compact_cases,
            "scientific_gate_passed": raw_diagnostic[
                "scientific_gate_passed"],
            "classification": (
                "COMPACT_CONTROL_HAS_LAGB_CANDIDATE_REQUIRING_RELEASE"
                if raw_diagnostic["scientific_gate_passed"] else
                "COMPACT_TOPOLOGY_AND_FORCING_CONTROLS_NO_QUALIFIED_LAGB"),
        }

    result = {
        "schema": "asb-drx/v41/one-grain-decision/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verified_archive": {
            "path": str(args.archive.resolve()),
            "sha256": actual,
            "expected_sha256": expected,
            "checksum_verified": True,
        },
        "postprocessed_decision": {
            "path": str(args.decision.resolve()),
            "sha256": sha256(args.decision),
        },
        "attempt_history": {
            "path": str(args.attempts_log.resolve()),
            "sha256": sha256(args.attempts_log),
            "records": args.attempts_log.read_text().splitlines(),
        },
        "solver_completed": promoted["summary"]["status"] == "COMPLETED",
        "hard_invariants_passed": bool(source["fixture_passed"]),
        "final_strain": promoted["summary"]["final_strain"],
        "accepted_intervals": promoted["summary"]["accepted_intervals"],
        "orientation_span_deg": promoted["orientation"]["span_deg"],
        "nye_rms_m1": promoted["nye"]["rms_m1"],
        "ordered_line_m2_cells": promoted["summary"]["ordered_line_m2_cells"],
        "candidate_wall_present": candidate,
        "persistent_last_three_records": promoted[
            "persistent_last_three_records"],
        "frank_bilby": frank,
        "frank_bilby_relative_closure_applicable": relative_closure_applicable,
        "frank_bilby_classification": (
            "RELATIVE_CLOSURE_EVALUATED" if relative_closure_applicable else
            "NO_QUALIFIED_BOUNDARY_FOR_RELATIVE_CLOSURE"),
        "scientific_gate_passed": lagb,
        "post_endpoint_diagnostic": diagnostic,
        "phase_or_grain_allocation_present": bool(
            source["phase_or_grain_allocation_present"]),
        "drx_claimed": False,
        "classification": (
            "PERSISTENT_LAGB_CANDIDATE_REQUIRES_RELEASE" if lagb else
            "NO_CURRENT_SOURCE_PHYSICAL_LAGB_PRECURSOR_AT_RETAINED_ENDPOINT"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
