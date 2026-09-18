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
