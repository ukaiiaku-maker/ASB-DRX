#!/usr/bin/env python3
"""Compare the repaired 5.01% topology continuation with its exact-off control."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from full_model.analysis.run_v30_mura_tier_b1_case import create_case
from full_model.analysis.run_v42_topology_replay import state_record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument("--repaired-checkpoint", type=Path, required=True)
    parser.add_argument("--control-status", type=Path, required=True)
    parser.add_argument("--repaired-status", type=Path, required=True)
    parser.add_argument("--repair-decision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    context = create_case(64, "mechanical_heterogeneity", 42, 1e-5)
    control_state, control = state_record(args.control_checkpoint, context)
    repaired_state, repaired = state_record(args.repaired_checkpoint, context)
    control_status = json.loads(args.control_status.read_text())
    repaired_status = json.loads(args.repaired_status.read_text())
    cmet = control_status["latest_metrics"]; rmet = repaired_status["latest_metrics"]
    nonthermal_max = 0.0
    for group in ("common", "density", "reservoir_alignment"):
        left = getattr(control_state, group); right = getattr(repaired_state, group)
        for name in left.__dict__:
            if group == "common" and name == "temperature_K":
                continue
            nonthermal_max = max(nonthermal_max, float(np.max(np.abs(
                np.asarray(getattr(left, name))-np.asarray(getattr(right, name))))))
    temperature = (np.asarray(repaired_state.common.temperature_K)
                   -np.asarray(control_state.common.temperature_K))
    hard = bool(
        rmet["accepted_step_hard_invariant_passed"]
        and rmet["authoritative_source_offset_relative_rms"] <= .05
        and rmet["normalized_line_continuity_residual"] <= .05)
    payload = {
        "schema": "asb-drx/v42/topology-matched-continuation/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "control": control, "repaired": repaired,
        "control_status": control_status, "repaired_status": repaired_status,
        "matched_elapsed_time_and_strain": bool(
            control["strain"] == repaired["strain"]
            and control_status["physical_time_s"] == repaired_status["physical_time_s"]),
        "hard_invariants_passed": hard,
        "metrics": {
            "control": cmet, "repaired": rmet,
            "nonthermal_maximum_absolute_field_difference": nonthermal_max,
            "temperature_difference_K": {
                "minimum": float(np.min(temperature)),
                "maximum": float(np.max(temperature)),
                "mean": float(np.mean(temperature)),
            },
        },
        "classification": (
            "VALID_NEGATIVE_AT_TESTED_CONDITION;NO_QUALIFIED_BOUNDARY"
            if hard else "HARD_INVALID"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    decision = json.loads(args.repair_decision.read_text())
    decision["matched_5p01_continuation"] = {
        "artifact": str(args.output),
        "classification": payload["classification"],
        "hard_invariants_passed": hard,
        "matched_elapsed_time_and_strain": payload[
            "matched_elapsed_time_and_strain"],
    }
    args.repair_decision.write_text(json.dumps(
        decision, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
