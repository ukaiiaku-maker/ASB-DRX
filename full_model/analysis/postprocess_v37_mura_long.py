#!/usr/bin/env python3
"""Postprocess the selected long V37 one-grain Mura continuation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from full_model.analysis.postprocess_v37_mura_organization import summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    case = summarize(args.case_dir)
    history = [json.loads(line) for line in (
        args.case_dir/"history.jsonl").read_text().splitlines() if line.strip()]
    orientation_compatible = case["orientation_span_deg"] >= 1.0
    persistent = False
    if len(history) >= 3:
        tail = history[-3:]
        persistent = all(row["orientation_span_deg"] >= 1.0
                         and row["ordered_fraction_wall_local"] > 0.0
                         for row in tail)
    if not case["hard_invariants_passed"]:
        classification = "HARD_INVALID_LONG_ORGANIZATION"
    elif persistent:
        classification = "PERSISTENT_ORIENTATION_COMPATIBLE_LAGB_PRECURSOR"
    elif case["ordered_line_m2_cells"] > 0.0:
        classification = "ORDERED_LINE_WITHOUT_PERSISTENT_ORIENTATION_WALL"
    else:
        classification = "NO_CAPTURED_WALL_LINE_AT_EXPOSURE"
    result = {
        "schema": "asb-drx/v37/mura-long-organization/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "case": case, "classification": classification,
        "orientation_compatible": orientation_compatible,
        "persistent_last_three_records": persistent,
        "lagb_precursor_claimed": bool(persistent),
        "phase_support_present": False, "drx_claimed": False,
        "claim_boundary": "one-grain signed-dislocation organization without phase or grain allocation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    strain = [row["applied_strain"] for row in history]
    axes[0].plot(strain, [row["ordered_fraction_wall_local"] for row in history])
    axes[1].plot(strain, [row["ordered_polarization_wall_local_mean"]
                          for row in history])
    axes[2].plot(strain, [row["orientation_span_deg"] for row in history])
    axes[0].set_ylabel("wall-local ordered fraction")
    axes[1].set_ylabel("wall-local polarization")
    axes[2].set_ylabel("orientation span (deg)")
    for axis in axes:
        axis.set_xlabel("applied strain"); axis.grid(alpha=.25)
    figure.tight_layout(); figure.savefig(args.figure, dpi=180); plt.close(figure)


if __name__ == "__main__":
    main()
