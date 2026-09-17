#!/usr/bin/env python3
"""Matched-time overlap test for affordable front-only hold intervals."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np

from full_model.analysis.run_v36_recurrent_physical_response import run_response


def metrics(result):
    records = result["records"]
    return {
        "cumulative_contour_displacement_m": float(sum(
            row["accepted_contour_displacement_m"] for row in records)),
        "cumulative_signed_sweep_m3": float(result["cumulative_signed_sweep_m3"]),
        "cumulative_processed_line_m": float(sum(
            row["processed_line_change_m"] for row in records)),
        "cumulative_boundary_line_m": float(sum(
            row["boundary_inventory_change_m"] for row in records)),
        "cumulative_sink_line_m": float(sum(
            row["material_sink_change_m"] for row in records)),
        "cumulative_complete_energy_delta_J": float(sum(
            row["complete_energy_delta_J"] for row in records)),
        "accepted_intervals": int(result["accepted_front_intervals"]),
        "wall_seconds": float(result["wall_seconds_this_invocation"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--horizon-s", type=float, default=1e-3)
    parser.add_argument("--dt-s", type=float, nargs="+",
                        default=(5e-6, 2.5e-5, 1e-4, 2.5e-4, 1e-3))
    args = parser.parse_args()
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    rows = []
    for dt_s in args.dt_s:
        intervals = int(round(args.horizon_s/dt_s))
        if not np.isclose(intervals*dt_s, args.horizon_s):
            raise ValueError("every interval must divide the matched horizon")
        case = f"dt_{dt_s:.3e}".replace("+", "")
        result = run_response(
            output_dir=args.output_root/case, protocol="hold",
            grid=args.grid, intervals=intervals, dt_s=dt_s,
            proposal_fraction=.0625, checkpoint_every=max(intervals, 1),
            mura_enabled=False, front_enabled=True,
            front_exp_n=1.0, front_symmetric_availability=1.0)
        rows.append({"dt_s": dt_s, "intervals": intervals,
                     "metrics": metrics(result),
                     "result_path": str((args.output_root/case/"result.json").resolve())})
    reference = rows[0]["metrics"]
    scales = {name: max(abs(float(value)), 1e-300)
              for name, value in reference.items()
              if name not in ("accepted_intervals", "wall_seconds")}
    for row in rows:
        errors = {name: abs(float(row["metrics"][name])-float(reference[name]))/scales[name]
                  for name in scales}
        row["relative_errors_from_small_step"] = errors
        row["maximum_physical_relative_error"] = max(errors.values())
        row["overlap_passed_5pct"] = bool(
            row["maximum_physical_relative_error"] <= .05)
    passing = [row for row in rows if row["overlap_passed_5pct"]]
    result = {
        "schema": "asb-drx/v38/front-macro-overlap/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": source, "grid": args.grid,
        "matched_horizon_s": args.horizon_s,
        "protocol": "predeformed_hold_front_only_zero_applied_work",
        "front_hypothesis": "shape_n_low_pressure_sensitivity",
        "rows": rows,
        "largest_passing_dt_s": max((row["dt_s"] for row in passing), default=None),
        "common_clock_claimed": False,
        "purpose": (
            "select an affordable front macro interval; coupled Mura/thermal "
            "subcycling requires an independent common-clock qualification"),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
