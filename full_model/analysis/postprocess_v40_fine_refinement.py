#!/usr/bin/env python3
"""Evaluate V40 common-state temporal and spatial refinement in physical units."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import hashlib
import json
import math
from pathlib import Path


METRICS = (
    "cumulative_contour_displacement_m",
    "family_nye_rms_m1",
    "ordered_line_m_per_m_thickness",
    "orientation_range_rad",
    "beta_p_rms",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(a: float, b: float) -> float:
    return abs(a-b)/max(abs(b), 1e-30)


def load(path: Path) -> dict:
    payload = json.loads(path.read_text())
    final = payload["records"][-1]["state_metrics"]
    count = int(payload["grid"])**2
    row = {
        "path": str(path.resolve()), "sha256": digest(path),
        "source_sha": payload["source_sha"], "grid": int(payload["grid"]),
        "macro_dt_s": float(payload["macro_dt_s"]),
        "intervals": int(payload["completed_intervals"]),
        "physical_time_s": float(payload["physical_time_s"]),
        "wall_seconds": float(payload["wall_seconds"]),
        "cumulative_contour_displacement_m": float(
            payload["cumulative_contour_displacement_m"]),
    }
    for key in METRICS[1:]:
        row[key] = float(final[key])
    for stem in ("plastic_work", "deposited_heat", "stored_line_energy"):
        raw = sum(float(record[f"mura_{stem}_increment_J_m3_cells"])
                  for record in payload["records"])
        row[f"cumulative_{stem}_raw_J_m3_cells"] = raw
        row[f"cumulative_{stem}_mean_J_m3"] = raw/count
    return row


def compare(coarse: dict, fine: dict) -> dict:
    return {
        "coarse_grid": coarse["grid"], "fine_grid": fine["grid"],
        "coarse_macro_dt_s": coarse["macro_dt_s"],
        "fine_macro_dt_s": fine["macro_dt_s"],
        "relative_differences": {
            key: relative(coarse[key], fine[key]) for key in METRICS},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [Path(value) for value in glob.glob(str(args.root/"*"/"result.json"))]
    rows = sorted((load(path) for path in paths),
                  key=lambda x: (x["grid"], -x["macro_dt_s"]))
    by_grid = {}
    for row in rows:
        by_grid.setdefault(row["grid"], []).append(row)
    temporal = {}
    temporal_pass = {}
    for grid, group in by_grid.items():
        pairs = [compare(group[index], group[index+1])
                 for index in range(len(group)-1)]
        temporal[str(grid)] = pairs
        latest = pairs[-1] if pairs else None
        temporal_pass[str(grid)] = bool(latest and all(
            latest["relative_differences"][key] <= .05 for key in (
                "cumulative_contour_displacement_m", "family_nye_rms_m1",
                "orientation_range_rad", "beta_p_rms")))
    matched = []
    for left in by_grid.get(128, []):
        for right in by_grid.get(192, []):
            if math.isclose(left["macro_dt_s"], right["macro_dt_s"],
                            rel_tol=0.0, abs_tol=1e-15):
                matched.append(compare(left, right))
    selected = matched[-1] if matched else None
    spatial_pass = bool(selected and all(
        selected["relative_differences"][key] <= .05 for key in METRICS))
    result = {
        "schema": "asb-drx/v40/fine-common-refinement/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "threshold_relative": .05,
        "physical_normalization": {
            "cell_sum_to_mean_energy_density": "divide by grid**2",
            "line_measure": "sum(rho_m^-2)*dx*dy; m per represented m",
            "front_measure": "signed swept volume/interface area; m",
        },
        "cases": rows, "temporal_comparisons": temporal,
        "temporal_passed": temporal_pass,
        "matched_spatial_comparisons": matched,
        "selected_spatial_comparison": selected,
        "scientific_spatial_refinement_passed": spatial_pass,
        "classification": (
            "QUALIFIED_FINE_PAIR" if spatial_pass else
            "FRONT_QUALIFIED_GRADIENT_STATE_SPATIALLY_UNRESOLVED"),
        "long_horizon_permission": (
            "QUALIFIED" if spatial_pass else "EXPLORATORY_COARSE_ONLY"),
        "claim_boundary": (
            "The 5% rule is applied independently to physical observables; "
            "solver invariants do not substitute for refinement."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
