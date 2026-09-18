#!/usr/bin/env python3
"""Decision-grade timestep/grid summary for V39 common trajectories."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(result, record_limit=None):
    records = result["records"] if record_limit is None else result["records"][:record_limit]
    last = records[-1]
    state = last["state_metrics"]
    return {
        "physical_time_s": float(last["physical_time_end_s"]),
        "contour_displacement_m": float(sum(
            row["front_contour_displacement_m"] for row in records)),
        "signed_sweep_m3": float(sum(row["front_signed_sweep_m3"] for row in records)),
        "absolute_sweep_m3": float(sum(row["front_absolute_sweep_m3"] for row in records)),
        "complete_energy_delta_J": float(sum(
            row["front_complete_energy_delta_J"] for row in records)),
        "plastic_work_increment_J_m3_cells": float(sum(
            row["mura_plastic_work_increment_J_m3_cells"] for row in records)),
        "deposited_heat_increment_J_m3_cells": float(sum(
            row["mura_deposited_heat_increment_J_m3_cells"] for row in records)),
        **{key: float(value) for key, value in state.items()},
    }


def differences(actual, reference):
    floors = {
        "contour_displacement_m": 1e-13, "signed_sweep_m3": 1e-28,
        "absolute_sweep_m3": 1e-28, "complete_energy_delta_J": 1e-22,
        "ordered_line_m_per_m_thickness": 1e-12,
        "family_nye_rms_m1": 1.0, "orientation_range_rad": 1e-10,
        "temperature_minimum_K": 1e-3, "temperature_maximum_K": 1e-3,
    }
    result = {}
    for key in reference:
        if key == "physical_time_s":
            continue
        absolute = abs(actual[key]-reference[key])
        denominator = max(abs(reference[key]), floors.get(key, 1e-14))
        result[key] = {"absolute": absolute, "relative": absolute/denominator,
                       "normalization_floor": floors.get(key, 1e-14)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    cases = {}
    for path in sorted(args.root.glob("*/result.json")):
        value = json.loads(path.read_text())
        cases[path.parent.name] = {
            "path": str(path.resolve()), "sha256": sha256(path),
            "source_sha": value["source_sha"], "grid": value["grid"],
            "macro_dt_s": value["macro_dt_s"],
            "intervals": value["completed_intervals"],
            "wall_seconds": value["wall_seconds"],
            "metrics": metrics(value),
        }
    timestep_names = sorted(
        [name for name, item in cases.items()
         if item["grid"] == 16 and np.isclose(item["metrics"]["physical_time_s"], 1e-3)],
        key=lambda name: cases[name]["macro_dt_s"], reverse=True)
    reference_name = min(timestep_names, key=lambda name: cases[name]["macro_dt_s"])
    reference = cases[reference_name]["metrics"]
    timestep = []
    for name in timestep_names:
        comparison = differences(cases[name]["metrics"], reference)
        timestep.append({"case": name, **cases[name], "difference_to_reference": comparison,
                         "provisional_five_percent_passed": bool(max(
                             comparison[key]["relative"] for key in (
                                 "contour_displacement_m", "total_line_m_per_m_thickness",
                                 "beta_p_rms", "orientation_range_rad",
                                 "temperature_maximum_K")) <= .05)})
    grid_cases = {}
    for name in ("n16_H6p25e-5", "n64_H6p25e-5_2", "n128_H6p25e-5_1"):
        path = Path(cases[name]["path"])
        value = json.loads(path.read_text())
        grid_cases[name] = {**cases[name], "metrics": metrics(value, record_limit=1)}
    grid_reference = grid_cases["n128_H6p25e-5_1"]["metrics"]
    grid = []
    for name in ("n16_H6p25e-5", "n64_H6p25e-5_2", "n128_H6p25e-5_1"):
        comparison = differences(grid_cases[name]["metrics"], grid_reference)
        grid.append({"case": name, **grid_cases[name],
                     "difference_to_n128": comparison,
                     "provisional_five_percent_passed": bool(max(
                         comparison[key]["relative"] for key in (
                             "contour_displacement_m", "total_line_m_per_m_thickness",
                             "ordered_line_m_per_m_thickness", "beta_p_rms",
                             "temperature_maximum_K")) <= .05)})
    result = {
        "schema": "asb-drx/v39/common-refinement/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(args.root.resolve()), "cases": cases,
        "timestep_reference": reference_name, "timestep_refinement": timestep,
        "grid_reference": "n128_H6p25e-5_1", "grid_refinement": grid,
        "selected_macro_dt_s": 6.25e-5,
        "timestep_gate_passed": timestep[[r["case"] for r in timestep].index(
            "n16_H6p25e-5")]["provisional_five_percent_passed"],
        "grid_gate_passed": grid[1]["provisional_five_percent_passed"],
        "interpretation": (
            "n16 temporal refinement qualifies 62.5 us against 31.25 us; "
            "n64/n128 spatial overlap remains unqualified, dominated by the "
            "near-extinction ordered reservoir and front displacement."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    dt = np.asarray([row["macro_dt_s"] for row in timestep])
    disp = np.asarray([row["metrics"]["contour_displacement_m"] for row in timestep])
    axes[0].plot(dt*1e6, disp*1e9, "o-")
    axes[0].set(xlabel="common macro interval (microseconds)",
                ylabel="1 ms contour displacement (nm)", xscale="log")
    grids = np.asarray([row["grid"] for row in grid])
    ordered = np.asarray([row["metrics"]["ordered_line_m_per_m_thickness"]
                          for row in grid])
    axes[1].plot(grids, ordered, "o-")
    axes[1].set(xlabel="grid points per axis", ylabel="ordered line (m/m)",
                yscale="log")
    fig.tight_layout(); args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=180); plt.close(fig)


if __name__ == "__main__":
    main()
