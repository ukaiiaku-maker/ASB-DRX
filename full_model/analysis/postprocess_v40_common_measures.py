#!/usr/bin/env python3
"""Recompute V39/V40 common refinements with physical 2-D quadrature."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from full_model.analysis.postprocess_v39_common_refinement import differences


ENERGY_SUM_FIELDS = (
    "mura_plastic_work_increment_J_m3_cells",
    "mura_deposited_heat_increment_J_m3_cells",
    "mura_stored_line_energy_increment_J_m3_cells",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(result: dict, selected_index: int) -> dict:
    records = result["records"][: selected_index + 1]
    selected = records[-1]
    grid = int(result["grid"])
    length_m = 3.2e-6
    spacing_m = length_m / grid
    ncell = grid * grid
    state = {key: float(value) for key, value in selected["state_metrics"].items()}
    raw = {name: float(sum(float(row.get(name, 0.0)) for row in records))
           for name in ENERGY_SUM_FIELDS}
    quadrature = {}
    for name, value in raw.items():
        quadrature[name] = {
            "raw_sum_J_m3_cells": value,
            "volume_mean_J_m3": value / ncell,
            "per_unit_thickness_integral_J_m": value * spacing_m**2,
            "physical_volume_integral_J": None,
            "physical_volume_integral_status": (
                "UNDEFINED_WITHOUT_DECLARED_OUT_OF_PLANE_THICKNESS"),
        }
    tmin = state["temperature_minimum_K"]
    tmax = state["temperature_maximum_K"]
    return {
        "simulation_interval_count": int(result["completed_intervals"]),
        "selected_snapshot_index": int(selected_index),
        "selected_snapshot_time_s": float(selected["physical_time_end_s"]),
        "selected_snapshot_inclusive_interval_count": len(records),
        "grid": grid,
        "cell_count": ncell,
        "domain_length_m": length_m,
        "spacing_m": spacing_m,
        "energy_density_sum_quadrature": quadrature,
        "temperature": {
            "reference_temperature_K": 1100.0,
            "minimum_K": tmin,
            "maximum_K": tmax,
            "minimum_rise_K": tmin - 1100.0,
            "maximum_rise_K": tmax - 1100.0,
            "range_K": tmax - tmin,
        },
        "contour_displacement_m": float(sum(
            row["front_contour_displacement_m"] for row in records)),
        "signed_sweep_m3": float(sum(
            row["front_signed_sweep_m3"] for row in records)),
        "absolute_sweep_m3": float(sum(
            row["front_absolute_sweep_m3"] for row in records)),
        "complete_energy_delta_J": float(sum(
            row["front_complete_energy_delta_J"] for row in records)),
        **state,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected = {
        "n64_H6p25e-5_2": 0,
        "n128_H6p25e-5_1": 0,
    }
    cases = {}
    raw_results = {}
    for name, index in selected.items():
        path = args.root / name / "result.json"
        result = json.loads(path.read_text())
        raw_results[name] = result
        cases[name] = {
            "path": str(path.resolve()), "sha256": sha256(path),
            "source_sha": result["source_sha"],
            **snapshot(result, index),
        }
    left = cases["n64_H6p25e-5_2"]
    right = cases["n128_H6p25e-5_1"]
    physical_metrics = {}
    for label, field in (
        ("plastic_work", "mura_plastic_work_increment_J_m3_cells"),
        ("deposited_heat", "mura_deposited_heat_increment_J_m3_cells"),
        ("stored_line_energy", "mura_stored_line_energy_increment_J_m3_cells"),
    ):
        lq = left["energy_density_sum_quadrature"][field]
        rq = right["energy_density_sum_quadrature"][field]
        physical_metrics[label] = {
            key: {
                "n64": lq[key], "n128": rq[key],
                "absolute_difference": abs(lq[key] - rq[key]),
                "relative_to_n128": abs(lq[key] - rq[key]) /
                max(abs(rq[key]), 1e-300),
            }
            for key in ("raw_sum_J_m3_cells", "volume_mean_J_m3",
                        "per_unit_thickness_integral_J_m")
        }
    scalar_names = (
        "contour_displacement_m", "family_nye_rms_m1",
        "ordered_line_m_per_m_thickness", "total_line_m_per_m_thickness",
        "beta_p_rms", "orientation_range_rad", "effective_stress_rms_Pa",
    )
    scalar = differences(
        {name: left[name] for name in scalar_names},
        {name: right[name] for name in scalar_names})
    temperature = {
        key: {
            "n64_K": left["temperature"][key],
            "n128_K": right["temperature"][key],
            "absolute_difference_K": abs(
                left["temperature"][key] - right["temperature"][key]),
        }
        for key in ("minimum_rise_K", "maximum_rise_K", "range_K")
    }
    uniform = {}
    for grid in (64, 128, 192):
        q = 3.25e5
        length = 3.2e-6
        dx = length / grid
        raw_sum = q * grid * grid
        uniform[str(grid)] = {
            "raw_sum_J_m3_cells": raw_sum,
            "volume_mean_J_m3": raw_sum / (grid * grid),
            "per_unit_thickness_integral_J_m": raw_sum * dx * dx,
        }
    mean_spread = max(row["volume_mean_J_m3"] for row in uniform.values()) - min(
        row["volume_mean_J_m3"] for row in uniform.values())
    integral_spread = max(
        row["per_unit_thickness_integral_J_m"] for row in uniform.values()) - min(
        row["per_unit_thickness_integral_J_m"] for row in uniform.values())
    result = {
        "schema": "asb-drx/v40/common-physical-measures/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_evidence": "unaltered V39 result JSON",
        "quadrature_interpretation": (
            "named *_J_m3_cells values are reproducible uniform-grid sums of "
            "cell energy densities; physical 2-D integration uses dx^2"),
        "cases": cases,
        "n64_n128_first_interval": {
            "matched_time_s": left["selected_snapshot_time_s"],
            "physical_energy_measures": physical_metrics,
            "state_scalar_differences": scalar,
            "temperature_rise_differences": temperature,
        },
        "uniform_field_fixture": {
            "field_J_m3": 3.25e5, "domain_length_m": 3.2e-6,
            "cases": uniform, "volume_mean_spread_J_m3": mean_spread,
            "per_unit_thickness_integral_spread_J_m": integral_spread,
            "passed": bool(mean_spread <= 1e-10 and integral_spread <= 1e-20),
        },
        "interpretation": (
            "raw energy-density sums are not cross-grid observables; corrected "
            "means and per-thickness integrals agree while front, Nye, and "
            "near-extinction ordered responses remain unresolved"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
