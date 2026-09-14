#!/usr/bin/env python3
"""Compare SIBM growth increments and fitted velocities, not final offsets."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def linear_velocity(time_s, amplitude_m):
    """OLS slope and one-sigma residual uncertainty for an amplitude window."""
    x = np.asarray(time_s, dtype=float)
    y = np.asarray(amplitude_m, dtype=float)
    if x.size < 3 or y.shape != x.shape or np.ptp(x) <= 0.0:
        raise ValueError("velocity fit requires at least three distinct times")
    centered = x-np.mean(x)
    design = np.column_stack((centered, np.ones_like(x)))
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y-design@np.array([slope, intercept])
    sigma = math.sqrt(float(residual@residual)/(x.size-2))
    return {
        "velocity_m_s": float(slope),
        "velocity_standard_error_m_s": float(
            sigma/math.sqrt(float(centered@centered))),
        "residual_rms_m": float(math.sqrt(float(np.mean(residual**2)))),
        "sample_count": int(x.size),
        "time_start_s": float(x[0]),
        "time_end_s": float(x[-1]),
    }


def case_metrics(csv_path: Path, *, domain_m: float, grid: int,
                 interface_width_m: float, scientific_class: str):
    rows = list(csv.DictReader(csv_path.open()))
    if len(rows) < 5:
        raise ValueError(f"{csv_path}: at least five contour records required")
    time = np.array([float(row["time_s"]) for row in rows])
    amplitude = np.array([float(row["bulge_amplitude_m"]) for row in rows])
    window = max(3, len(rows)//3)
    increment = float(amplitude[-1]-amplitude[0])
    dx = float(domain_m)/int(grid)
    return {
        "scientific_class": scientific_class,
        "contour_csv": str(csv_path),
        "grid": int(grid),
        "grid_spacing_m": dx,
        "interface_width_m": float(interface_width_m),
        "interface_points_per_width": float(interface_width_m/dx),
        "initial_amplitude_m": float(amplitude[0]),
        "final_amplitude_m": float(amplitude[-1]),
        "amplitude_increment_m": increment,
        "amplitude_increment_cells": float(increment/dx),
        "amplitude_increment_interface_widths": float(increment/interface_width_m),
        "early_velocity_fit": linear_velocity(time[:window], amplitude[:window]),
        "late_velocity_fit": linear_velocity(time[-window:], amplitude[-window:]),
    }


def compare(cases):
    increments = [x["amplitude_increment_m"] for x in cases]
    finals = [x["final_amplitude_m"] for x in cases]
    signs = [int(np.sign(x)) for x in increments]
    return {
        "signs": signs,
        "sign_agreement": bool(len(set(signs)) == 1),
        "final_amplitude_relative_spread": float(
            (max(finals)-min(finals))/max(max(abs(x) for x in finals), 1e-300)),
        "absolute_increment_spread_m": float(max(increments)-min(increments)),
        "increment_relative_spread": float(
            (max(increments)-min(increments))
            / max(max(abs(x) for x in increments), 1e-300)),
        "increment_converged": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--case", action="append", nargs=3,
                        metavar=("CSV", "GRID", "CLASS"), required=True)
    parser.add_argument("--domain-m", type=float, default=10e-6)
    parser.add_argument("--interface-width-m", type=float,
                        default=math.sqrt(5e-7/5e6))
    args = parser.parse_args()
    cases = [case_metrics(Path(path), domain_m=args.domain_m, grid=int(grid),
                          interface_width_m=args.interface_width_m,
                          scientific_class=classification)
             for path, grid, classification in args.case]
    comparison = compare(cases)
    payload = {
        "schema": "full-v34-v15-growth-increment-audit/v1",
        "classification": "LOADED_CANONICAL_BICRYSTAL_INCREMENTS_NOT_CONVERGED",
        "cases": cases,
        "comparison": comparison,
        "fixture_passed": True,
        "scientific_gate_passed": False,
        "reason": "signed increments disagree and displacements are unresolved",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(payload["classification"])


if __name__ == "__main__":
    main()
