#!/usr/bin/env python3
"""Quantify live V65 stress curvature without claiming an unobserved peak."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(paths: list[Path]) -> tuple[list[dict], list[dict]]:
    unique = {}
    sources = []
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines()
                if line.strip()]
        sources.append({"path": str(path.resolve()), "sha256": digest(path),
                        "row_count": len(rows)})
        for row in rows:
            unique[(float(row["physical_time_s"]),
                    int(row["adaptive_interval"]))] = row
    return sorted(unique.values(), key=lambda row: row["physical_time_s"]), sources


def fit_window(rows: list[dict], count: int) -> dict:
    if len(rows) < count:
        raise ValueError(f"stress-curvature fit requires {count} rows")
    selected = rows[-count:]
    time_us = np.asarray([row["physical_time_s"] for row in selected])*1e6
    stress_GPa = np.asarray([row["shear_stress_Pa"] for row in selected])/1e9
    relative_us = time_us-time_us[-1]
    linear = np.polyfit(relative_us, stress_GPa, 1)
    quadratic = np.polyfit(relative_us, stress_GPa, 2)
    fitted = np.polyval(quadratic, relative_us)
    terminal_slope = float(quadratic[1])
    curvature = float(2.0*quadratic[0])
    peak_delta = (None if curvature >= 0.0 else
                  float(-terminal_slope/curvature))
    return {
        "accepted_row_count": count,
        "start_time_s": float(time_us[0]*1e-6),
        "end_time_s": float(time_us[-1]*1e-6),
        "linear_slope_GPa_per_us": float(linear[0]),
        "terminal_quadratic_slope_GPa_per_us": terminal_slope,
        "quadratic_curvature_GPa_per_us2": curvature,
        "projected_peak_delta_us": peak_delta,
        "projected_peak_time_s": (
            None if peak_delta is None else float((time_us[-1]+peak_delta)*1e-6)),
        "quadratic_rms_residual_MPa": float(
            np.sqrt(np.mean((fitted-stress_GPa)**2))*1e3),
    }


def analyze(rows: list[dict], sources: list[dict], windows: tuple[int, ...],
            target_time_s: float) -> dict:
    fits = [fit_window(rows, count) for count in windows]
    projected = [fit["projected_peak_time_s"] for fit in fits
                 if fit["projected_peak_time_s"] is not None]
    slopes_positive = all(
        fit["terminal_quadratic_slope_GPa_per_us"] > 0.0 for fit in fits)
    curvatures_negative = all(
        fit["quadratic_curvature_GPa_per_us2"] < 0.0 for fit in fits)
    spread_s = max(projected)-min(projected) if projected else None
    consistent = bool(
        len(projected) == len(fits) and spread_s is not None
        and spread_s <= 5e-9)
    within_target = bool(projected and max(projected) <= target_time_s)
    classification = (
        "CONSISTENT_QUADRATIC_PEAK_PROJECTION_NOT_OBSERVED"
        if slopes_positive and curvatures_negative and consistent and within_target
        else "NO_ROBUST_IN_TARGET_PEAK_PROJECTION"
    )
    return {
        "schema": "asb-drx-v65-live-stress-curvature-v1",
        "sources": sources,
        "target_time_s": target_time_s,
        "fits": fits,
        "decision": {
            "classification": classification,
            "stress_peak_observed": False,
            "terminal_slopes_positive": slopes_positive,
            "curvatures_negative": curvatures_negative,
            "projection_consistent_across_windows": consistent,
            "projected_peak_spread_s": spread_s,
            "projected_peak_before_target": within_target,
            "strict_asb_claim": False,
            "semantics": (
                "least-squares extrapolation of accepted scalar stress history; "
                "not an observed peak, softening event, persistence result, or "
                "spatial localization qualification"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry", required=True, type=Path,
                        action="append")
    parser.add_argument("--window", type=int, action="append",
                        default=[])
    parser.add_argument("--target-time", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows, sources = load_rows(args.telemetry)
    windows = tuple(args.window or (64, 128, 256, 512))
    payload = analyze(rows, sources, windows, args.target_time)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name+".tmp")
    temporary.write_text(json.dumps(payload, indent=2)+"\n")
    os.replace(temporary, args.output)
    print(json.dumps(payload["decision"], indent=2))


if __name__ == "__main__":
    main()
