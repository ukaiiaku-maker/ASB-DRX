#!/usr/bin/env python3
"""Reduce accepted V65 telemetry to physical-time trend and ETA estimates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(paths: list[Path]) -> tuple[list[dict], list[dict]]:
    by_interval = {}
    sources = []
    for path in paths:
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text().splitlines()
                    if line.strip()]
        else:
            rows = json.loads(path.read_text())
        sources.append({"path": str(path.resolve()), "sha256": sha256(path),
                        "row_count": len(rows)})
        for row in rows:
            index = int(row["adaptive_interval"])
            if index in by_interval and by_interval[index] != row:
                raise ValueError(f"conflicting telemetry at interval {index}")
            by_interval[index] = row
    ordered = [by_interval[index] for index in sorted(by_interval)]
    if any(float(right["physical_time_s"]) <= float(left["physical_time_s"])
           for left, right in zip(ordered, ordered[1:])):
        raise ValueError("accepted telemetry time is not strictly increasing")
    return ordered, sources


def slope(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    centered = x-np.mean(x)
    denominator = float(np.dot(centered, centered))
    if denominator <= 0.0:
        return 0.0
    return float(np.dot(centered, y-np.mean(y))/denominator)


def summarize(rows: list[dict], sources: list[dict], *, window: int,
              target_time_s: float, grain_index: int) -> dict:
    if len(rows) < 2:
        raise ValueError("trend reduction requires at least two accepted states")
    selected = rows[-min(len(rows), max(int(window), 2)):]
    time_s = np.asarray([row["physical_time_s"] for row in selected])
    strain = np.asarray([row["applied_shear_strain"] for row in selected])
    stress = np.asarray([row["shear_stress_Pa"] for row in selected])
    temperature = np.asarray([row["temperature_mean_K"] for row in selected])
    contrast = np.asarray([row["temperature_contrast_K"] for row in selected])
    grain_volume = np.asarray([
        row["grain_volume_m3"][grain_index] for row in selected])
    duration = np.asarray([row["accepted_duration_s"] for row in selected])
    wall = np.asarray([row["wall_time_s"] for row in selected])
    ratios = np.asarray([row["maximum_tolerance_ratio"] for row in selected])
    remaining = max(float(target_time_s)-float(time_s[-1]), 0.0)
    seconds_per_physical_second = float(np.median(wall/duration))
    limiting = {}
    for row in selected:
        attempts = row["accuracy_decision"]["attempts"]
        errors = attempts[-1].get("metric_errors", {})
        if not errors:
            continue
        name = max(errors, key=lambda key: (
            errors[key]["normalized_error"]
            /errors[key]["relative_tolerance"]))
        limiting[name] = limiting.get(name, 0)+1
    return {
        "schema": "asb-drx-v65-live-trend-summary-v1",
        "sources": sources,
        "window": {
            "row_count": len(selected),
            "first_interval": int(selected[0]["adaptive_interval"]),
            "last_interval": int(selected[-1]["adaptive_interval"]),
            "first_time_s": float(time_s[0]),
            "last_time_s": float(time_s[-1]),
            "duration_s": float(time_s[-1]-time_s[0]),
        },
        "current": {
            "physical_time_s": float(time_s[-1]),
            "applied_shear_strain": float(strain[-1]),
            "shear_stress_Pa": float(stress[-1]),
            "temperature_mean_K": float(temperature[-1]),
            "temperature_contrast_K": float(contrast[-1]),
            "target_grain_volume_m3": float(grain_volume[-1]),
        },
        "slopes": {
            "stress_Pa_per_s": slope(time_s, stress),
            "stress_Pa_per_shear": slope(strain, stress),
            "temperature_mean_K_per_s": slope(time_s, temperature),
            "temperature_contrast_K_per_s": slope(time_s, contrast),
            "target_grain_volume_m3_per_s": slope(time_s, grain_volume),
        },
        "regime": {
            "stress_hardening": bool(slope(time_s, stress) > 0.0),
            "mean_heating": bool(slope(time_s, temperature) > 0.0),
            "contrast_increasing": bool(slope(time_s, contrast) > 0.0),
            "target_grain_receding": bool(slope(time_s, grain_volume) < 0.0),
        },
        "accuracy": {
            "maximum_tolerance_ratio": float(np.max(ratios)),
            "median_tolerance_ratio": float(np.median(ratios)),
            "limiting_metric_counts": limiting,
        },
        "throughput": {
            "median_accepted_duration_s": float(np.median(duration)),
            "median_interval_wall_time_s": float(np.median(wall)),
            "median_wall_seconds_per_physical_ns": (
                seconds_per_physical_second*1e-9),
            "target_time_s": float(target_time_s),
            "remaining_physical_time_s": remaining,
            "estimated_remaining_wall_time_s": (
                remaining*seconds_per_physical_second),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry", type=Path, action="append", required=True)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--target-time", type=float, required=True)
    parser.add_argument("--grain-index", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, sources = load_rows(args.telemetry)
    payload = summarize(
        rows, sources, window=args.window, target_time_s=args.target_time,
        grain_index=args.grain_index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps({
        "output": str(args.output), "sha256": sha256(args.output),
        "last_interval": payload["window"]["last_interval"],
        "estimated_remaining_wall_time_s": payload[
            "throughput"]["estimated_remaining_wall_time_s"],
        "regime": payload["regime"],
    }))


if __name__ == "__main__":
    main()
