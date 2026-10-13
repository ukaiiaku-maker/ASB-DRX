#!/usr/bin/env python3
"""Prospectively validate the preserved V65 scalar stress-peak forecast."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(payload, indent=2)+"\n")
    os.replace(temporary, path)


def load_rows(paths: list[Path]) -> list[dict]:
    unique = {}
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            unique[(float(row["physical_time_s"]),
                    int(row["adaptive_interval"]))] = row
    return sorted(unique.values(), key=lambda row: row["physical_time_s"])


def evaluate(rows: list[dict], forecast: dict, *, confirmation_lag_s: float,
             forecast_tolerance_s: float) -> dict:
    if not rows:
        raise ValueError("peak forecast validation requires accepted telemetry")
    projected = [float(value) for value in forecast[
        "preterminal_stress_peak_projection"]["projected_peak_times_s"]]
    lower = min(projected)-forecast_tolerance_s
    upper = max(projected)+forecast_tolerance_s
    peak_index = max(range(len(rows)), key=lambda index: abs(float(
        rows[index]["shear_stress_Pa"])))
    peak = rows[peak_index]
    peak_time = float(peak["physical_time_s"])
    peak_stress = abs(float(peak["shear_stress_Pa"]))
    latest = rows[-1]
    latest_time = float(latest["physical_time_s"])
    latest_stress = abs(float(latest["shear_stress_Pa"]))
    confirmation_elapsed = latest_time-peak_time
    post_peak_rows = rows[peak_index+1:]
    post_peak_lower = bool(
        post_peak_rows and max(abs(float(row["shear_stress_Pa"]))
                               for row in post_peak_rows) < peak_stress)
    confirmed = bool(
        confirmation_elapsed >= confirmation_lag_s and post_peak_lower)
    inside_band = lower <= peak_time <= upper
    if confirmed:
        classification = (
            "FORECAST_VALIDATED_LOCAL_SCALAR_PEAK"
            if inside_band else "FORECAST_MISSED_BUT_LOCAL_SCALAR_PEAK_OBSERVED"
        )
    else:
        classification = "PEAK_FORECAST_VALIDATION_PENDING"
    midpoint = .5*(min(projected)+max(projected))
    return {
        "schema": "asb-drx-v65-peak-forecast-validation-v1",
        "classification": classification,
        "local_scalar_peak_confirmed": confirmed,
        "strict_asb_claim": False,
        "forecast_minimum_time_s": min(projected),
        "forecast_maximum_time_s": max(projected),
        "predeclared_forecast_tolerance_s": forecast_tolerance_s,
        "confirmation_lag_s": confirmation_lag_s,
        "observed_peak_time_s": peak_time,
        "observed_peak_interval": int(peak["adaptive_interval"]),
        "observed_peak_stress_Pa": peak_stress,
        "observed_peak_inside_validation_band": inside_band,
        "observed_minus_forecast_midpoint_s": peak_time-midpoint,
        "latest_time_s": latest_time,
        "latest_interval": int(latest["adaptive_interval"]),
        "confirmation_elapsed_s": confirmation_elapsed,
        "current_softening_fraction": max(
            0.0, (peak_stress-latest_stress)/max(peak_stress, 1e-300)),
        "semantics": (
            "accepted scalar stress history and a predeclared post-peak lag; "
            "confirmation is neither spatial localization nor strict ASB"),
    }


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry", required=True, action="append", type=Path)
    parser.add_argument("--forecast", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--solver-pid", required=True, type=int)
    parser.add_argument("--confirmation-lag", type=float, default=5e-9)
    parser.add_argument("--forecast-tolerance", type=float, default=2e-9)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    args = parser.parse_args()
    forecast_payload = {
        "preterminal_stress_peak_projection": {
            "projected_peak_times_s": [row["projected_peak_time_s"]
                                       for row in json.loads(
                                           args.forecast.read_text())["fits"]]
        }
    }
    while True:
        payload = evaluate(
            load_rows(args.telemetry), forecast_payload,
            confirmation_lag_s=args.confirmation_lag,
            forecast_tolerance_s=args.forecast_tolerance)
        payload["forecast_artifact"] = str(args.forecast.resolve())
        payload["forecast_artifact_sha256"] = digest(args.forecast)
        atomic_json(args.output, payload)
        alive = process_alive(args.solver_pid)
        finished = bool(payload["local_scalar_peak_confirmed"] or not alive)
        atomic_json(args.status, {
            "schema": "asb-drx-v65-peak-forecast-monitor-status-v1",
            "status": "COMPLETE" if finished else "RUNNING",
            "solver_pid": args.solver_pid,
            "classification": payload["classification"],
            "latest_interval": payload["latest_interval"],
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        })
        if finished:
            break
        time.sleep(max(args.poll_seconds, 1.0))


if __name__ == "__main__":
    main()
