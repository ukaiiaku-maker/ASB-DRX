#!/usr/bin/env python3
"""Repair V37 front-rate selection semantics without changing raw evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from full_model.production.arrhenius_kinetics import EV_J
from full_model.production.coupled_front_event import (
    front_barrier_activation_volume_m3,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.screen.read_text())
    rows = []
    for raw in source["records"]:
        parameters = raw["parameters"]
        channels = {}
        for direction in ("a_to_b", "b_to_a"):
            old = raw[f"channel_{direction}"]
            pressure = float(old["driving_pressure_magnitude_Pa"])
            channels[direction] = {
                **old,
                "front_channel_pressure_Pa": pressure,
                "front_event_volume_m3": float(raw["event_volume_m3"]),
                "front_activation_volume_m3": (
                    front_barrier_activation_volume_m3(
                        pressure,
                        float(parameters["activation_h0_eV"])*EV_J,
                        1.0e9, float(parameters["exp_a"]),
                        float(parameters["exp_n"]), 0.10)),
                "legacy_pressure_key_preserved": (
                    "driving_pressure_magnitude_Pa"),
            }
        rows.append({
            "name": raw["name"],
            "net_velocity_m_s": float(raw["net_velocity_m_s"]),
            "time_to_quarter_width_s": float(raw["time_to_quarter_width_s"]),
            "front_event_volume_m3": float(raw["event_volume_m3"]),
            "front_jump_length_m": float(raw["jump_length_m"]),
            "availability": float(parameters["availability"]),
            "exp_n": float(parameters["exp_n"]),
            "channels": channels,
            "raw_record": raw,
        })
    rows.sort(key=lambda row: row["net_velocity_m_s"])
    names = [row["name"] for row in rows]
    baseline = next(row for row in rows if row["name"] == "baseline")
    slower = next(row for row in rows if row["name"] == "availability_mid")
    assert slower["net_velocity_m_s"] < baseline["net_velocity_m_s"]
    result = {
        "schema": "asb-drx/v38/front-rate-ranking/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_screen": str(args.screen.resolve()),
        "source_screen_sha256": sha256(args.screen),
        "raw_rate_order_slowest_to_fastest": names,
        "rows_slowest_to_fastest": rows,
        "corrected_selection": {
            "slow_reference": "baseline",
            "slower_availability_control": "availability_mid",
            "genuine_intermediate_candidate": "shape_a_high",
            "fast_pressure_sensitivity_hypothesis": "shape_n_low",
        },
        "availability_ratio_to_baseline": (
            slower["net_velocity_m_s"]/baseline["net_velocity_m_s"]),
        "semantic_decision": (
            "availability_mid is a slower-availability control; shape_n_low "
            "changes front-barrier pressure sensitivity and is not a mere multiplier"),
        "screen_is_trajectory": False,
        "material_calibration_claimed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
