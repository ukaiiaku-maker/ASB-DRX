#!/usr/bin/env python3
"""Classify a matched long-horizon prepared-boundary DRX continuation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


BURGERS_M = 2.48e-10
SUBSTANTIAL_TRANSFORMED_FRACTION = 0.01


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def continuation_summary(path: Path, start_interval: int) -> dict:
    result = load(path)
    rows = result["records"]
    if len(rows) < start_interval or rows[-1]["interval"] <= start_interval:
        raise ValueError(f"{path} does not extend interval {start_interval}")
    before = rows[start_interval - 1]
    selected = rows[start_interval:]
    ledgers = [row.get("loading_energy_audit") for row in selected]
    ledgers = [ledger for ledger in ledgers if ledger is not None]
    front_ledgers = [row.get("complete_energy_decision") for row in selected]
    front_ledgers = [ledger for ledger in front_ledgers if ledger is not None]
    displacement = sum(
        float(row["accepted_contour_displacement_m"]) for row in selected)
    fraction_change = sum(float(row["child_fraction_change"]) for row in selected)
    return {
        "result": str(path.resolve()),
        "sha256": digest(path),
        "source_commit": result.get("source_commit"),
        "completed_intervals": int(result["completed_intervals"]),
        "physical_time_s": float(result["physical_time_s"]),
        "continuation_time_s": float(
            selected[-1]["physical_time_end_s"] - before["physical_time_end_s"]),
        "continuation_displacement_m": displacement,
        "continuation_displacement_b": displacement / BURGERS_M,
        "continuation_child_fraction_change": fraction_change,
        "continuation_newly_swept_volume_m3": float(
            selected[-1]["cumulative_newly_swept_volume_m3"]
            - before["cumulative_newly_swept_volume_m3"]),
        "continuation_processed_line_m": float(
            selected[-1]["cumulative_processed_line_m"]
            - before["cumulative_processed_line_m"]),
        "endpoint_child_fraction": float(selected[-1]["current_child_fraction"]),
        "endpoint_stress_Pa": float(selected[-1]["mean_shear_stress_Pa"]),
        "endpoint_plastic_shear": float(selected[-1]["engineering_plastic_shear"]),
        "endpoint_temperature_range_K": selected[-1]["temperature_range_K"],
        "loading_ledgers_passed": bool(ledgers and all(
            ledger["first_law_passed"] for ledger in ledgers)),
        "front_ledgers_passed": bool(all(
            abs(float(ledger["first_law_residual_J"]))
            <= float(ledger["tolerance_J"]) for ledger in front_ledgers)),
        "numerical_method_transitions": result.get(
            "numerical_method_transitions", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enabled-result", type=Path, required=True)
    parser.add_argument("--disabled-result", type=Path, required=True)
    parser.add_argument("--start-interval", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    enabled = continuation_summary(args.enabled_result, args.start_interval)
    disabled = continuation_summary(args.disabled_result, args.start_interval)
    common_clock = bool(
        enabled["completed_intervals"] == disabled["completed_intervals"]
        and np.isclose(enabled["physical_time_s"], disabled["physical_time_s"],
                       rtol=0.0, atol=32*np.finfo(float).eps
                       * max(enabled["physical_time_s"], 1e-300)))
    hard_valid = bool(
        common_clock and enabled["loading_ledgers_passed"]
        and disabled["loading_ledgers_passed"]
        and enabled["front_ledgers_passed"]
        and disabled["front_ledgers_passed"]
        and disabled["continuation_newly_swept_volume_m3"] == 0.0
        and disabled["continuation_displacement_m"] == 0.0)
    substantial = bool(
        abs(enabled["continuation_child_fraction_change"])
        >= SUBSTANTIAL_TRANSFORMED_FRACTION)
    if not hard_valid:
        classification = "HARD_INVALID_OR_MATCHED_CONTROL_FAILED"
    elif substantial:
        classification = "PREPARED_BOUNDARY_SUBSTANTIAL_GROWTH_DEMONSTRATED"
    elif enabled["continuation_newly_swept_volume_m3"] > 0.0:
        classification = "VALID_CONTINUUM_MOTION_WITHOUT_SUBSTANTIAL_TRANSFORMATION"
    else:
        classification = "VALID_ARREST_OR_NO_RESOLVED_GROWTH"
    payload = {
        "schema": "asb-drx/v55/drx-long-continuation/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "start_interval": args.start_interval,
        "classification": classification,
        "hard_valid": hard_valid,
        "common_clock": common_clock,
        "substantial_growth_threshold_fraction": SUBSTANTIAL_TRANSFORMED_FRACTION,
        "substantial_growth_demonstrated": substantial,
        "enabled": enabled,
        "front_disabled_control": disabled,
        "effect": {
            "child_fraction_change_difference": (
                enabled["continuation_child_fraction_change"]
                - disabled["continuation_child_fraction_change"]),
            "stress_difference_Pa": (
                enabled["endpoint_stress_Pa"] - disabled["endpoint_stress_Pa"]),
            "plastic_shear_difference": (
                enabled["endpoint_plastic_shear"]
                - disabled["endpoint_plastic_shear"]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"classification": classification,
                      "hard_valid": hard_valid,
                      "output": str(args.output),
                      "sha256": digest(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
