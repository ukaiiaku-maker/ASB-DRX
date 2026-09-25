#!/usr/bin/env python3
"""Compare a fine-subdivision and transitioned-macro DRX continuation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(left, right, floor=1e-300):
    return abs(float(left)-float(right))/max(abs(float(left)), abs(float(right)), floor)


def increments(result, start):
    rows = result["records"]
    previous = rows[start-1]
    selected = rows[start:]
    return {
        "area_equivalent_displacement_m": sum(
            row["accepted_contour_displacement_m"] for row in selected),
        "newly_swept_volume_m3": (
            rows[-1]["cumulative_newly_swept_volume_m3"]
            -previous["cumulative_newly_swept_volume_m3"]),
        "processed_line_m": (
            rows[-1]["cumulative_processed_line_m"]
            -previous["cumulative_processed_line_m"]),
        "boundary_stored_line_m": (
            rows[-1]["cumulative_boundary_stored_line_m"]
            -previous["cumulative_boundary_stored_line_m"]),
        "stress_change_Pa": (
            rows[-1]["mean_shear_stress_Pa"]-previous["mean_shear_stress_Pa"]),
        "plastic_shear_change": (
            rows[-1]["engineering_plastic_shear"]
            -previous["engineering_plastic_shear"]),
        "child_owner_density_change_m2": (
            rows[-1]["child_owner_total_line_density_mean_m2"]
            -previous["child_owner_total_line_density_mean_m2"]),
        "temperature_peak_change_K": (
            rows[-1]["temperature_range_K"][1]
            -previous["temperature_range_K"][1]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--fine-result", type=Path, required=True)
    parser.add_argument("--fine-checkpoint", type=Path, required=True)
    parser.add_argument("--macro-result", type=Path, required=True)
    parser.add_argument("--macro-checkpoint", type=Path, required=True)
    parser.add_argument("--start-interval", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fine = json.loads(args.fine_result.read_text())
    macro = json.loads(args.macro_result.read_text())
    fine_increment = increments(fine, args.start_interval)
    macro_increment = increments(macro, args.start_interval)
    observable_errors = {
        key: relative(fine_increment[key], macro_increment[key])
        for key in fine_increment
    }
    fields = (
        "eta", "mechanical__v24_common__beta_p",
        "mechanical__v24_common__family_nye_m1",
        "mechanical__v24_common__orientation_rad",
        "mechanical__v24_common__temperature_K",
    )
    field_errors = {}
    with np.load(args.base_checkpoint, allow_pickle=False) as base, np.load(
            args.fine_checkpoint, allow_pickle=False) as fine_state, np.load(
            args.macro_checkpoint, allow_pickle=False) as macro_state:
        for name in fields:
            initial = np.asarray(base[name], dtype=float)
            fine_value = np.asarray(fine_state[name], dtype=float)
            macro_value = np.asarray(macro_state[name], dtype=float)
            difference = float(np.linalg.norm(macro_value-fine_value))
            increment_scale = max(
                float(np.linalg.norm(fine_value-initial)),
                float(np.linalg.norm(macro_value-initial)), 1e-300)
            state_scale = max(float(np.linalg.norm(fine_value)),
                              float(np.linalg.norm(macro_value)), 1e-300)
            field_errors[name] = {
                "difference_l2": difference,
                "relative_to_increment_l2": difference/increment_scale,
                "relative_to_state_l2": difference/state_scale,
                "maximum_absolute_difference": float(
                    np.max(np.abs(macro_value-fine_value))),
            }
    front_keys = (
        "area_equivalent_displacement_m", "newly_swept_volume_m3",
        "processed_line_m", "boundary_stored_line_m",
    )
    physical_time_exact = bool(np.isclose(
        fine["physical_time_s"], macro["physical_time_s"], rtol=0.0,
        atol=32*np.finfo(float).eps*max(fine["physical_time_s"], 1e-300)))
    ledgers_pass = all(
        row.get("loading_energy_audit", {}).get("first_law_passed", False)
        for result in (fine, macro)
        for row in result["records"][args.start_interval:])
    passed = bool(
        physical_time_exact and ledgers_pass
        and max(observable_errors[key] for key in front_keys) <= 0.05
        and field_errors["mechanical__v24_common__beta_p"][
            "relative_to_increment_l2"] <= 0.05
        and field_errors["mechanical__v24_common__family_nye_m1"][
            "relative_to_increment_l2"] <= 0.05)
    payload = {
        "schema": "asb-drx/v55/drx-dt-overlap/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "start_interval": args.start_interval,
        "physical_time_exact": physical_time_exact,
        "loading_ledgers_passed": ledgers_pass,
        "fine": {"result": str(args.fine_result.resolve()),
                 "sha256": digest(args.fine_result),
                 "increment": fine_increment},
        "macro": {"result": str(args.macro_result.resolve()),
                  "sha256": digest(args.macro_result),
                  "increment": macro_increment,
                  "numerical_method_transitions": macro.get(
                      "numerical_method_transitions", [])},
        "observable_relative_errors": observable_errors,
        "field_errors": field_errors,
        "provisional_five_percent_overlap_passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"output": str(args.output), "passed": passed,
                      "sha256": digest(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
