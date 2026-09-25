#!/usr/bin/env python3
"""Compare fresh fine-step and macrostep DRX trajectories at equal time."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(a: float, b: float) -> float:
    return abs(float(a) - float(b)) / max(abs(float(a)), abs(float(b)), 1e-300)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fine-result", type=Path, required=True)
    parser.add_argument("--fine-checkpoint", type=Path, required=True)
    parser.add_argument("--macro-result", type=Path, required=True)
    parser.add_argument("--macro-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fine = json.loads(args.fine_result.read_text())
    macro = json.loads(args.macro_result.read_text())
    fine_end, macro_end = fine["records"][-1], macro["records"][-1]
    cumulative = {
        "contour_displacement_m": (
            sum(row["accepted_contour_displacement_m"]
                for row in fine["records"]),
            sum(row["accepted_contour_displacement_m"]
                for row in macro["records"])),
        "newly_swept_volume_m3": (
            fine_end["cumulative_newly_swept_volume_m3"],
            macro_end["cumulative_newly_swept_volume_m3"]),
        "processed_line_m": (fine_end["cumulative_processed_line_m"],
                             macro_end["cumulative_processed_line_m"]),
        "boundary_stored_line_m": (
            fine_end["cumulative_boundary_stored_line_m"],
            macro_end["cumulative_boundary_stored_line_m"]),
        "child_fraction": (fine_end["current_child_fraction"],
                           macro_end["current_child_fraction"]),
        "stress_Pa": (fine_end["mean_shear_stress_Pa"],
                      macro_end["mean_shear_stress_Pa"]),
        "plastic_shear": (fine_end["engineering_plastic_shear"],
                          macro_end["engineering_plastic_shear"]),
    }
    observable_errors = {key: relative(*values)
                         for key, values in cumulative.items()}
    field_names = (
        "eta", "mechanical__v24_common__beta_p",
        "mechanical__v24_common__family_nye_m1",
        "mechanical__v24_common__orientation_rad",
        "mechanical__v24_common__temperature_K")
    field_errors = {}
    with np.load(args.fine_checkpoint, allow_pickle=False) as fine_state, \
            np.load(args.macro_checkpoint, allow_pickle=False) as macro_state:
        for name in field_names:
            a = np.asarray(fine_state[name], dtype=float)
            b = np.asarray(macro_state[name], dtype=float)
            difference = float(np.linalg.norm(a-b))
            field_errors[name] = {
                "relative_to_endpoint_l2": difference / max(
                    float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1e-300),
                "maximum_absolute_difference": float(np.max(np.abs(a-b))),
            }
    exact_time = bool(np.isclose(
        fine["physical_time_s"], macro["physical_time_s"], rtol=0.0,
        atol=32*np.finfo(float).eps*max(fine["physical_time_s"], 1e-300)))
    ledgers_passed = all(
        row.get("loading_energy_audit", {}).get("first_law_passed", False)
        for result in (fine, macro) for row in result["records"])
    passed = bool(
        exact_time and ledgers_passed
        and max(observable_errors.values()) <= 0.05
        and field_errors["mechanical__v24_common__beta_p"][
            "relative_to_endpoint_l2"] <= 0.05
        and field_errors["mechanical__v24_common__family_nye_m1"][
            "relative_to_endpoint_l2"] <= 0.05)
    payload = {
        "schema": "asb-drx/v55/drx-fresh-overlap/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "fine": {"result": str(args.fine_result.resolve()),
                 "sha256": digest(args.fine_result)},
        "macro": {"result": str(args.macro_result.resolve()),
                  "sha256": digest(args.macro_result)},
        "physical_time_exact": exact_time,
        "loading_ledgers_passed": ledgers_passed,
        "observable_endpoint_values": {
            key: {"fine": values[0], "macro": values[1]}
            for key, values in cumulative.items()},
        "observable_relative_errors": observable_errors,
        "field_errors": field_errors,
        "provisional_five_percent_overlap_passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "passed": passed,
                      "sha256": digest(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
