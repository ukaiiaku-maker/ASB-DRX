#!/usr/bin/env python3
"""Compare paired post-arrest common-state continuations at one physical horizon."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


RELATIVE_METRICS = (
    "family_nye_rms_m1",
    "orientation_range_rad",
    "beta_p_rms",
    "total_line_m_per_m_thickness",
    "temperature_maximum_K",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_difference(left: float, right: float) -> float:
    return abs(left-right)/max(abs(right), 1e-30)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse", type=Path, required=True)
    parser.add_argument("--fine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    coarse = json.loads(args.coarse.read_text())
    fine = json.loads(args.fine.read_text())
    coarse_final = coarse["records"][-1]["state_metrics"]
    fine_final = fine["records"][-1]["state_metrics"]
    relative = {key: relative_difference(coarse_final[key], fine_final[key])
                for key in RELATIVE_METRICS}
    displacement_difference = abs(
        coarse["cumulative_contour_displacement_m"]
        -fine["cumulative_contour_displacement_m"])
    same_horizon = abs(coarse["physical_time_s"]-fine["physical_time_s"]) <= 1e-15
    passed = bool(
        same_horizon
        and all(value <= .05 for value in relative.values())
        and displacement_difference <= 1e-12)
    result = {
        "schema": "asb-drx/v40/post-arrest-temporal/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "coarse": {"path": str(args.coarse.resolve()),
                       "sha256": digest(args.coarse)},
            "fine": {"path": str(args.fine.resolve()),
                     "sha256": digest(args.fine)},
        },
        "physical_time_s": fine["physical_time_s"],
        "same_physical_horizon": same_horizon,
        "coarse_macro_dt_s": coarse["macro_dt_s"],
        "fine_macro_dt_s": fine["macro_dt_s"],
        "relative_threshold": .05,
        "near_zero_front_absolute_tolerance_m": 1e-12,
        "relative_differences": relative,
        "incremental_front_displacement_difference_m": displacement_difference,
        "temporal_refinement_passed": passed,
        "classification": (
            "POST_ARREST_TEMPORAL_PAIR_QUALIFIED" if passed else
            "POST_ARREST_TEMPORAL_PAIR_UNRESOLVED"),
        "claim_boundary": (
            "This pair tests time integration after the observed arrest; it "
            "does not repair the failed n128/n192 gradient-state spatial gate."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
