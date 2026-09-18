#!/usr/bin/env python3
"""Classify V40 accepted front motion, arrest, and reverse control."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def compact_event(row: dict | None) -> dict | None:
    if row is None:
        return None
    keys = ("interval", "physical_time_end_s", "front_rate_m_s",
            "front_classification", "front_published",
            "front_contour_displacement_m", "front_complete_energy_delta_J")
    return {**{key: row[key] for key in keys},
            "state_metrics": row["state_metrics"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward", type=Path, required=True)
    parser.add_argument("--reverse", type=Path, required=True)
    parser.add_argument("--forward-diagnostic", type=Path, required=True)
    parser.add_argument("--reverse-diagnostic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    forward = load(args.forward); reverse = load(args.reverse)
    forward_diagnostic = load(args.forward_diagnostic)["records"][-1]
    reverse_diagnostic = load(args.reverse_diagnostic)["records"][-1]
    records = forward["records"]
    first_unpublished = next((row for row in records
                              if not row["front_published"]), None)
    first_negative = next((row for row in records
                           if row["front_rate_m_s"] < 0.0), None)
    accepted = [row for row in records if row["front_published"]]
    last_accepted = accepted[-1]
    start = records.index(first_unpublished) if first_unpublished else len(records)
    tail = records[start:]
    internal_change = {}
    if len(tail) >= 2:
        for key in ("family_nye_rms_m1", "orientation_range_rad",
                    "temperature_maximum_K", "total_line_m_per_m_thickness"):
            internal_change[key] = (
                tail[-1]["state_metrics"][key]
                -tail[0]["state_metrics"][key])
    target = 0.25*4e-7
    maximum_advance = max(np.cumsum([
        row["front_contour_displacement_m"] for row in records]),
        default=0.0)
    direction_diagnostics = {
        "forward_proposal": {key: value for key, value in
                             forward_diagnostic.items() if key.startswith("front_")},
        "reverse_proposal": {key: value for key, value in
                             reverse_diagnostic.items() if key.startswith("front_")},
    }
    pinned = bool(
        forward_diagnostic["front_kinetic_accepted"] is False
        and reverse_diagnostic["front_kinetic_accepted"] is True
        and reverse_diagnostic["front_thermodynamic_accepted"] is False)
    result = {
        "schema": "asb-drx/v40/common-arrest-decision/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {name: {"path": str(path.resolve()), "sha256": digest(path)}
                   for name, path in (("forward", args.forward),
                                      ("reverse", args.reverse),
                                      ("forward_diagnostic", args.forward_diagnostic),
                                      ("reverse_diagnostic", args.reverse_diagnostic))},
        "forward_source_sha": forward["source_sha"],
        "forward_horizon_s": forward["physical_time_s"],
        "maximum_accepted_advance_m": float(maximum_advance),
        "quarter_interface_width_target_m": target,
        "target_fraction_reached": float(maximum_advance/target),
        "first_rate_sign_change": compact_event(first_negative),
        "first_unpublished_transaction": compact_event(first_unpublished),
        "last_accepted_transaction": compact_event(last_accepted),
        "direction_diagnostics": direction_diagnostics,
        "reverse_control_horizon_s": reverse["physical_time_s"],
        "reverse_control_additional_published_events": int(sum(
            row["front_published"] for row in reverse["records"]
            if row["physical_time_end_s"] > first_unpublished[
                "physical_time_end_s"])),
        "internal_state_change_during_stationary_front": internal_change,
        "stationary_front_is_stationary_internal_state": bool(
            not internal_change or all(value == 0.0
                                       for value in internal_change.values())),
        "physical_arrest_supported": pinned,
        "classification": (
            "VALID_PINNED_CRITICAL_ARREST_RATE_SIGN_CHANGE_WITH_REVERSE_ENERGY_REJECTION"
            if pinned else "ARREST_MECHANISM_UNRESOLVED"),
        "drx_claimed": False, "strict_asb_claimed": False,
        "claim_boundary": (
            "n128 exploratory common trajectory; migration arrest is distinct "
            "from continuing dislocation, orientation, and thermal evolution."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    time = np.array([row["physical_time_end_s"] for row in records])
    displacement = np.cumsum([
        row["front_contour_displacement_m"] for row in records])*1e9
    figure, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes[0, 0].plot(time*1e3, displacement)
    axes[0, 0].set_ylabel("accepted cumulative displacement (nm)")
    axes[0, 1].plot(time*1e3, [row["front_rate_m_s"]*1e6 for row in records])
    axes[0, 1].axhline(0.0, color="k", linewidth=.7)
    axes[0, 1].set_ylabel("proposed net rate ($\\mu$m/s)")
    axes[1, 0].plot(time*1e3, [row["state_metrics"]["family_nye_rms_m1"]
                              for row in records])
    axes[1, 0].set_ylabel("family Nye RMS (m$^{-1}$)")
    axes[1, 1].plot(time*1e3, [np.rad2deg(
        row["state_metrics"]["orientation_range_rad"]) for row in records],
        label="orientation span (deg)")
    axes[1, 1].plot(time*1e3, [row["state_metrics"]["temperature_maximum_K"]-1100
                              for row in records], label="$T_{max}-T_0$ (K)")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.set_xlabel("time (ms)"); axis.grid(alpha=.25)
    figure.tight_layout(); args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure, dpi=180); plt.close(figure)


if __name__ == "__main__":
    main()
