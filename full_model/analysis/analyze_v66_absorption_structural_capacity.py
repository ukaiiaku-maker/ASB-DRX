#!/usr/bin/env python3
"""Upper-bound Frank--Bilby capacity of the absorbed signed precursor."""

from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from full_model.production.common_tensorial_wall import CommonWallState


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT/"full_model"/"verification"/"v66_polarized_absorption_handoff"
          /"final_state.npz")
OUT = (ROOT/"full_model"/"verification"/
       "v66_absorption_structural_capacity.json")
BURGERS_M = 2.48e-10
SPACING_M = 2.0e-7


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _periodic_separation_capacity_m1(signed_density_m2):
    """Largest zero-mean signed inventory between two periodic cuts.

    For each family and each candidate wall normal, integrate the fluctuating
    signed density.  The cumulative range is the most generous content that
    can be separated into a compatible wall pair without creating Burgers
    content.  A crystallographic projection factor of one is deliberately
    used, so this is an upper bound rather than a boundary claim.
    """
    values = np.asarray(signed_density_m2, dtype=float)
    records = []
    for axis in (0, 1):
        zero_mean = values-np.mean(values, axis=axis, keepdims=True)
        cumulative = np.cumsum(zero_mean, axis=axis)*SPACING_M
        ranges = np.max(cumulative, axis=axis)-np.min(cumulative, axis=axis)
        location = np.unravel_index(int(np.argmax(ranges)), ranges.shape)
        records.append({
            "normal_axis": axis,
            "maximum_separable_signed_content_m1": float(np.max(ranges)),
            "transverse_index": int(location[0]),
            "family": int(location[1]),
        })
    return max(records, key=lambda item: item[
        "maximum_separable_signed_content_m1"]), records


def main():
    with np.load(SOURCE) as data:
        state = CommonWallState(**{
            item.name: np.asarray(data[item.name]).copy()
            for item in fields(CommonWallState)})
    wall_signed = state.wall_plus_m2-state.wall_minus_m2
    family_signed = (
        state.mobile_plus_m2-state.mobile_minus_m2
        +state.forest_plus_m2-state.forest_minus_m2
        +wall_signed)
    wall_best, wall_axes = _periodic_separation_capacity_m1(wall_signed)
    all_best, all_axes = _periodic_separation_capacity_m1(family_signed)
    observed_span = float(np.ptp(state.orientation_rad))
    threshold = np.deg2rad(2.0)
    wall_upper = BURGERS_M*wall_best["maximum_separable_signed_content_m1"]
    all_upper = BURGERS_M*all_best["maximum_separable_signed_content_m1"]
    result = {
        "schema": "v66-absorption-structural-capacity-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(SOURCE.relative_to(ROOT)),
        "source_sha256": _sha(SOURCE),
        "method": (
            "periodic cumulative range of zero-mean signed density by family "
            "and candidate wall normal; unit crystallographic projection gives "
            "a generous Frank--Bilby upper bound"),
        "spacing_m": SPACING_M,
        "burgers_m": BURGERS_M,
        "wall_reservoir": {
            "best": wall_best, "all_axes": wall_axes,
            "misorientation_upper_bound_rad": wall_upper,
            "misorientation_upper_bound_deg": float(np.rad2deg(wall_upper)),
        },
        "all_family_reservoirs": {
            "best": all_best, "all_axes": all_axes,
            "misorientation_upper_bound_rad": all_upper,
            "misorientation_upper_bound_deg": float(np.rad2deg(all_upper)),
        },
        "observed_orientation_span_rad": observed_span,
        "observed_orientation_span_deg": float(np.rad2deg(observed_span)),
        "recognition_threshold_deg": 2.0,
        "decision": {
            "absorbed_wall_content_can_support_recognition_threshold": bool(
                wall_upper >= threshold),
            "all_signed_content_can_support_recognition_threshold": bool(
                all_upper >= threshold),
            "structural_handoff_allowed": False,
            "reason": (
                "capacity is an upper bound and no closed ordered interface "
                "or two-degree compatible plateau is present"),
        },
        "claim_boundary": (
            "no sign selection, no imposed rotation, no grain allocation; "
            "crystallographic projection and closure can only reduce capacity"),
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
