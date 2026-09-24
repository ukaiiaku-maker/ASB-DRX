#!/usr/bin/env python3
"""Apply read-only common-state subgrain recognition to an evolved checkpoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from full_model.production.common_subgrain_recognition import (
    recognize_common_orientation_plateau,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()
                if not key.endswith("_mask")}
    if isinstance(value, list):
        return [clean(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prefix", default="v24_common__")
    parser.add_argument("--spacing-m", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.checkpoint, allow_pickle=False) as state:
        prefix = args.prefix
        orientation = np.asarray(state[prefix+"orientation_rad"])
        nye = np.asarray(state[prefix+"family_nye_m1"])
        order = np.asarray(state[prefix+"wall_order"])
        density = sum(np.sum(np.asarray(state[prefix+name]), axis=2) for name in (
            "mobile_plus_m2", "mobile_minus_m2", "forest_plus_m2",
            "forest_minus_m2", "wall_plus_m2", "wall_minus_m2",
            "junction_m2"))
        spacing = args.spacing_m
        metadata = None
        for key in ("v30_metadata_json", "v36_response_metadata_json"):
            if key in state.files:
                metadata = json.loads(str(state[key].item()))
                break
        if spacing is None and metadata is not None:
            configuration = metadata.get("configuration", metadata)
            length = configuration.get("length_m")
            grid = configuration.get("grid", orientation.shape[0])
            if length is not None:
                spacing = float(length)/int(grid)
        if spacing is None:
            raise ValueError("spacing is not present in metadata; pass --spacing-m")
        before = {name: np.asarray(state[prefix+name]).copy() for name in (
            "orientation_rad", "family_nye_m1", "wall_order")}
        recognition = recognize_common_orientation_plateau(
            orientation, nye, order, density, spacing)
        unchanged = all(np.array_equal(before[name], state[prefix+name])
                        for name in before)
    payload = {
        "schema": "asb-drx/v55/current-state-subgrain-audit/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": digest(args.checkpoint),
        "field_prefix": args.prefix,
        "spacing_m": spacing,
        "recognition_is_read_only": unchanged,
        "recognition": clean(recognition),
        "phase_allocation_attempted": False,
        "decision": (
            "QUALIFIED_FOR_SEPARATE_NEUTRAL_HANDOFF"
            if recognition.get("qualified", False)
            else "NO_QUALIFIED_INTRAGRANULAR_BOUNDARY_IN_THIS_EVOLVED_STATE"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"output": str(args.output), "decision": payload["decision"],
                      "sha256": digest(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
