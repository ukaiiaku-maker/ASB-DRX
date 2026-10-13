#!/usr/bin/env python3
"""Track a prepared grain with one registered geometric convention."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from full_model.analysis.grain_response_metrics import grain_response
from full_model.analysis.run_v58_three_grain_production import _load_checkpoint


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--grain-id", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    initial, initial_runtime, initial_step, initial_load, _, config, _ = (
        _load_checkpoint(args.origin))
    physical = (config["physical_configuration"] if config and config.get(
        "schema") == "asb-drx-v64-adaptive-production-v1" else config)
    spacing = float(physical["length_m"])/int(physical["n"])
    thickness = 2.0*2.48e-10
    try:
        grain_index = tuple(initial.grain_ids).index(args.grain_id)
    except ValueError as error:
        raise ValueError("requested grain is absent from origin") from error
    rows = []
    for checkpoint in args.checkpoint:
        state, runtime, step, load, *_ = _load_checkpoint(checkpoint)
        if tuple(state.grain_ids) != tuple(initial.grain_ids):
            raise ValueError("grain identities changed across tracked checkpoints")
        response = grain_response(
            initial.supports, state.supports, spacing_m=spacing,
            represented_thickness_m=thickness)
        rows.append({
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256(checkpoint),
            "adaptive_interval": int(step),
            "physical_time_s": runtime.ledger.physical_time_s,
            "applied_shear_strain": float(load),
            "grain_id": args.grain_id,
            "volume_growth_fraction": float(
                response["volume_growth_fraction"][grain_index]),
            "equivalent_contour_displacement_m": float(
                response["equivalent_contour_displacement_m"][grain_index]),
            "equivalent_contour_displacement_cells": float(
                response["equivalent_contour_displacement_m"][grain_index]
                /spacing),
            "initial_perimeter_only_displacement_cells": float(
                response["initial_perimeter_only_displacement_m"][grain_index]
                /spacing),
            "initial_perimeter_m": float(
                response["initial_perimeter_m"][grain_index]),
            "final_perimeter_m": float(
                response["final_perimeter_m"][grain_index]),
        })
    rows.sort(key=lambda row: row["physical_time_s"])
    for index, row in enumerate(rows):
        if index == 0:
            row["incremental_displacement_velocity_m_s"] = None
            row["growth_trend"] = "ORIGIN"
            continue
        previous = rows[index-1]
        dt = row["physical_time_s"]-previous["physical_time_s"]
        velocity = ((row["equivalent_contour_displacement_m"]
                     -previous["equivalent_contour_displacement_m"])/dt)
        row["incremental_displacement_velocity_m_s"] = velocity
        row["growth_trend"] = ("ADVANCING" if velocity > 0.0 else
                               "RECEDING" if velocity < 0.0 else "STATIONARY")
    payload = {
        "schema": "asb-drx-v65-grain-selection-history-v1",
        "origin": {"path": str(args.origin.resolve()),
                   "sha256": sha256(args.origin),
                   "step": int(initial_step),
                   "physical_time_s": initial_runtime.ledger.physical_time_s,
                   "applied_shear_strain": float(initial_load)},
        "grain_id": args.grain_id,
        "spacing_m": spacing,
        "displacement_convention": response["convention"],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output),
                      "rows": len(rows)}))


if __name__ == "__main__":
    main()
