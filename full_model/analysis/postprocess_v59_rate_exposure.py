#!/usr/bin/env python3
"""Compare fixed-parameter V59 states at common strain and common time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from full_model.analysis.postprocess_v59_physical_asb import checkpoint_snapshot
from full_model.analysis.run_v58_three_grain_production import _load_checkpoint


INITIAL_ENGINEERING_SHEAR = 0.012


def exposure_record(path: Path) -> dict:
    state, runtime, step, gamma, initial_volume, configuration, provenance = (
        _load_checkpoint(path))
    snapshot, _ = checkpoint_snapshot(path)
    spacing = float(configuration["length_m"])/int(configuration["n"])
    thickness = 2.0*2.48e-10
    cell_volume = spacing**2*thickness
    final_volume = np.sum(state.supports, axis=(1, 2))*cell_volume
    return {
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": snapshot["checkpoint_sha256"],
        "source_commit": (provenance or {}).get("source_commit"),
        "step": step, "physical_time_s": runtime.ledger.physical_time_s,
        "shear_rate_s-1": configuration["shear_rate_s"],
        "initial_temperature_K": configuration["temperature_K"],
        "engineering_shear": gamma,
        "additional_engineering_shear": gamma-INITIAL_ENGINEERING_SHEAR,
        "post_front_equilibrated_stress_Pa": snapshot[
            "post_front_equilibrated_stress_Pa"],
        "temperature_mean_K": snapshot["temperature_mean_K"],
        "temperature_max_minus_mean_K": snapshot[
            "temperature_max_minus_mean_K"],
        "temperature_max_minus_min_K": snapshot["temperature_max_minus_min_K"],
        "power_participation": snapshot["plastic_power"][
            "inverse_participation_fraction"],
        "power_width_minor_m": snapshot["power_width_minor_m"],
        "power_aspect_ratio": snapshot["power_aspect_ratio"],
        "net_grain_volume_change_m3": (final_volume-initial_volume).tolist(),
        "fresh_sweep_volume_m3": state.ledger.fresh_sweep_fraction*cell_volume,
        "revisit_sweep_volume_m3": state.ledger.revisit_sweep_fraction*cell_volume,
        "accepted_front_events": runtime.ledger.accepted_events,
        "rejected_front_events": runtime.ledger.rejected_events,
    }


def compare(left: dict, right: dict, *, mode: str) -> dict:
    if mode == "common_strain":
        scale = max(abs(left["additional_engineering_shear"]),
                    abs(right["additional_engineering_shear"]), 1e-300)
        matched = abs(left["additional_engineering_shear"]-
                      right["additional_engineering_shear"]) <= 1e-10*scale
    else:
        scale = max(abs(left["physical_time_s"]),
                    abs(right["physical_time_s"]), 1e-300)
        matched = abs(left["physical_time_s"]-
                      right["physical_time_s"]) <= 1e-10*scale
    return {
        "mode": mode, "exposure_matched": bool(matched),
        "left": left, "right": right,
        "differences_right_minus_left": {
            key: right[key]-left[key] for key in (
                "post_front_equilibrated_stress_Pa", "temperature_mean_K",
                "temperature_max_minus_mean_K", "temperature_max_minus_min_K",
                "power_participation", "power_width_minor_m",
                "power_aspect_ratio", "fresh_sweep_volume_m3")},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--mode", choices=("common_strain", "common_time"),
                        required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"schema": "asb-drx-v59-rate-exposure-v1", **compare(
        exposure_record(args.left), exposure_record(args.right), mode=args.mode)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result["differences_right_minus_left"], indent=2))


if __name__ == "__main__":
    main()
