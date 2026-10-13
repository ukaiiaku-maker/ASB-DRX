#!/usr/bin/env python3
"""Classify each newly durable V65 checkpoint while production continues."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from full_model.analysis.grain_response_metrics import grain_response
from full_model.analysis.postprocess_v59_physical_asb import (
    PhysicalASBCriteria, checkpoint_snapshot,
)
from full_model.analysis.run_v58_three_grain_production import _load_checkpoint
from full_model.analysis.summarize_v65_live_trends import (
    load_rows as load_trend_rows, summarize as summarize_trends,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload) -> None:
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    os.replace(temporary, path)


def checkpoint_index(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


def nonnegative_softening(peak_stress: float, stress: float) -> float:
    """Return physical post-peak loss without roundoff-scale negatives."""
    if peak_stress <= 0.0:
        return 0.0
    return max(0.0, (peak_stress-stress)/peak_stress)


def accepted_history(run: Path, maximum_interval: int) -> list[dict]:
    telemetry = run/"accepted_telemetry.jsonl"
    if not telemetry.is_file():
        return []
    rows = []
    for line in telemetry.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row["adaptive_interval"]) <= maximum_interval:
            rows.append(row)
    return rows


def classify(checkpoint: Path, origin: Path, run: Path,
             target_grain_id: int = 20) -> dict:
    initial, *_ = _load_checkpoint(origin)
    state, runtime, step, load, _, raw_configuration, provenance = (
        _load_checkpoint(checkpoint))
    physical = (raw_configuration["physical_configuration"]
                if raw_configuration.get("schema") ==
                "asb-drx-v64-adaptive-production-v1"
                else raw_configuration)
    spacing = float(physical["length_m"])/int(physical["n"])
    thickness = float(physical.get(
        "represented_thickness_m", 2.0*2.48e-10))
    grain_index = tuple(state.grain_ids).index(target_grain_id)
    response = grain_response(
        initial.supports, state.supports, spacing_m=spacing,
        represented_thickness_m=thickness)
    row, _ = checkpoint_snapshot(checkpoint)
    history = accepted_history(run, int(step))
    peak = max((abs(float(item["shear_stress_Pa"])) for item in history),
               default=abs(float(row["post_front_equilibrated_stress_Pa"])))
    stress = abs(float(row["post_front_equilibrated_stress_Pa"]))
    softening = nonnegative_softening(peak, stress)
    criteria = PhysicalASBCriteria()
    width = row["power_width_minor_m"]
    checks = {
        "power_participation": row["plastic_power"][
            "inverse_participation_fraction"]
            <= criteria.maximum_power_participation,
        "temperature_peak_minus_mean": row[
            "temperature_max_minus_mean_K"]
            >= criteria.minimum_temperature_peak_minus_mean_K,
        "post_peak_softening": softening >= criteria.minimum_softening_fraction,
        "resolved_width": bool(width is not None and width >=
            criteria.minimum_width_to_interface*row["interface_width_m"]),
        "narrow_width": bool(width is not None and width <=
            criteria.maximum_width_to_domain*row["domain_length_m"]),
        "elongated_morphology": bool(row["power_aspect_ratio"] is not None and
            row["power_aspect_ratio"] >= criteria.minimum_aspect_ratio),
        "heat_power_colocation": row["heat_power_component_overlap"] >=
            criteria.minimum_heat_power_overlap,
        "matched_temperature_control": False,
        "spatial_refinement": False,
    }
    return {
        "schema": "asb-drx-v65-live-checkpoint-classification-v1",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "adaptive_interval": int(step),
        "physical_time_s": runtime.ledger.physical_time_s,
        "applied_shear_strain": float(load),
        "source_commit": (provenance or {}).get("source_commit"),
        "grain_20": {
            "volume_growth_fraction": float(
                response["volume_growth_fraction"][grain_index]),
            "equivalent_contour_displacement_m": float(
                response["equivalent_contour_displacement_m"][grain_index]),
            "equivalent_contour_displacement_cells": float(
                response["equivalent_contour_displacement_m"][grain_index]
                /spacing),
            "fresh_sweep_volume_m3": float(
                state.ledger.fresh_sweep_fraction*spacing**2*thickness),
            "revisit_sweep_volume_m3": float(
                state.ledger.revisit_sweep_fraction*spacing**2*thickness),
        },
        "localization": {
            "stress_Pa": stress,
            "preceding_suffix_peak_stress_Pa": peak,
            "softening_fraction": softening,
            "temperature_mean_K": row["temperature_mean_K"],
            "temperature_peak_minus_mean_K": row[
                "temperature_max_minus_mean_K"],
            "temperature_contrast_K": row[
                "temperature_max_minus_min_K"],
            "power_participation": row["plastic_power"][
                "inverse_participation_fraction"],
            "power_aspect_ratio": row["power_aspect_ratio"],
            "power_width_minor_m": width,
            "heat_power_overlap": row["heat_power_component_overlap"],
            "checks": checks,
            "strict_snapshot": all(checks.values()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--origin", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--solver-pid", type=int)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--telemetry", type=Path, action="append", default=[])
    parser.add_argument("--target-time", type=float)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir/"live_checkpoint_monitor_state.json"
    last = -1
    if state_path.is_file():
        last = int(json.loads(state_path.read_text()).get(
            "last_classified_interval", -1))
    while True:
        checkpoints = sorted(
            args.run.glob("checkpoint_adaptive_*.npz"),
            key=checkpoint_index)
        pending = [path for path in checkpoints if checkpoint_index(path) > last]
        for checkpoint in pending:
            payload = classify(checkpoint, args.origin, args.run)
            if args.telemetry and args.target_time is not None:
                trend_rows, trend_sources = load_trend_rows(args.telemetry)
                payload["accepted_time_trend"] = summarize_trends(
                    trend_rows, trend_sources, window=64,
                    target_time_s=args.target_time, grain_index=1)
            with (args.output_dir/"live_checkpoint_classifications.jsonl").open(
                    "a") as stream:
                stream.write(json.dumps(payload, sort_keys=True)+"\n")
                stream.flush(); os.fsync(stream.fileno())
            atomic_json(args.output_dir/"live_checkpoint_latest.json", payload)
            last = payload["adaptive_interval"]
            atomic_json(state_path, {
                "schema": "asb-drx-v65-live-checkpoint-monitor-state-v1",
                "last_classified_interval": last,
                "last_checkpoint_sha256": payload["checkpoint_sha256"],
            })
            print(json.dumps({
                "adaptive_interval": last,
                "physical_time_s": payload["physical_time_s"],
                "grain_20_displacement_cells": payload[
                    "grain_20"]["equivalent_contour_displacement_cells"],
                "temperature_peak_minus_mean_K": payload[
                    "localization"]["temperature_peak_minus_mean_K"],
                "strict_snapshot": payload[
                    "localization"]["strict_snapshot"],
            }), flush=True)
        if args.once:
            break
        if args.solver_pid is not None:
            try:
                os.kill(args.solver_pid, 0)
            except ProcessLookupError:
                break
        time.sleep(max(float(args.poll_seconds), 1.0))


if __name__ == "__main__":
    main()
