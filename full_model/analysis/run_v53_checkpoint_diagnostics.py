#!/usr/bin/env python3
"""Read-only structural and kinetic diagnostics from accepted checkpoints."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from full_model.analysis.run_v34_finite_coupled_response import resolved_bicrystal
from full_model.analysis.run_v39_common_horizon import atomic_json, load_stage
from full_model.analysis.run_v49_physical_continuation import driving
from full_model.production.common_tensorial_wall import resolved_driving_components


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def statistics(value):
    array = np.asarray(value, dtype=float)
    return {
        "minimum": float(np.min(array)),
        "mean": float(np.mean(array)),
        "maximum": float(np.max(array)),
        "rms": float(np.sqrt(np.mean(array*array))),
    }


def diagnose(label, checkpoint):
    with np.load(checkpoint, allow_pickle=False) as archive:
        grid = int(np.asarray(archive["eta"]).shape[0])
    context = resolved_bicrystal(
        grid=grid, length_m=3.2e-6, interface_width_m=4e-7,
        temperature_K=1100.0, child_line_fraction=.35)
    state, metadata = load_stage(checkpoint, context)
    mechanical = state.mechanical; common = mechanical.common
    endpoint_drive = driving(
        grid, float(metadata["initial_tensor_shear"]), metadata["protocol"],
        float(metadata["strain_rate_s"]),
        float(metadata["physical_time_s"])-float(metadata["load_origin_time_s"]))
    fields = resolved_driving_components(
        common, endpoint_drive, context["systems"], context["topologies"],
        context["wall_parameters"])
    spacing = float(context["spacing_m"])
    thickness = float(context["represented_thickness_m"])
    cell_volume = spacing*spacing*thickness
    density = mechanical.density
    reservoir_line_m = {
        name: np.sum(np.asarray(getattr(density, name)), axis=(0, 1),
                     dtype=np.longdouble).astype(float).tolist()
        if np.asarray(getattr(density, name)).ndim == 3 else
        float(np.sum(getattr(density, name), dtype=np.longdouble))
        for name in density.__dataclass_fields__}
    reservoir_line_m = {
        name: (float(value)*cell_volume if np.isscalar(value)
               else [float(item)*cell_volume for item in value])
        for name, value in reservoir_line_m.items()}
    signed_family = np.zeros(len(context["systems"]))
    total_family = np.zeros(len(context["systems"]))
    for stem in ("mobile", "forest", "wall_tangle", "wall_ordered"):
        plus = np.asarray(getattr(density, stem+"_plus_m2"))
        minus = np.asarray(getattr(density, stem+"_minus_m2"))
        signed_family += np.sum(plus-minus, axis=(0, 1))*cell_volume
        total_family += np.sum(plus+minus, axis=(0, 1))*cell_volume
    mobile = common.mobile_plus_m2+common.mobile_minus_m2
    speed = np.asarray(fields["speed_m_s"])
    activity = mobile*np.abs(speed)
    orientation = np.asarray(common.orientation_rad)
    family_nye = np.asarray(common.family_nye_m1)
    endpoint_records = [row for row in metadata.get("records", [])
                        if int(row["interval"]) == int(
                            metadata["completed_intervals"])]
    return {
        "label": label,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": digest(checkpoint),
        "grid": grid,
        "completed_intervals": int(metadata["completed_intervals"]),
        "physical_time_s": float(metadata["physical_time_s"]),
        "load_elapsed_time_s": float(metadata["physical_time_s"]
                                     -metadata["load_origin_time_s"]),
        "endpoint_record": endpoint_records[0] if len(endpoint_records) == 1 else None,
        "reservoir_line_length_m": reservoir_line_m,
        "signed_family_line_length_m": signed_family.tolist(),
        "total_family_line_length_m": total_family.tolist(),
        "mobile_line_length_m": float(np.sum(mobile)*cell_volume),
        "slip_by_family_area_integral_m2": (
            np.sum(common.slip, axis=(0, 1))*spacing**2).tolist(),
        "raw_resolved_stress_Pa": statistics(fields["raw_stress_Pa"]),
        "taylor_resistance_Pa": statistics(fields["taylor_resistance_Pa"]),
        "resistance_reduced_effective_stress_Pa": statistics(
            fields["effective_stress_Pa"]),
        "actual_glide_speed_m_s": statistics(speed),
        "mobile_speed_activity_integral_m2_s-1": float(
            np.sum(activity, dtype=np.longdouble)*cell_volume),
        "wall_order": statistics(common.wall_order),
        "compatible_family_nye_m-1": {
            "rms_by_family": np.sqrt(np.mean(
                family_nye*family_nye, axis=(0, 1, 3, 4))).tolist(),
            "maximum_abs_by_family": np.max(
                np.abs(family_nye), axis=(0, 1, 3, 4)).tolist(),
        },
        "orientation_rad": {
            **statistics(orientation),
            "p01": float(np.quantile(orientation, .01)),
            "p99": float(np.quantile(orientation, .99)),
            "robust_p99_minus_p01": float(
                np.quantile(orientation, .99)-np.quantile(orientation, .01)),
        },
        "retained_field_availability": {
            "instantaneous_plastic_power_field": False,
            "instantaneous_irreversible_heat_rate_field": False,
            "event_availability_field": False,
            "limiter_activity_field": False,
            "endpoint_state_counterfactual_kinetic_evaluation": True,
        },
        "diagnostic_scope": (
            "stress, velocity and activity are constitutive evaluations on the saved endpoint state; "
            "they are not independently evolved causal trajectories or missing retained power fields"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="LABEL=PATH; repeat for each accepted state")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for specification in args.checkpoint:
        label, separator, raw = specification.partition("=")
        if not separator or not label or not raw:
            raise ValueError("checkpoint arguments require LABEL=PATH")
        rows.append(diagnose(label, Path(raw)))
    payload = {
        "schema": "asb-drx/v53/checkpoint-diagnostics/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "read_only": True,
        "states": rows,
        "instantaneous_power_or_heat_claimed_when_unavailable": False,
        "drx_claimed": False,
        "persistent_lagb_claimed": False,
        "strict_asb_claimed": False,
        "material_calibration_claimed": False,
    }
    atomic_json(args.output, payload)
    print(json.dumps({"output": str(args.output), "sha256": digest(args.output),
                      "states": [row["label"] for row in rows]}, sort_keys=True))


if __name__ == "__main__":
    main()
