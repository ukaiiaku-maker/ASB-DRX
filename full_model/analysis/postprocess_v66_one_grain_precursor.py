#!/usr/bin/env python3
"""Classify fresh one-grain common-state subgrain precursor formation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from full_model.analysis.run_v30_mura_tier_b1_case import (
    create_case, load_checkpoint,
)
from full_model.production.common_subgrain_recognition import (
    recognize_common_orientation_plateau,
)
from full_model.production.density_state_map import derived_density_fields
from full_model.production.tensorial_nye import spectral_derivatives


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _serial_recognition(result, state, spacing):
    serial = {key: value for key, value in result.items()
              if not isinstance(value, np.ndarray)}
    if "shell_mask" not in result:
        return serial
    shell = np.asarray(result["shell_mask"], dtype=bool)
    interior = np.asarray(result["interior_mask"], dtype=bool)
    alpha = np.sum(state.common.family_nye_m1, axis=2)
    gnd = np.linalg.norm(alpha, axis=(-2, -1))
    gx, gy = spectral_derivatives(state.common.orientation_rad, spacing)
    gradient = np.hypot(gx, gy)
    serial.update({
        "shell_gnd_rms_m1": float(np.sqrt(np.mean(gnd[shell]**2))),
        "interior_gnd_rms_m1": float(np.sqrt(np.mean(gnd[interior]**2))),
        "shell_orientation_gradient_rms_m1": float(np.sqrt(
            np.mean(gradient[shell]**2))),
        "interior_orientation_gradient_rms_m1": float(np.sqrt(
            np.mean(gradient[interior]**2))),
        "gnd_shell_concentration": bool(
            np.sqrt(np.mean(gnd[shell]**2))
            >1.5*max(np.sqrt(np.mean(gnd[interior]**2)), 1.0)),
        "orientation_gradient_shell_concentration": bool(
            np.sqrt(np.mean(gradient[shell]**2))
            >1.5*max(np.sqrt(np.mean(gradient[interior]**2)), 1.0)),
    })
    return serial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid", type=int, required=True)
    parser.add_argument("--length-m", type=float, default=1e-5)
    args = parser.parse_args()
    (_, _, _, systems, topologies, _, _, _, spacing) = create_case(
        args.grid, "mechanical_heterogeneity", 42, args.length_m)
    checkpoints = sorted(args.case_dir.glob("checkpoint_step_*.npz"))
    if not checkpoints:
        raise FileNotFoundError("no fresh precursor checkpoints found")
    records = []
    for path in checkpoints:
        state, metadata = load_checkpoint(path, systems, topologies)
        fields = derived_density_fields(state.density, topologies)
        total = fields["rho_total_m2"]
        order = np.asarray(state.common.wall_order)
        strict = recognize_common_orientation_plateau(
            state.common.orientation_rad, state.common.family_nye_m1,
            order, total, spacing)
        thresholds = {}
        for threshold in (.25, .5, 1.0, 2.0):
            result = recognize_common_orientation_plateau(
                state.common.orientation_rad, state.common.family_nye_m1,
                order, total, spacing,
                minimum_misorientation_deg=threshold,
                minimum_boundary_order=0.0,
                minimum_ordered_fraction=0.0)
            thresholds[f"{threshold:g}_deg"] = _serial_recognition(
                result, state, spacing)
        records.append({
            "checkpoint": str(path.relative_to(args.case_dir)),
            "sha256": _sha(path),
            "step": int(metadata["step"]),
            "physical_time_s": float(metadata["physical_time_s"]),
            "applied_strain": float(metadata["applied_strain"]),
            "orientation_span_deg": float(np.degrees(np.ptp(
                state.common.orientation_rad))),
            "strict_boundary": _serial_recognition(strict, state, spacing),
            "precursor_thresholds": thresholds,
        })
    final = records[-1]
    candidates = [row for row in final["precursor_thresholds"].values()
                  if row.get("gate_status", {}).get("geometry_resolved", False)]
    physical = [row for row in candidates
                if row.get("gate_status", {}).get("frank_bilby", False)
                and row.get("gnd_shell_concentration", False)
                and row.get("orientation_gradient_shell_concentration", False)]
    payload = {
        "schema": "v66-one-grain-structural-precursor-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": "96e75b4+precursor-analysis",
        "case_dir": str(args.case_dir),
        "grid": [args.grid, args.grid], "spacing_m": spacing,
        "records": records,
        "decision": {
            "strict_new_boundary_qualified": bool(
                final["strict_boundary"].get("qualified", False)),
            "gnd_bearing_closed_precursor_detected": bool(physical),
            "physical_grain_count_change": 0,
            "classification": (
                "GND_BEARING_STRUCTURAL_PRECURSOR"
                if physical else "NO_RESOLVED_GND_BEARING_PRECURSOR"),
            "claim_boundary": (
                "fresh current-source one-grain evolution; a precursor is not "
                "a promoted grain, sustained DRX, or material calibration"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload["decision"], indent=2))


if __name__ == "__main__":
    main()
