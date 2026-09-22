#!/usr/bin/env python3
"""Create the same analytic bicrystal on a companion physical grid."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from full_model.analysis.run_v34_finite_coupled_response import (
    checkpoint_payload, resolved_bicrystal,
)
from full_model.analysis.run_v39_common_horizon import load_stage, save_stage


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retained-n128", type=Path, required=True)
    parser.add_argument("--companion-checkpoint", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=192)
    parser.add_argument("--dt-s", type=float, default=4.8828125e-7)
    args = parser.parse_args()
    kwargs = dict(length_m=3.2e-6, interface_width_m=4e-7,
                  temperature_K=1100.0, child_line_fraction=.35)
    reference_context = resolved_bicrystal(grid=128, **kwargs)
    retained, retained_metadata = load_stage(
        args.retained_n128, reference_context)
    analytic_reference = reference_context["state"]
    checks = {
        "eta_maximum_absolute": float(np.max(np.abs(
            retained.eta-analytic_reference.eta))),
        "beta_p_maximum_absolute": float(np.max(np.abs(
            retained.mechanical.common.beta_p
            -analytic_reference.mechanical.common.beta_p))),
        "temperature_maximum_absolute_K": float(np.max(np.abs(
            retained.mechanical.common.temperature_K
            -analytic_reference.mechanical.common.temperature_K))),
        "orientation_maximum_absolute_rad": float(np.max(np.abs(
            retained.mechanical.common.orientation_rad
            -analytic_reference.mechanical.common.orientation_rad))),
        "mobile_plus_maximum_absolute_m2": float(np.max(np.abs(
            retained.mechanical.density.mobile_plus_m2
            -analytic_reference.mechanical.density.mobile_plus_m2))),
    }
    if any(value != 0.0 for value in checks.values()):
        raise RuntimeError("retained n128 checkpoint is not the analytic initializer")
    # Audit the complete restart state, not a hand-selected field subset.  This
    # covers every scalar reservoir, alignment moment, parent/child/wake owner,
    # phase field, sparse-front owner, runtime/history array, and its metadata.
    retained_payload = checkpoint_payload(retained, reference_context)
    analytic_payload = checkpoint_payload(analytic_reference, reference_context)
    payload_identity = {}
    for name in sorted(set(retained_payload) | set(analytic_payload)):
        if name not in retained_payload or name not in analytic_payload:
            payload_identity[name] = {"present_in_both": False, "exact": False}
            continue
        left = np.asarray(retained_payload[name])
        right = np.asarray(analytic_payload[name])
        exact = bool(left.shape == right.shape and np.array_equal(left, right))
        row = {"present_in_both": True, "shape": list(left.shape),
               "dtype": str(left.dtype), "exact": exact}
        if (left.shape == right.shape and np.issubdtype(left.dtype, np.number)
                and np.issubdtype(right.dtype, np.number)):
            row["maximum_absolute_difference"] = float(
                np.max(np.abs(left-right))) if left.size else 0.0
        payload_identity[name] = row
    complete_payload_exact = bool(all(
        row["exact"] for row in payload_identity.values()))
    metadata_identity = {
        "completed_intervals_zero": retained_metadata.get(
            "completed_intervals") == 0,
        "physical_time_zero": retained_metadata.get("physical_time_s") == 0.0,
        "grid_128": retained_metadata.get("grid") == 128,
        "macro_dt_exact": retained_metadata.get("macro_dt_s") == args.dt_s,
        "records_empty_or_absent": not retained_metadata.get("records", []),
    }
    if not complete_payload_exact or not all(metadata_identity.values()):
        raise RuntimeError("retained n128 complete restart state is not the analytic initializer")
    companion_context = resolved_bicrystal(grid=args.grid, **kwargs)
    metadata = {
        "schema": "asb-drx/v52/analytic-common-physical-initial/v1",
        "stage": "V52_ANALYTIC_COMMON_PHYSICAL_INITIAL",
        "grid": int(args.grid),
        "macro_dt_s": float(args.dt_s),
        "completed_intervals": 0,
        "physical_time_s": 0.0,
        "source_role": (
            "direct analytic evaluation of the retained deterministic "
            "bicrystal on a companion grid"),
    }
    args.companion_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_stage(args.companion_checkpoint, companion_context["state"],
               companion_context, metadata)
    payload = {
        "schema": "asb-drx/v52/analytic-spatial-initialization-audit/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "route": "A_COMMON_ANALYTIC_PHYSICAL_INITIALIZATION",
        "retained_n128_checkpoint": str(args.retained_n128.resolve()),
        "retained_n128_sha256": digest(args.retained_n128),
        "retained_metadata": retained_metadata,
        "n128_initializer_identity": checks,
        "n128_complete_restart_payload_identity": {
            "all_arrays_and_owner_metadata_exact": complete_payload_exact,
            "field_count": len(payload_identity),
            "fields": payload_identity,
        },
        "n128_stage_metadata_identity": metadata_identity,
        "companion_grid": int(args.grid),
        "companion_checkpoint": str(args.companion_checkpoint.resolve()),
        "companion_checkpoint_sha256": digest(args.companion_checkpoint),
        "physical_configuration": {
            "length_m": kwargs["length_m"],
            "interface_width_m": kwargs["interface_width_m"],
            "spacing_m": companion_context["spacing_m"],
            "represented_thickness_m": companion_context[
                "represented_thickness_m"],
            "continuum_representation_length_m": companion_context[
                "wall_parameters"].capture_deposition_length_m,
            "temperature_K": kwargs["temperature_K"],
            "child_line_fraction": kwargs["child_line_fraction"],
            "deterministic_physical_fields": True,
            "grid_index_noise_used": False,
        },
        "constitutive_and_thermal_parameters": {
            "wall": asdict(companion_context["wall_parameters"]),
            "extensive_ordering": asdict(
                companion_context["extensive_parameters"]),
            "topology_kinetics": asdict(
                companion_context["topology_kinetics"]),
        },
        "scope": (
            "same analytic history origin; no evolved-state interpolation "
            "or remeshing event"),
    }
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps({
        "checkpoint_sha256": payload["companion_checkpoint_sha256"],
        "evidence_sha256": digest(args.evidence),
        "n128_initializer_exact": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
