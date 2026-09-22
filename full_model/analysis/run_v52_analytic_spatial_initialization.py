#!/usr/bin/env python3
"""Create the same analytic bicrystal on a companion physical grid."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from full_model.analysis.run_v34_finite_coupled_response import resolved_bicrystal
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
        "companion_grid": int(args.grid),
        "companion_checkpoint": str(args.companion_checkpoint.resolve()),
        "companion_checkpoint_sha256": digest(args.companion_checkpoint),
        "physical_configuration": {
            "length_m": kwargs["length_m"],
            "interface_width_m": kwargs["interface_width_m"],
            "temperature_K": kwargs["temperature_K"],
            "child_line_fraction": kwargs["child_line_fraction"],
            "deterministic_physical_fields": True,
            "grid_index_noise_used": False,
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
