#!/usr/bin/env python3
"""Conditional four-quarter near-flow branch required by the V53 sign audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

from full_model.analysis.run_v34_finite_coupled_response import resolved_bicrystal
from full_model.analysis.run_v39_common_horizon import atomic_json, load_stage, save_stage
from full_model.analysis.run_v48_physical_continuation import observables
from full_model.analysis.run_v49_physical_continuation import driving
from full_model.analysis.run_v53_near_flow_temporal import (
    branch_metadata, digest, run_segments,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prior-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dt-s", type=float, default=4.8828125e-7)
    args = parser.parse_args()
    prior = json.loads(args.prior_result.read_text())
    if not prior["quarter_step_branch_required"]:
        raise RuntimeError("prior temporal result does not require a quarter branch")
    if prior["source_checkpoint_sha256"] != digest(args.checkpoint):
        raise RuntimeError("quarter branch parent differs from prior temporal fork")
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    context = resolved_bicrystal(
        grid=128, length_m=3.2e-6, interface_width_m=4e-7,
        temperature_K=1100.0, child_line_fraction=.35)
    initial, metadata = load_stage(args.checkpoint, context)
    started = time.perf_counter()
    state, work, residuals, rows = run_segments(
        context, initial, metadata, 4, args.dt_s)
    checkpoint = output/"four_quarter.npz"
    save_stage(checkpoint, state, context, branch_metadata(
        metadata, "four_quarter", args.dt_s, work, residuals))
    end_drive = driving(
        128, float(metadata["initial_tensor_shear"]), metadata["protocol"],
        float(metadata["strain_rate_s"]),
        float(metadata["physical_time_s"])+args.dt_s
        -float(metadata["load_origin_time_s"]))
    start_drive = driving(
        128, float(metadata["initial_tensor_shear"]), metadata["protocol"],
        float(metadata["strain_rate_s"]),
        float(metadata["physical_time_s"])-float(metadata["load_origin_time_s"]))
    start = observables(context, initial, start_drive)
    endpoint = observables(context, state, end_drive)
    increments = {key: float(endpoint[key]-start[key]) for key in endpoint}
    stress_key = "mean_shear_stress_sigma_12_Pa"
    plastic_key = "engineering_plastic_shear_gamma_p"
    half_stress = prior["comparison"]["stress_increment_two_half_Pa"]
    quarter_stress = increments[stress_key]
    difference = quarter_stress-half_stress
    half_deficit = prior["comparison"]["rate_deficit_from_imposed_s-1"][
        "two_half"]
    quarter_rate = increments[plastic_key]/args.dt_s
    quarter_deficit = 2.0*float(metadata["strain_rate_s"])-quarter_rate
    resolved = bool(
        (quarter_stress > 0.0) == (half_stress > 0.0)
        and min(abs(quarter_stress), abs(half_stress)) > abs(difference))
    payload = {
        "schema": "asb-drx/v53/near-flow-quarter/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": digest(args.checkpoint),
        "prior_result": str(args.prior_result.resolve()),
        "prior_result_sha256": digest(args.prior_result),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest(checkpoint),
        "wall_seconds": time.perf_counter()-started,
        "rows": rows,
        "summed_external_work_J": work,
        "summed_first_law_residual_J": float(sum(residuals)),
        "all_first_law_checks_passed": bool(all(
            row["first_law_passed"] for row in rows)),
        "endpoint_observables": endpoint,
        "observable_increments": increments,
        "two_half_vs_four_quarter": {
            "stress_increment_two_half_Pa": half_stress,
            "stress_increment_four_quarter_Pa": quarter_stress,
            "absolute_stress_increment_difference_Pa": abs(difference),
            "rate_deficit_two_half_s-1": half_deficit,
            "rate_deficit_four_quarter_s-1": quarter_deficit,
            "absolute_rate_deficit_difference_s-1": abs(
                half_deficit-quarter_deficit),
            "stress_increment_sign_resolved_by_subdivision": resolved,
            "sign_margin_definition": (
                "same sign and smaller absolute increment exceeds the two-half-minus-quarter difference"),
        },
        "drx_claimed": False,
        "persistent_lagb_claimed": False,
        "strict_asb_claimed": False,
        "material_calibration_claimed": False,
    }
    result = output/"near_flow_quarter.json"
    atomic_json(result, payload)
    print(json.dumps({"result": str(result), "sha256": digest(result),
                      **payload["two_half_vs_four_quarter"]}, sort_keys=True))


if __name__ == "__main__":
    main()
