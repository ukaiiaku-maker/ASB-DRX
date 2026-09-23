#!/usr/bin/env python3
"""Local temporal sign audit at the retained V52 near-flow endpoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from full_model.analysis.run_v34_finite_coupled_response import resolved_bicrystal
from full_model.analysis.run_v39_common_horizon import (
    atomic_json, load_stage, save_stage,
)
from full_model.analysis.run_v48_physical_continuation import observables
from full_model.analysis.run_v49_physical_continuation import (
    advance_consistent_midpoint_segment, driving,
)


SCHEMA = "asb-drx/v53/near-flow-temporal/v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative_l2(left, right):
    left = np.asarray(left); right = np.asarray(right)
    return float(np.linalg.norm(left-right)/max(np.linalg.norm(right), 1e-300))


def branch_metadata(parent, label, duration, work, residuals):
    return {
        **parent,
        "stage": "V53_NEAR_FLOW_TEMPORAL_FORK",
        "fork_branch": label,
        "physical_time_s": float(parent["physical_time_s"]+duration),
        "cumulative_external_work_J": float(
            parent["cumulative_external_work_J"]+work),
        "fork_external_work_J": float(work),
        "fork_first_law_residuals_J": [float(value) for value in residuals],
        "fork_is_not_a_published_continuation_prefix": True,
    }


def run_segments(context, initial, metadata, count, macro_dt):
    state = initial; time_s = float(metadata["physical_time_s"])
    work = 0.0; residuals = []; rows = []
    duration = float(macro_dt)/int(count)
    for index in range(int(count)):
        result = advance_consistent_midpoint_segment(
            context, state, grid=int(metadata["grid"]),
            initial_tensor_shear=float(metadata["initial_tensor_shear"]),
            protocol=metadata["protocol"], rate=float(metadata["strain_rate_s"]),
            physical_time=time_s,
            load_origin=float(metadata["load_origin_time_s"]),
            requested_duration=duration)
        state = result["candidate"]
        time_s += result["accepted_duration_s"]
        work += result["external_work_J"]
        residuals.append(result["first_law_residual_J"])
        rows.append({
            "substep": index+1,
            "requested_duration_s": duration,
            "accepted_duration_s": result["accepted_duration_s"],
            "physical_time_end_s": time_s,
            "midpoint_mean_strain": np.asarray(
                result["midpoint_drive"].mean_strain).tolist(),
            "external_work_J": result["external_work_J"],
            "first_law_residual_J": result["first_law_residual_J"],
            "first_law_passed": result["first_law_passed"],
            "discarded_shortened_candidates": result[
                "discarded_shortened_candidates"],
        })
    return state, work, residuals, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dt-s", type=float, default=4.8828125e-7)
    args = parser.parse_args()
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    result_path = output/"near_flow_temporal.json"
    expected_parent = digest(args.checkpoint)
    context = resolved_bicrystal(
        grid=128, length_m=3.2e-6, interface_width_m=4e-7,
        temperature_K=1100.0, child_line_fraction=.35)
    initial, metadata = load_stage(args.checkpoint, context)
    if int(metadata["completed_intervals"]) != 104:
        raise ValueError("V53 temporal audit requires retained interval 104")
    start_time = float(metadata["physical_time_s"])
    endpoint_drive = driving(
        128, float(metadata["initial_tensor_shear"]), metadata["protocol"],
        float(metadata["strain_rate_s"]),
        start_time+args.dt_s-float(metadata["load_origin_time_s"]))
    start_drive = driving(
        128, float(metadata["initial_tensor_shear"]), metadata["protocol"],
        float(metadata["strain_rate_s"]),
        start_time-float(metadata["load_origin_time_s"]))
    start_obs = observables(context, initial, start_drive)

    branches = {}
    states = {}
    for label, count in (("full", 1), ("two_half", 2)):
        checkpoint = output/f"{label}.npz"
        journal = output/f"{label}.json"
        if checkpoint.exists() and journal.exists():
            saved = json.loads(journal.read_text())
            if (saved.get("parent_checkpoint_sha256") != expected_parent
                    or saved.get("checkpoint_sha256") != digest(checkpoint)):
                raise RuntimeError(f"stale or corrupt saved {label} branch")
            state, saved_metadata = load_stage(checkpoint, context)
            branch = saved
        else:
            started = time.perf_counter()
            state, work, residuals, rows = run_segments(
                context, initial, metadata, count, args.dt_s)
            save_stage(checkpoint, state, context, branch_metadata(
                metadata, label, args.dt_s, work, residuals))
            branch = {
                "branch": label,
                "substeps": count,
                "rows": rows,
                "wall_seconds": time.perf_counter()-started,
                "external_work_J": work,
                "summed_first_law_residual_J": float(sum(residuals)),
                "all_first_law_checks_passed": bool(all(
                    row["first_law_passed"] for row in rows)),
                "parent_checkpoint_sha256": expected_parent,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": digest(checkpoint),
            }
            atomic_json(journal, branch)
        states[label] = state
        branches[label] = branch

    observations = {
        name: observables(context, state, endpoint_drive)
        for name, state in states.items()}
    increments = {name: {
        key: float(value[key]-start_obs[key]) for key in value}
        for name, value in observations.items()}
    stress_key = "mean_shear_stress_sigma_12_Pa"
    plastic_key = "engineering_plastic_shear_gamma_p"
    full_stress = increments["full"][stress_key]
    half_stress = increments["two_half"][stress_key]
    stress_difference = full_stress-half_stress
    imposed_engineering_rate = 2.0*float(metadata["strain_rate_s"])
    rates = {name: row[plastic_key]/args.dt_s for name, row in increments.items()}
    deficits = {name: imposed_engineering_rate-value
                for name, value in rates.items()}
    sign_resolved = bool(
        np.sign(full_stress) == np.sign(half_stress)
        and min(abs(full_stress), abs(half_stress)) > abs(stress_difference))
    comparison = {
        "stress_increment_full_Pa": full_stress,
        "stress_increment_two_half_Pa": half_stress,
        "absolute_stress_increment_difference_Pa": abs(stress_difference),
        "plastic_rate_s-1": rates,
        "rate_deficit_from_imposed_s-1": deficits,
        "absolute_rate_deficit_difference_s-1": abs(
            deficits["full"]-deficits["two_half"]),
        "stress_increment_sign_resolved_by_subdivision": sign_resolved,
        "sign_margin_definition": (
            "same sign and smaller absolute increment exceeds the full-minus-split difference"),
        "eta_relative_l2": relative_l2(states["full"].eta,
                                         states["two_half"].eta),
        "beta_p_relative_l2": relative_l2(
            states["full"].mechanical.common.beta_p,
            states["two_half"].mechanical.common.beta_p),
        "temperature_rise_increment_relative_difference": abs(
            increments["full"]["temperature_mean_K"]
            -increments["two_half"]["temperature_mean_K"])/max(
                abs(increments["two_half"]["temperature_mean_K"]), 1e-300),
        "family_nye_relative_l2": relative_l2(
            states["full"].mechanical.common.family_nye_m1,
            states["two_half"].mechanical.common.family_nye_m1),
    }
    payload = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": expected_parent,
        "physical_time_start_s": start_time,
        "physical_time_end_s": start_time+args.dt_s,
        "macro_duration_s": args.dt_s,
        "actual_midpoint_loads_retained": True,
        "cumulative_baseline_retained": True,
        "branches": branches,
        "start_observables": start_obs,
        "endpoint_observables": observations,
        "observable_increments": increments,
        "comparison": comparison,
        "quarter_step_branch_required": not sign_resolved,
        "accepted_state_invariants_passed": bool(all(
            row["all_first_law_checks_passed"] for row in branches.values())),
        "drx_claimed": False,
        "persistent_lagb_claimed": False,
        "strict_asb_claimed": False,
        "material_calibration_claimed": False,
    }
    atomic_json(result_path, payload)
    print(json.dumps({"result": str(result_path),
                      "sha256": digest(result_path), **comparison},
                     sort_keys=True))


if __name__ == "__main__":
    main()
