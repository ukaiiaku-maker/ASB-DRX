#!/usr/bin/env python3
"""Qualify V48 ordering on the exact retained pre-Mura trajectory fork."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from full_model.analysis.run_v34_finite_coupled_response import (
    I3Controls, resolved_bicrystal, run_i3_cycle,
)
from full_model.analysis.run_v36_recurrent_physical_response import driving_at_time
from full_model.analysis.run_v39_common_horizon import load_stage
from full_model.analysis.run_v47_ordering_finite_time import DT_S, pair_paths


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ordered(state):
    density = state.mechanical.density
    return {sign: np.asarray(getattr(
        density, f"wall_ordered_{sign}_m2")) for sign in ("plus", "minus")}


def run_backend(context, initial, driving, backend, rtol, atol):
    local = dict(context)
    local["extensive_parameters"] = replace(
        context["extensive_parameters"],
        ordering_finite_time_backend=backend,
        ordering_finite_relative_tolerance=rtol,
        ordering_finite_absolute_tolerance=atol,
        ordering_asymptotic_certificate_mode="disabled")
    started = time.perf_counter()
    result, audit = run_i3_cycle(
        local, initial, initial.eta.copy(), driving,
        I3Controls(
            mura_enabled=True, front_enabled=False, trial_dt_s=DT_S,
            front_dt_s=DT_S,
            mura_transport_operator="compatible_dealiased"))
    return result, audit, time.perf_counter()-started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v46-results", type=Path, required=True)
    parser.add_argument("--stage", choices=(
        "first_nonzero", "post_front", "late_macro"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rtol", type=float, default=2e-4)
    parser.add_argument("--atol", type=float, default=2e-8)
    parser.add_argument("--tightening-factor", type=float, default=4.0)
    parser.add_argument("--include-rk2-diagnostic", action="store_true")
    args = parser.parse_args()
    before_path, old_reference_path, driving_time = pair_paths(
        args.v46_results, args.stage)
    context = resolved_bicrystal(
        grid=128, length_m=3.2e-6, interface_width_m=4e-7,
        temperature_K=1100.0, child_line_fraction=.35)
    initial, metadata = load_stage(before_path, context)
    old_reference, old_metadata = load_stage(old_reference_path, context)
    driving = driving_at_time(128, .01, "hold", 0.0, driving_time)
    adaptive, adaptive_audit, adaptive_seconds = run_backend(
        context, initial, driving,
        "matrix_free_adaptive_rosenbrock_euler", args.rtol, args.atol)
    runs = {"adaptive_rosenbrock": {
        "wall_seconds": adaptive_seconds,
        "mura": adaptive_audit["mura"],
        "complete_energy": adaptive_audit["complete_energy"],
    }}
    tightened, tightened_audit, tightened_seconds = run_backend(
        context, initial, driving,
        "matrix_free_adaptive_rosenbrock_euler",
        args.rtol/args.tightening_factor,
        args.atol/args.tightening_factor)
    runs["adaptive_rosenbrock_tightened"] = {
        "wall_seconds": tightened_seconds,
        "relative_tolerance": args.rtol/args.tightening_factor,
        "absolute_tolerance": args.atol/args.tightening_factor,
        "mura": tightened_audit["mura"],
        "complete_energy": tightened_audit["complete_energy"],
    }
    rk2 = None
    if args.include_rk2_diagnostic:
        rk2, rk2_audit, rk2_seconds = run_backend(
            context, initial, driving, "matrix_free_projected_rk2",
            args.rtol, args.atol)
        runs["projected_rk2_diagnostic"] = {
            "wall_seconds": rk2_seconds,
            "mura": rk2_audit["mura"],
            "complete_energy": rk2_audit["complete_energy"],
        }
    initial_fields = ordered(initial)
    adaptive_fields = ordered(adaptive)
    tightened_fields = ordered(tightened)
    old_fields = ordered(old_reference)
    comparisons = {}
    for sign in ("plus", "minus"):
        scale = max(float(np.linalg.norm(adaptive_fields[sign])), 1e-300)
        row = {
            "adaptive_change_from_initial_relative_l2": float(np.linalg.norm(
                adaptive_fields[sign]-initial_fields[sign])/scale),
            "adaptive_vs_old_asymptotic_relative_l2": float(np.linalg.norm(
                adaptive_fields[sign]-old_fields[sign])/scale),
            "default_vs_tightened_relative_l2": float(np.linalg.norm(
                adaptive_fields[sign]-tightened_fields[sign])/max(
                    np.linalg.norm(tightened_fields[sign]), 1e-300)),
            "adaptive_inventory_m_per_m": float(np.sum(
                adaptive_fields[sign])*context["spacing_m"]**2),
            "initial_inventory_m_per_m": float(np.sum(
                initial_fields[sign])*context["spacing_m"]**2),
            "old_asymptotic_inventory_m_per_m": float(np.sum(
                old_fields[sign])*context["spacing_m"]**2),
        }
        if rk2 is not None:
            rk2_fields = ordered(rk2)
            row["rk2_change_from_initial_relative_l2"] = float(np.linalg.norm(
                rk2_fields[sign]-initial_fields[sign])/scale)
            row["adaptive_vs_rk2_relative_l2"] = float(np.linalg.norm(
                adaptive_fields[sign]-rk2_fields[sign])/scale)
        comparisons[sign] = row
    pre_hashes = [row["mura"]["pre_ordering_density_sha256"]
                  for row in runs.values()]
    tightening_difference = max(
        row["default_vs_tightened_relative_l2"]
        for row in comparisons.values())
    payload = {
        "schema": "asb-drx/v48/actual-ordering-reference/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "stage": args.stage,
        "before_checkpoint": str(before_path.resolve()),
        "before_checkpoint_sha256": digest(before_path),
        "old_reference_checkpoint": str(old_reference_path.resolve()),
        "old_reference_checkpoint_sha256": digest(old_reference_path),
        "before_physical_time_s": metadata["physical_time_s"],
        "old_reference_physical_time_s": old_metadata["physical_time_s"],
        "requested_interval_s": DT_S,
        "same_pre_ordering_fork_verified": len(set(pre_hashes)) == 1,
        "pre_ordering_density_sha256": pre_hashes[0],
        "runs": runs,
        "comparisons": comparisons,
        "tightening_factor": args.tightening_factor,
        "maximum_default_vs_tightened_relative_l2": tightening_difference,
        "declared_endpoint_tolerance_relative": .01,
        "classification": (
            "FINITE_REFERENCE_ERROR_CONTROLLED"
            if (adaptive_audit["mura"][
                    "ordering_finite_time_error_tolerance_satisfied"]
                and tightened_audit["mura"][
                    "ordering_finite_time_error_tolerance_satisfied"]
                and tightening_difference <= .01)
            else "FINITE_REFERENCE_UNQUALIFIED"),
        "old_asymptotic_promoted_as_reference": False,
        "projected_rk2_promoted_as_reference": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps({
        "classification": payload["classification"],
        "same_pre_ordering_fork_verified": payload[
            "same_pre_ordering_fork_verified"],
        "wall_seconds": adaptive_seconds,
        "sha256": digest(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
