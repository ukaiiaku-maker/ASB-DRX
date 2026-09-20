#!/usr/bin/env python3
"""Finite-rate checks for nonzero ordering states retained from V46."""

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


DT_S = 4.8828125e-7


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_scalar_counterexample():
    q0 = .25; equilibrium = .5; c = 1.0; exposure = .5
    exact = equilibrium+np.arcsinh(
        np.sinh(c*(q0-equilibrium))*np.exp(-c*exposure))/c
    return {
        "q0": q0, "equilibrium_q": equilibrium, "c": c,
        "attempt_exposure": exposure,
        "accessibility_passed": abs(equilibrium-q0) <= exposure,
        "equilibrium_kkt_residual": 0.0,
        "exact_finite_time_q": float(exact),
        "equilibrium_absolute_error": abs(equilibrium-float(exact)),
    }


def pair_paths(root, stage):
    first = root/"v46-checkpointed"/"n128"
    long = root/"v46-original-horizon"/"n128"
    if stage == "first_nonzero":
        return (first/"operation_001_mura.npz",
                first/"operation_002_mura.npz", 1.5*DT_S)
    if stage == "post_front":
        return (first/"operation_009_front.npz",
                first/"operation_010_mura.npz", 8.5*DT_S)
    if stage == "late_macro":
        return (long/"continuation_118_mura.npz",
                long/"continuation_119_mura.npz", 6.2255859375e-5)
    raise ValueError(stage)


def ordered_arrays(state):
    density = state.mechanical.density
    return {sign: np.asarray(getattr(
        density, f"wall_ordered_{sign}_m2"))
        for sign in ("plus", "minus")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v46-results", type=Path, required=True)
    parser.add_argument("--stage", choices=(
        "first_nonzero", "post_front", "late_macro"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-attempt-exposure", type=float, default=.05)
    parser.add_argument("--production-dispatch", action="store_true")
    args = parser.parse_args()
    before_path, reference_path, driving_time = pair_paths(
        args.v46_results, args.stage)
    context = resolved_bicrystal(
        grid=128, length_m=3.2e-6, interface_width_m=4e-7,
        temperature_K=1100.0, child_line_fraction=.35)
    before, before_metadata = load_stage(before_path, context)
    reference, reference_metadata = load_stage(reference_path, context)
    finite_context = dict(context)
    if args.production_dispatch:
        finite_context["extensive_parameters"] = replace(
            context["extensive_parameters"],
            ordering_finite_time_backend="matrix_free_projected_rk2",
            ordering_matrix_free_max_attempt_exposure=(
                args.max_attempt_exposure))
    else:
        finite_context["extensive_parameters"] = replace(
            context["extensive_parameters"],
            ordering_finite_time_backend="matrix_free_projected_rk2",
            ordering_matrix_free_max_attempt_exposure=args.max_attempt_exposure,
            ordering_asymptotic_minimum_attempt_exposure=1e300)
    driving = driving_at_time(128, .01, "hold", 0.0, driving_time)
    started = time.perf_counter()
    finite, audit = run_i3_cycle(
        finite_context, before, before.eta.copy(), driving,
        I3Controls(
            mura_enabled=True, front_enabled=False, trial_dt_s=DT_S,
            front_dt_s=DT_S,
            mura_transport_operator="compatible_dealiased"))
    wall_seconds = time.perf_counter()-started
    finite_arrays = ordered_arrays(finite)
    reference_arrays = ordered_arrays(reference)
    before_arrays = ordered_arrays(before)
    comparisons = {}
    for sign in ("plus", "minus"):
        scale = max(float(np.linalg.norm(reference_arrays[sign])), 1e-300)
        comparisons[sign] = {
            "finite_vs_asymptotic_relative_l2": float(np.linalg.norm(
                finite_arrays[sign]-reference_arrays[sign])/scale),
            "finite_vs_asymptotic_max_abs_m2": float(np.max(np.abs(
                finite_arrays[sign]-reference_arrays[sign]))),
            "finite_inventory_m_per_m": float(np.sum(
                finite_arrays[sign])*context["spacing_m"]**2),
            "asymptotic_inventory_m_per_m": float(np.sum(
                reference_arrays[sign])*context["spacing_m"]**2),
            "initial_inventory_m_per_m": float(np.sum(
                before_arrays[sign])*context["spacing_m"]**2),
        }
    ordering = audit["mura"]
    payload = {
        "schema": "asb-drx/v47/actual-ordering-finite-time/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "retained_state_stage": args.stage,
        "before_checkpoint": str(before_path.resolve()),
        "before_checkpoint_sha256": digest(before_path),
        "reference_checkpoint": str(reference_path.resolve()),
        "reference_checkpoint_sha256": digest(reference_path),
        "before_physical_time_s": before_metadata["physical_time_s"],
        "reference_physical_time_s": reference_metadata["physical_time_s"],
        "driving_time_s": driving_time,
        "requested_and_accepted_dt_s": DT_S,
        "production_dispatch_enabled": args.production_dispatch,
        "finite_backend": "matrix_free_projected_rk2",
        "maximum_internal_attempt_exposure": args.max_attempt_exposure,
        "wall_seconds": wall_seconds,
        "ordering_ledger": {
            key: ordering.get(key) for key in (
                "ordering_integration_method", "ordering_stiff_dispatch",
                "ordering_complete_elapsed_time_s",
                "ordering_discarded_reaction_time_s",
                "ordering_maximum_attempt_exposure",
                "ordering_finite_time_kinetic_accuracy_certified_by_this_solve",
                "ordering_active_degrees_of_freedom",
                "ordering_asymptotic_endpoint_distance_relative",
                "ordering_asymptotic_endpoint_distance_tolerance_relative",
                "ordering_asymptotic_endpoint_distance_bound_semantics")},
        "comparison_to_retained_asymptotic_endpoint": comparisons,
        "exact_scalar_counterexample": exact_scalar_counterexample(),
    }
    maximum = max(row["finite_vs_asymptotic_relative_l2"]
                  for row in comparisons.values())
    payload["classification"] = (
        "ACTUAL_STATE_ASYMPTOTIC_FINITE_TIME_AGREE_5PCT"
        if maximum <= .05 else
        "ACTUAL_STATE_ASYMPTOTIC_FINITE_TIME_DISAGREE")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps({
        "stage": args.stage, "classification": payload["classification"],
        "maximum_relative_l2": maximum, "wall_seconds": wall_seconds,
        "sha256": digest(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
