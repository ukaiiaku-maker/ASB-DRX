#!/usr/bin/env python3
"""Current-source n128/n192/n256 one-interval fixed-400-nm refinement."""

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
from full_model.analysis.run_v39_common_horizon import save_stage, stage_diagnostics
from full_model.production.density_state_map import derived_density_fields
from full_model.production.tensorial_nye import nye_from_plastic_distortion


DT_S = 4.8828125e-7


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spectrum(value):
    n = value.shape[0]
    return np.fft.fftshift(np.fft.fftn(value, axes=(0, 1))/(n*n), axes=(0, 1))


def crop(value, half):
    n = value.shape[0]; center = n//2
    return spectrum(value)[center-half:center+half+1,
                           center-half:center+half+1]


def relative(a, b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(a), np.linalg.norm(b), 1e-300))


def fields(state, spacing, topologies):
    density = derived_density_fields(state.mechanical.density, topologies)
    return {
        "curl_nye": nye_from_plastic_distortion(
            state.mechanical.common.beta_p, spacing),
        "ordered_density": density["rho_wall_ordered_m2"],
        "total_density": density["rho_total_m2"],
        "plastic_distortion": state.mechanical.common.beta_p,
        "temperature_rise": state.mechanical.common.temperature_K-1100.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    runs = {}; retained = {}
    for grid in (128, 192, 256):
        context = resolved_bicrystal(
            grid=grid, length_m=3.2e-6, interface_width_m=4e-7,
            temperature_K=1100.0, child_line_fraction=.35)
        context["extensive_parameters"] = replace(
            context["extensive_parameters"],
            ordering_finite_time_backend="matrix_free_projected_rk2",
            ordering_matrix_free_max_attempt_exposure=.05,
            ordering_asymptotic_minimum_attempt_exposure=1e300)
        driving = driving_at_time(grid, .01, "hold", 0.0, .5*DT_S)
        started = time.perf_counter()
        state, audit = run_i3_cycle(
            context, context["state"], context["state"].eta.copy(), driving,
            I3Controls(mura_enabled=True, front_enabled=False,
                       trial_dt_s=DT_S, front_dt_s=DT_S,
                       mura_transport_operator="compatible_dealiased"))
        elapsed = time.perf_counter()-started
        checkpoint = args.output/f"n{grid}_one_interval.npz"
        save_stage(checkpoint, state, context, {
            "source_sha": source, "stage": "v47_one_finite_mura",
            "grid": grid, "physical_time_s": DT_S,
            "completed_operation_count": 1, "records": []})
        mura = audit["mura"]
        runs[str(grid)] = {
            "spacing_m": context["spacing_m"],
            "representation_length_m": 4e-7,
            "representation_cells": 4e-7/context["spacing_m"],
            "accepted_dt_s": float(mura["accepted_dt_s"]),
            "event_scale": float(mura["event_scale"]),
            "ordering_method": mura.get("ordering_integration_method"),
            "ordering_finite_time_certified": mura.get(
                "ordering_finite_time_kinetic_accuracy_certified_by_this_solve"),
            "wall_seconds": elapsed, "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": digest(checkpoint),
            "diagnostics": stage_diagnostics(state, context, driving),
        }
        retained[grid] = fields(state, context["spacing_m"], context["topologies"])
    comparisons = {}
    for left, right, bands in ((128, 192, (24, 63)),
                               (192, 256, (24, 63, 95))):
        key = f"n{left}_n{right}"
        comparisons[key] = {}
        for half in bands:
            comparisons[key][f"half_width_{half}"] = {
                name: relative(crop(retained[left][name], half),
                               crop(retained[right][name], half))
                for name in retained[left]}
    payload = {
        "schema": "asb-drx/v47/fixed-scale-short-refinement/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": source, "domain_m": 3.2e-6,
        "representation_length_m": 4e-7,
        "physical_interval_s": DT_S,
        "scope": "one finite-time Mura interval from identical declared initial states",
        "full_62p5us_n256_completed": False,
        "full_horizon_exclusion": (
            "short profiling/refinement selected because finite-time ordering "
            "makes a matched n256 62.5 us run unaffordable alongside active local DDD"),
        "runs": runs, "comparisons": comparisons,
    }
    curl_192_256 = comparisons["n192_n256"]["half_width_63"]["curl_nye"]
    payload["classification"] = (
        "SHORT_CURRENT_SOURCE_FIXED_SCALE_WITHIN_5PCT"
        if curl_192_256 <= .05 else
        "SHORT_CURRENT_SOURCE_FIXED_SCALE_NOT_WITHIN_5PCT")
    manifest = args.output/"v47_short_refinement.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"classification": payload["classification"],
                      "curl_n192_n256_half63": curl_192_256,
                      "sha256": digest(manifest)}, sort_keys=True))


if __name__ == "__main__":
    main()
