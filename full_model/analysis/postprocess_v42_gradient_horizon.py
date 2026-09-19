#!/usr/bin/env python3
"""Matched 62.5-us strong/common-mode comparison of the V42 Mura states."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from full_model.analysis.run_v34_finite_coupled_response import resolved_bicrystal
from full_model.analysis.run_v39_common_horizon import load_stage
from full_model.production.common_front_state import reconstruct_common
from full_model.production.density_state_map import derived_density_fields
from full_model.production.tensorial_nye import nye_from_plastic_distortion


def rms(value):
    return float(np.sqrt(np.mean(np.abs(value)**2)))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def coefficients(value, half_width):
    n = value.shape[0]
    spectrum = np.fft.fftshift(
        np.fft.fftn(value, axes=(0, 1))/(n*n), axes=(0, 1))
    center = n//2
    return spectrum[center-half_width:center+half_width+1,
                    center-half_width:center+half_width+1]


def record(path, grid):
    context = resolved_bicrystal(
        grid=grid, length_m=3.2e-6, interface_width_m=4e-7,
        temperature_K=1100.0, child_line_fraction=.35)
    state, metadata = load_stage(path, context)
    common, _ = reconstruct_common(state.common_front, context["spacing_m"])
    alpha = nye_from_plastic_distortion(common.beta_p, context["spacing_m"])
    norm = np.linalg.norm(alpha, axis=(-2, -1)); dx = context["spacing_m"]
    density = derived_density_fields(
        state.mechanical.density, context["topologies"])
    return alpha, {
        "path": str(Path(path).resolve()), "sha256": digest(path),
        "grid": grid, "spacing_m": dx, "metadata": metadata,
        "physical_time_s": float(metadata["physical_time_s"]),
        "curl_nye_rms_m1": rms(alpha),
        "curl_nye_maximum_tensor_norm_m1": float(np.max(norm)),
        "curl_nye_l1_tensor_integral_m": float(
            np.sum(norm, dtype=np.longdouble)*dx*dx),
        "ordered_line_m_per_m": float(np.sum(
            density["rho_wall_ordered_m2"], dtype=np.longdouble)*dx*dx),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n128", type=Path, required=True)
    parser.add_argument("--n192", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    a, ar = record(args.n128, 128); b, br = record(args.n192, 192)
    comparisons = {}
    for key in ("curl_nye_rms_m1", "curl_nye_maximum_tensor_norm_m1",
                "curl_nye_l1_tensor_integral_m", "ordered_line_m_per_m"):
        comparisons[key] = abs(ar[key]-br[key])/max(abs(ar[key]), abs(br[key]), 1e-300)
    modes = {}
    for half in (24, 63):
        ca = coefficients(a, half); cb = coefficients(b, half)
        modes[str(half)] = {
            "shortest_axis_wavelength_m": 3.2e-6/half,
            "coefficient_relative_rms_difference": rms(ca-cb)/max(
                rms(ca), rms(cb), 1e-300),
            "coefficient_count_per_axis": 2*half+1,
        }
    same_time = abs(ar["physical_time_s"]-br["physical_time_s"]) <= 1e-15
    strong = comparisons["curl_nye_rms_m1"]
    payload = {
        "schema": "asb-drx/v42/matched-gradient-refinement/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": "ee4b805e4576af5110301692d72bc4c4d76a68c7",
        "matched_physical_horizon_s": 6.25e-5,
        "same_physical_time": same_time,
        "grids": {"128": ar, "192": br},
        "relative_differences": comparisons,
        "common_mode_comparisons": modes,
        "strong_rms_five_percent_passed": bool(same_time and strong <= .05),
        "classification": (
            "MATCHED_HORIZON_STRONG_NORM_PASSED" if same_time and strong <= .05
            else "UNRESOLVED_SPATIAL_SCALE"),
        "qualification_scope": (
            "mode 24 reproduces the V41 selected physical band; mode 63 spans "
            "the full centered n128 coefficient square except its Nyquist edge. "
            "Neither comparison filters or changes authoritative state."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
