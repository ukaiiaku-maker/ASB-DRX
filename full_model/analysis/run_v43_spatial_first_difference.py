#!/usr/bin/env python3
"""Stage-resolved matched-grid localization of the V42 spatial discrepancy."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from full_model.analysis.run_v34_finite_coupled_response import resolved_bicrystal
from full_model.analysis.run_v36_recurrent_physical_response import driving_at_time
from full_model.analysis.run_v39_common_horizon import (
    front_stage, load_stage, mura_to_time, save_stage, stage_diagnostics,
)
from full_model.production.common_front_state import reconstruct_common
from full_model.production.density_state_map import derived_density_fields
from full_model.production.tensorial_nye import nye_from_plastic_distortion
from full_model.production.wall_topology_supply import reservoir_nye_m1


STAGES = ("initial", "after_first_mura", "after_front", "after_second_mura")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_grid(output, grid, macro_dt_s):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    context = resolved_bicrystal(
        grid=grid, length_m=3.2e-6, interface_width_m=4e-7,
        temperature_K=1100.0, child_line_fraction=.35)
    state = context["state"]
    midpoint = .5*macro_dt_s
    driving = driving_at_time(grid, .01, "hold", 0.0, midpoint)
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip()
    records = {}

    def retain(name, current, extra=None):
        path = output/f"{name}.npz"
        diagnostic = stage_diagnostics(current, context, driving)
        metadata = {
            "source_sha": source, "stage": name, "grid": grid,
            "macro_dt_s": macro_dt_s, "completed_intervals": 0,
            "physical_time_s": (0.0 if name == "initial" else
                                  .5*macro_dt_s if name == "after_first_mura"
                                  else macro_dt_s),
            "records": [], "v43_stage_diagnostics": diagnostic,
            **(extra or {}),
        }
        save_stage(path, current, context, metadata)
        records[name] = {"path": str(path.resolve()), "sha256": digest(path),
                         "diagnostics": diagnostic, **(extra or {})}

    retain("initial", state)
    state, pre, pre_elapsed = mura_to_time(
        context, state, .5*macro_dt_s, driving)
    retain("after_first_mura", state, {
        "operator_exposure_s": pre_elapsed, "subcycle_count": len(pre),
        "accepted_dt_s": [float(row["accepted_dt_s"]) for row in pre],
        "event_scales": [float(row["mura_event_scale"]) for row in pre],
    })
    state, front = front_stage(
        context, state, macro_dt_s, driving, .0625, 1.0, 1)
    retain("after_front", state, {
        "operator_exposure_s": macro_dt_s,
        "published": bool(front["candidate_sweep_published"]),
        "classification": front["front_decision"]["classification"],
    })
    state, post, post_elapsed = mura_to_time(
        context, state, .5*macro_dt_s, driving)
    retain("after_second_mura", state, {
        "operator_exposure_s": post_elapsed, "subcycle_count": len(post),
        "accepted_dt_s": [float(row["accepted_dt_s"]) for row in post],
        "event_scales": [float(row["mura_event_scale"]) for row in post],
    })
    result = {
        "schema": "asb-drx/v43/spatial-stage-run/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": source, "grid": grid, "macro_dt_s": macro_dt_s,
        "records": records,
    }
    (output/"stage_run.json").write_text(
        json.dumps(result, indent=2, sort_keys=True)+"\n")
    return result


def fields(path, grid):
    context = resolved_bicrystal(
        grid=grid, length_m=3.2e-6, interface_width_m=4e-7,
        temperature_K=1100.0, child_line_fraction=.35)
    state, metadata = load_stage(path, context)
    common, _ = reconstruct_common(state.common_front, context["spacing_m"])
    density = derived_density_fields(state.mechanical.density,
                                     context["topologies"])
    reservoir = reservoir_nye_m1(
        state.mechanical.reservoir_alignment, context["systems"],
        common.orientation_rad, context["topologies"])["total"]
    alpha = nye_from_plastic_distortion(common.beta_p, context["spacing_m"])
    return {
        "beta_p": common.beta_p,
        "curl_nye": alpha,
        "reservoir_nye": reservoir,
        "orientation": common.orientation_rad,
        "temperature": common.temperature_K,
        "ordered_density": density["rho_wall_ordered_m2"],
        "total_density": density["rho_total_m2"],
        "child_phase": state.eta[..., 1],
    }, metadata, context


def coefficients(value, half_width):
    n = value.shape[0]
    spectrum = np.fft.fftshift(
        np.fft.fftn(value, axes=(0, 1))/(n*n), axes=(0, 1))
    center = n//2
    return spectrum[center-half_width:center+half_width+1,
                    center-half_width:center+half_width+1]


def rms(value):
    return float(np.sqrt(np.mean(np.abs(value)**2)))


def compare_field(a, b, half=24):
    ca = coefficients(a, half); cb = coefficients(b, half)
    scale = max(rms(ca), rms(cb), 1e-300)
    power_scale = max(rms(np.abs(ca)**2), rms(np.abs(cb)**2), 1e-300)
    active = np.maximum(np.abs(ca), np.abs(cb)) > 1e-8*max(
        float(np.max(np.abs(ca))), float(np.max(np.abs(cb))), 1e-300)
    phase = np.angle(ca[active]*np.conj(cb[active])) if np.any(active) else np.zeros(1)
    return {
        "complex_coefficient_relative_rms": rms(ca-cb)/scale,
        "power_relative_rms": rms(np.abs(ca)**2-np.abs(cb)**2)/power_scale,
        "active_coefficient_phase_rms_rad": rms(phase),
        "n128_rms": rms(a), "n192_rms": rms(b),
        "rms_relative_difference": abs(rms(a)-rms(b))/max(rms(a), rms(b), 1e-300),
    }


def compare(root128, root192, output):
    roots = {128: Path(root128), 192: Path(root192)}
    payload = {
        "schema": "asb-drx/v43/spatial-first-difference/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "coordinate_convention": (
            "periodic cell-centered physical coordinates on [0,3.2 um)^2; "
            "spectral derivatives; raw complex Fourier coefficients normalized by n^2"),
        "common_mode_half_width": 24, "stages": {},
    }
    first = None
    for stage in STAGES:
        fa, ma, ca = fields(roots[128]/f"{stage}.npz", 128)
        fb, mb, cb = fields(roots[192]/f"{stage}.npz", 192)
        comparisons = {name: compare_field(fa[name], fb[name]) for name in fa}
        curl_error = comparisons["curl_nye"]["complex_coefficient_relative_rms"]
        if first is None and curl_error > .05:
            first = stage
        payload["stages"][stage] = {
            "source_sha": {"128": ma["source_sha"], "192": mb["source_sha"]},
            "physical_time_s": {"128": ma["physical_time_s"],
                                "192": mb["physical_time_s"]},
            "spacing_m": {"128": ca["spacing_m"], "192": cb["spacing_m"]},
            "fields": comparisons,
            "within_grid_nye_identity_relative_rms": {
                str(n): roots[n].joinpath(f"{stage}.npz").is_file()
                and json.loads(str(np.load(
                    roots[n]/f"{stage}.npz", allow_pickle=False)[
                        "v39_stage_metadata_json"].item()))[
                            "v43_stage_diagnostics"]["nye"]["difference_rms_m1"]
                for n in (128, 192)},
        }
    payload["first_stage_above_five_percent_curl_complex_error"] = first
    payload["classification"] = (
        "INITIAL_REPRESENTATION_DISCREPANCY" if first == "initial" else
        "FIRST_MURA_HALF_STAGE_DISCREPANCY" if first == "after_first_mura" else
        "LATER_SPLIT_STAGE_DISCREPANCY" if first else "COMMON_MODE_PASSED")
    Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    return payload


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run"); run.add_argument("--output", type=Path, required=True)
    run.add_argument("--grid", type=int, choices=(128, 192), required=True)
    run.add_argument("--macro-dt-s", type=float, default=7.8125e-6)
    comp = sub.add_parser("compare"); comp.add_argument("--n128", type=Path, required=True)
    comp.add_argument("--n192", type=Path, required=True)
    comp.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = (run_grid(args.output, args.grid, args.macro_dt_s)
              if args.command == "run" else compare(args.n128, args.n192, args.output))
    print(json.dumps({key: result.get(key) for key in
                      ("schema", "classification", "source_sha", "grid")},
                     sort_keys=True))


if __name__ == "__main__":
    main()
