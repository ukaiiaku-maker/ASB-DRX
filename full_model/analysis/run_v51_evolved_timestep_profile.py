#!/usr/bin/env python3
"""One copied-endpoint timestep fork with a lightweight cumulative profile."""

from __future__ import annotations

import argparse
import cProfile
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pstats
import subprocess
import time

import numpy as np

from full_model.analysis.run_v34_finite_coupled_response import resolved_bicrystal
from full_model.analysis.run_v39_common_horizon import load_stage
from full_model.analysis.run_v48_physical_continuation import observables
from full_model.analysis.run_v49_physical_continuation import (
    advance_consistent_midpoint_segment, driving,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative_l2(a, b):
    av = np.asarray(a); bv = np.asarray(b)
    return float(np.linalg.norm(av-bv)/max(np.linalg.norm(bv), 1e-300))


def cumulative_profile_groups(profile):
    stats = pstats.Stats(profile)
    groups = {
        "ordering_and_chemical_potential": ("ordering", "chemical_potential"),
        "krylov_and_linear_solve": ("gmres", "iterative", "linear_solve"),
        "fft_jvp_and_matvec": ("fft", "jvp", "matvec"),
        "energy_reconstruction": ("energy", "free_energy"),
        "state_construction_validation": ("validate", "replace", "synchronize"),
        "io": ("save", "load", "write", "read"),
    }
    result = {}
    for label, tokens in groups.items():
        rows = []
        for (filename, line, function), values in stats.stats.items():
            calls, primitive, own, cumulative, callers = values
            descriptor = f"{filename}:{line}:{function}".lower()
            if any(token in descriptor for token in tokens):
                rows.append({"function": f"{Path(filename).name}:{line}:{function}",
                             "calls": int(calls), "self_seconds": float(own),
                             "cumulative_seconds": float(cumulative)})
        rows.sort(key=lambda item: item["cumulative_seconds"], reverse=True)
        result[label] = {
            "top_rows": rows[:12],
            "maximum_cumulative_seconds": (
                rows[0]["cumulative_seconds"] if rows else 0.0),
            "interpretation": (
                "overlapping cumulative call-stack exposure; categories are not additive"),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--dt-s", type=float, default=4.8828125e-7)
    args = parser.parse_args()
    context = resolved_bicrystal(
        grid=args.grid, length_m=3.2e-6, interface_width_m=4e-7,
        temperature_K=1100.0, child_line_fraction=.35)
    initial, metadata = load_stage(args.checkpoint, context)
    t0 = float(metadata["physical_time_s"])
    load_origin = float(metadata["load_origin_time_s"])
    initial_shear = float(metadata["initial_tensor_shear"])
    rate = float(metadata["strain_rate_s"])
    kwargs = dict(
        grid=args.grid, initial_tensor_shear=initial_shear,
        protocol=metadata["protocol"], rate=rate, physical_time=t0,
        load_origin=load_origin)
    profile = cProfile.Profile(); profile.enable(); started = time.perf_counter()
    full = advance_consistent_midpoint_segment(
        context, initial, requested_duration=args.dt_s, **kwargs)
    full_wall = time.perf_counter()-started; profile.disable()
    split_state = initial; split_time = t0; split_work = 0.0
    split_residual = 0.0; split_rows = []; split_started = time.perf_counter()
    for index in range(2):
        half = advance_consistent_midpoint_segment(
            context, split_state, requested_duration=.5*args.dt_s,
            **{**kwargs, "physical_time": split_time})
        split_state = half["candidate"]
        split_time += half["accepted_duration_s"]
        split_work += half["external_work_J"]
        split_residual += half["first_law_residual_J"]
        split_rows.append({
            "half": index+1,
            "accepted_duration_s": half["accepted_duration_s"],
            "discarded_shortened_candidates": half[
                "discarded_shortened_candidates"],
            "ordering_linear_iterations": half["audit"]["mura"].get(
                "ordering_linear_iterations"),
            "first_law_residual_J": half["first_law_residual_J"],
        })
    split_wall = time.perf_counter()-split_started
    endpoint_drive = driving(
        args.grid, initial_shear, metadata["protocol"], rate,
        t0+args.dt_s-load_origin)
    full_observables = observables(context, full["candidate"], endpoint_drive)
    split_observables = observables(context, split_state, endpoint_drive)
    full_mechanical = full["candidate"].mechanical
    split_mechanical = split_state.mechanical
    payload = {
        "schema": "asb-drx/v51/evolved-timestep-profile/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": digest(args.checkpoint),
        "fork_is_read_only_from_checkpoint": True,
        "physical_time_start_s": t0,
        "physical_time_end_s": t0+args.dt_s,
        "full_step": {
            "requested_duration_s": args.dt_s,
            "accepted_duration_s": full["accepted_duration_s"],
            "discarded_shortened_candidates": full[
                "discarded_shortened_candidates"],
            "wall_seconds": full_wall,
            "ordering_linear_iterations": full["audit"]["mura"].get(
                "ordering_linear_iterations"),
            "first_law_residual_J": full["first_law_residual_J"],
            "external_work_J": full["external_work_J"],
            "observables": full_observables,
        },
        "two_half_steps": {
            "rows": split_rows, "wall_seconds": split_wall,
            "summed_first_law_residual_J": split_residual,
            "summed_external_work_J": split_work,
            "observables": split_observables,
        },
        "full_vs_two_half": {
            "eta_relative_l2": relative_l2(
                full["candidate"].eta, split_state.eta),
            "beta_p_relative_l2": relative_l2(
                full_mechanical.common.beta_p, split_mechanical.common.beta_p),
            "temperature_relative_l2": relative_l2(
                full_mechanical.common.temperature_K,
                split_mechanical.common.temperature_K),
            "family_nye_relative_l2": relative_l2(
                full_mechanical.common.family_nye_m1,
                split_mechanical.common.family_nye_m1),
            "observable_differences": {
                key: float(full_observables[key]-split_observables[key])
                for key in full_observables},
        },
        "profile": cumulative_profile_groups(profile),
        "profile_scope": (
            "one copied evolved n128 full macro; cumulative categories overlap"),
        "front_enabled": False, "subcell_geometry_enabled": False,
        "drx_claimed": False, "strict_asb_claimed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"output_sha256": digest(args.output),
                      "full_wall_seconds": full_wall,
                      "split_wall_seconds": split_wall,
                      "beta_p_relative_l2": payload["full_vs_two_half"][
                          "beta_p_relative_l2"]}, sort_keys=True))


if __name__ == "__main__":
    main()
