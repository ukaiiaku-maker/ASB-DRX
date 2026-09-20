#!/usr/bin/env python3
"""Restartable V48 endpoint-load continuation with a closed split work clock."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from full_model.analysis.run_v34_finite_coupled_response import (
    I3Controls, _energy_options, resolved_bicrystal, run_i3_cycle,
)
from full_model.analysis.run_v36_recurrent_physical_response import driving_at_time
from full_model.analysis.run_v39_common_horizon import (
    atomic_json, load_stage, save_stage,
)
from full_model.production.complete_front_energy import evaluate_complete_front_energy
from full_model.production.common_front_state import reconstruct_common
from full_model.production.common_tensorial_wall import resolved_driving_components


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def energy(context, state, driving):
    value = evaluate_complete_front_energy(
        state.common_front, state.eta,
        spacing_m=context["spacing_m"],
        represented_thickness_m=context["represented_thickness_m"],
        **_energy_options(context, driving))
    return value


def observables(context, state, driving):
    common, _ = reconstruct_common(state.common_front, context["spacing_m"])
    drive = resolved_driving_components(
        common, driving, context["systems"], context["topologies"],
        context["wall_parameters"])
    beta01 = float(np.mean(common.beta_p[..., 0, 1]))
    beta10 = float(np.mean(common.beta_p[..., 1, 0]))
    return {
        "total_strain_tensor_e12": float(driving.mean_strain[0, 1]),
        "engineering_total_shear_gamma": float(
            driving.mean_strain[0, 1]+driving.mean_strain[1, 0]),
        "plastic_distortion_beta_01": beta01,
        "plastic_distortion_beta_10": beta10,
        "plastic_tensor_strain_e_p12": .5*(beta01+beta10),
        "engineering_plastic_shear_gamma_p": beta01+beta10,
        "mean_shear_stress_sigma_12_Pa": float(np.mean(
            drive["stress_tensor_Pa"][..., 0, 1])),
        "temperature_mean_K": float(np.mean(common.temperature_K)),
        "temperature_peak_K": float(np.max(common.temperature_K)),
    }


def load_record(driving):
    return {
        "mean_strain": (None if driving.mean_strain is None
                        else np.asarray(driving.mean_strain).tolist()),
        "fixed_eigenstrain_present": driving.fixed_eigenstrain is not None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--intervals", type=int, required=True)
    parser.add_argument("--dt-s", type=float, default=4.8828125e-7)
    parser.add_argument("--protocol", choices=("hold", "continued_deformation"),
                        required=True)
    parser.add_argument("--strain-rate-s", type=float, default=100.0)
    parser.add_argument("--restart", type=Path)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    context = resolved_bicrystal(
        grid=args.grid, length_m=3.2e-6, interface_width_m=4e-7,
        temperature_K=1100.0, child_line_fraction=.35)
    if args.restart is None:
        state, source_metadata = load_stage(args.initial_checkpoint, context)
        completed = 0
        physical_time = float(source_metadata["physical_time_s"])
        records = []
        source_checkpoint = args.initial_checkpoint
    else:
        state, restart_metadata = load_stage(args.restart, context)
        completed = int(restart_metadata["completed_intervals"])
        physical_time = float(restart_metadata["physical_time_s"])
        records = list(restart_metadata["records"])
        source_checkpoint = Path(restart_metadata["source_checkpoint"])
    rate = 0.0 if args.protocol == "hold" else float(args.strain_rate_s)
    started = time.perf_counter()
    for index in range(completed, int(args.intervals)):
        t0 = physical_time
        tm = t0+.5*args.dt_s
        t1 = t0+args.dt_s
        d0 = driving_at_time(args.grid, .01, args.protocol, rate, t0)
        dm = driving_at_time(args.grid, .01, args.protocol, rate, tm)
        d1 = driving_at_time(args.grid, .01, args.protocol, rate, t1)
        e0 = energy(context, state, d0)
        em0 = energy(context, state, dm)
        step_started = time.perf_counter()
        candidate, audit = run_i3_cycle(
            context, state, state.eta.copy(), dm,
            I3Controls(
                mura_enabled=True, front_enabled=False,
                trial_dt_s=args.dt_s, front_dt_s=args.dt_s,
                mura_transport_operator="compatible_dealiased"))
        accepted = float(audit["mura"]["accepted_dt_s"])
        if accepted != args.dt_s:
            raise RuntimeError("continuation interval did not close its clock")
        em1 = energy(context, candidate, dm)
        e1 = energy(context, candidate, d1)
        work_pre = em0.recoverable_elastic_J-e0.recoverable_elastic_J
        work_post = e1.recoverable_elastic_J-em1.recoverable_elastic_J
        work = work_pre+work_post
        delta_internal = e1.internal_J-e0.internal_J
        residual = delta_internal-work
        scale = max(abs(e0.internal_J), abs(e1.internal_J), abs(work), 1e-300)
        records.append({
            "interval": index+1,
            "physical_time_start_s": t0,
            "physical_time_midpoint_s": tm,
            "physical_time_end_s": t1,
            "accepted_constitutive_time_s": accepted,
            "load_clock_advanced_only_after_acceptance": True,
            "load_initial": load_record(d0),
            "load_midpoint": load_record(dm),
            "load_endpoint": load_record(d1),
            "initial_observables": observables(context, state, d0),
            "endpoint_observables": observables(context, candidate, d1),
            "energy_initial": asdict(e0),
            "energy_endpoint": asdict(e1),
            "external_work_pre_constitutive_J": work_pre,
            "external_work_post_constitutive_J": work_post,
            "external_work_total_J": work,
            "delta_internal_energy_J": delta_internal,
            "first_law_residual_J": residual,
            "first_law_relative_residual": abs(residual)/scale,
            "constitutive_fixed_midpoint_first_law_residual_J": audit[
                "complete_energy"]["first_law_residual_J"],
            "ordering_backend": audit["mura"]["ordering_finite_time_backend"],
            "ordering_accuracy_certified": audit["mura"][
                "ordering_finite_time_kinetic_accuracy_certified_by_this_solve"],
            "ordering_local_error_control_passed": bool(
                audit["mura"]["ordering_finite_time_error_control_assessed"]
                and audit["mura"][
                    "ordering_finite_time_error_tolerance_satisfied"]),
            "ordering_maximum_accepted_error_norm": audit["mura"][
                "ordering_finite_time_maximum_accepted_error_norm"],
            "pre_ordering_density_sha256": audit["mura"][
                "pre_ordering_density_sha256"],
            "wall_seconds": time.perf_counter()-step_started,
        })
        state = candidate
        physical_time = t1
        checkpoint = output/f"checkpoint_{index+1:06d}.npz"
        metadata = {
            "stage": "V48_COMPLETE_INTERVAL",
            "grid": args.grid,
            "macro_dt_s": args.dt_s,
            "completed_intervals": index+1,
            "physical_time_s": physical_time,
            "protocol": args.protocol,
            "strain_rate_s": rate,
            "source_checkpoint": str(source_checkpoint.resolve()),
            "records": records,
        }
        save_stage(checkpoint, state, context, metadata)
        manifest = {
            "schema": "asb-drx/v48/physical-continuation/v1",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True).strip(),
            "status": "RUNNING" if index+1 < args.intervals else "COMPLETE",
            "protocol": args.protocol,
            "strain_rate_s": rate,
            "grid": args.grid,
            "requested_intervals": args.intervals,
            "completed_intervals": index+1,
            "physical_time_s": physical_time,
            "latest_checkpoint": str(checkpoint.resolve()),
            "latest_checkpoint_sha256": digest(checkpoint),
            "source_checkpoint": str(source_checkpoint.resolve()),
            "source_checkpoint_sha256": digest(source_checkpoint),
            "records": records,
            "wall_seconds": time.perf_counter()-started,
        }
        atomic_json(output/"run_manifest.json", manifest)
    print(json.dumps({
        "status": manifest["status"],
        "completed_intervals": manifest["completed_intervals"],
        "physical_time_s": manifest["physical_time_s"],
        "latest_checkpoint_sha256": manifest["latest_checkpoint_sha256"],
        "wall_seconds": manifest["wall_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
