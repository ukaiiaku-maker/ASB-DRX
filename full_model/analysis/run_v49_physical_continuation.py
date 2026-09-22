#!/usr/bin/env python3
"""Restartable V49 current-source continuation with an explicit load origin."""

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
from full_model.analysis.run_v39_common_horizon import (
    atomic_json, load_stage, save_stage,
)
from full_model.analysis.run_v48_physical_continuation import (
    energy, load_record, observables,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def driving(grid, initial_tensor_shear, protocol, rate_s, load_time_s):
    shear = float(initial_tensor_shear)
    if protocol == "continued_deformation":
        shear += float(rate_s)*float(load_time_s)
    from full_model.production.common_tensorial_wall import CommonWallDriving
    return CommonWallDriving(
        mean_strain=np.asarray(((0.0, shear), (shear, 0.0))),
        fixed_eigenstrain=np.zeros((grid, grid, 2, 2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--restart", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--intervals", type=int, required=True)
    parser.add_argument("--dt-s", type=float, default=4.8828125e-7)
    parser.add_argument("--protocol", choices=("hold", "continued_deformation"),
                        required=True)
    parser.add_argument("--strain-rate-s", type=float, default=100.0)
    parser.add_argument("--initial-tensor-shear", type=float, default=.01)
    args = parser.parse_args()
    if (args.initial_checkpoint is None) == (args.restart is None):
        raise ValueError("select exactly one initial checkpoint or restart")
    output = args.output_dir; output.mkdir(parents=True, exist_ok=True)
    context = resolved_bicrystal(
        grid=args.grid, length_m=3.2e-6, interface_width_m=4e-7,
        temperature_K=1100.0, child_line_fraction=.35)
    rate = 0.0 if args.protocol == "hold" else float(args.strain_rate_s)
    if args.restart is None:
        state, source_metadata = load_stage(args.initial_checkpoint, context)
        physical_time = float(source_metadata.get("physical_time_s", 0.0))
        load_origin = physical_time
        completed = 0; records = []
        cumulative_work = 0.0
        d0 = driving(args.grid, args.initial_tensor_shear,
                     args.protocol, rate, 0.0)
        initial_internal = energy(context, state, d0).internal_J
        source_checkpoint = args.initial_checkpoint.resolve()
    else:
        state, metadata = load_stage(args.restart, context)
        required = {
            "grid": args.grid, "protocol": args.protocol,
            "strain_rate_s": rate, "macro_dt_s": args.dt_s,
            "initial_tensor_shear": args.initial_tensor_shear,
        }
        mismatch = {key: (metadata.get(key), value)
                    for key, value in required.items()
                    if metadata.get(key) != value}
        if mismatch:
            raise ValueError(f"restart protocol/configuration mismatch: {mismatch}")
        physical_time = float(metadata["physical_time_s"])
        load_origin = float(metadata["load_origin_time_s"])
        completed = int(metadata["completed_intervals"])
        records = list(metadata["records"])
        cumulative_work = float(metadata["cumulative_external_work_J"])
        initial_internal = float(metadata["initial_internal_energy_J"])
        source_checkpoint = Path(metadata["source_checkpoint"])
    started = time.perf_counter()
    for interval in range(completed, args.intervals):
        remaining = float(args.dt_s); segments = []
        while remaining > 64*np.finfo(float).eps*args.dt_s:
            t0 = physical_time; requested = remaining
            tm = t0+.5*requested; t1_requested = t0+requested
            d0 = driving(args.grid, args.initial_tensor_shear, args.protocol,
                         rate, t0-load_origin)
            dm = driving(args.grid, args.initial_tensor_shear, args.protocol,
                         rate, tm-load_origin)
            before = energy(context, state, d0)
            mid_before = energy(context, state, dm)
            segment_started = time.perf_counter()
            candidate, audit = run_i3_cycle(
                context, state, state.eta.copy(), dm,
                I3Controls(mura_enabled=True, front_enabled=False,
                           trial_dt_s=requested, front_dt_s=requested,
                           mura_transport_operator="compatible_dealiased"))
            accepted = float(audit["mura"]["accepted_dt_s"])
            if accepted <= 0.0 or accepted > requested*(1+1e-12):
                raise RuntimeError("invalid accepted physical substep")
            # If the solver shortened the step, recompute the split-load work
            # at that accepted endpoint; the remaining clock gets a new load.
            t1 = t0+accepted
            dm_accepted = driving(
                args.grid, args.initial_tensor_shear, args.protocol, rate,
                (t0+.5*accepted)-load_origin)
            if accepted != requested:
                mid_before = energy(context, state, dm_accepted)
            mid_after = energy(context, candidate, dm_accepted)
            d1 = driving(args.grid, args.initial_tensor_shear, args.protocol,
                         rate, t1-load_origin)
            after = energy(context, candidate, d1)
            work = ((mid_before.recoverable_elastic_J-before.recoverable_elastic_J)
                    +(after.recoverable_elastic_J-mid_after.recoverable_elastic_J))
            cumulative_work += work
            physical_time = t1; remaining -= accepted; state = candidate
            segments.append({
                "requested_duration_s": requested,
                "accepted_duration_s": accepted,
                "physical_time_start_s": t0,
                "physical_time_end_s": t1,
                "external_work_J": work,
                "delta_internal_energy_J": after.internal_J-before.internal_J,
                "first_law_residual_J": after.internal_J-before.internal_J-work,
                "ordering_linear_iterations": audit["mura"].get(
                    "ordering_linear_iterations"),
                "ordering_internal_substeps": audit["mura"].get(
                    "ordering_internal_substeps"),
                "wall_seconds": time.perf_counter()-segment_started,
            })
        endpoint_drive = driving(
            args.grid, args.initial_tensor_shear, args.protocol, rate,
            physical_time-load_origin)
        endpoint_energy = energy(context, state, endpoint_drive)
        records.append({
            "interval": interval+1,
            "physical_time_end_s": physical_time,
            "load_elapsed_time_s": physical_time-load_origin,
            "endpoint_load": load_record(endpoint_drive),
            "endpoint_observables": observables(context, state, endpoint_drive),
            "endpoint_energy": asdict(endpoint_energy),
            "segments": segments,
            "cumulative_external_work_J": cumulative_work,
            "cumulative_internal_energy_change_J": (
                endpoint_energy.internal_J-initial_internal),
            "cumulative_first_law_residual_J": (
                endpoint_energy.internal_J-initial_internal-cumulative_work),
        })
        checkpoint = output/f"checkpoint_{interval+1:06d}.npz"
        metadata = {
            "stage": "V49_COMPLETE_INTERVAL", "grid": args.grid,
            "macro_dt_s": args.dt_s, "completed_intervals": interval+1,
            "physical_time_s": physical_time,
            "load_origin_time_s": load_origin,
            "initial_tensor_shear": args.initial_tensor_shear,
            "protocol": args.protocol, "strain_rate_s": rate,
            "source_checkpoint": str(source_checkpoint), "records": records,
            "cumulative_external_work_J": cumulative_work,
            "initial_internal_energy_J": initial_internal,
        }
        save_stage(checkpoint, state, context, metadata)
        manifest = {
            "schema": "asb-drx/v49/physical-continuation/v1",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True).strip(),
            "status": "RUNNING" if interval+1 < args.intervals else "COMPLETE",
            **metadata, "latest_checkpoint": str(checkpoint.resolve()),
            "latest_checkpoint_sha256": digest(checkpoint),
            "source_checkpoint_sha256": digest(source_checkpoint),
            "wall_seconds": time.perf_counter()-started,
        }
        atomic_json(output/"run_manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in (
        "status", "completed_intervals", "physical_time_s",
        "latest_checkpoint_sha256", "wall_seconds")}, sort_keys=True))


if __name__ == "__main__":
    main()
