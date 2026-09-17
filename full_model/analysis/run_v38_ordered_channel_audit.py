#!/usr/bin/env python3
"""Locate the first channel responsible for ordered-line timestep sensitivity."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from full_model.analysis.run_v36_mura_rate_limit import context
from full_model.production.common_tensorial_wall import CommonWallDriving
from full_model.production.density_state_map import derived_density_fields
from full_model.production.extensive_wall import accepted_ordering_step
from full_model.production.v24_mechanical_wall import accepted_v24_mechanical_step


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scalar_sum(value, area):
    return float(np.sum(np.asarray(value, dtype=np.longdouble))*area)


def inventories(state, topologies, area):
    fields = derived_density_fields(state.density, topologies)
    return {name+"_line_m_per_m_thickness": scalar_sum(value, area)
            for name, value in fields.items() if name.startswith("rho_wall")}


def advance_trace(checkpoint, dt_s, horizon_s):
    (state, metadata, fixed, support, systems, topologies, common, extensive,
     kinetics, spacing) = context(checkpoint)
    extensive = replace(
        extensive, ordering_internal_substep_s=5e-13,
        ordering_internal_max_substeps=8192)
    initial = state
    rate = float(metadata.get("strain_rate_s", 1e4))
    physical_time = float(metadata["physical_time_s"])
    target = physical_time+horizon_s
    area = spacing*spacing
    cumulative = {
        "captured_line_m_per_m_thickness": 0.0,
        "mura_stretching_line_m_per_m_thickness": 0.0,
        "locking_mobile_to_forest_line_m_per_m_thickness": 0.0,
        "ordering_tangle_to_ordered_line_m_per_m_thickness": 0.0,
        "ordering_turnover_exposure_m_per_m_thickness": 0.0,
    }
    steps = []
    first_stress = None
    while physical_time < target-32*np.finfo(float).eps*target:
        strain = rate*physical_time
        driving = CommonWallDriving(
            mean_strain=np.array([[0.0, .5*strain], [.5*strain, 0.0]]),
            fixed_eigenstrain=fixed)
        requested = min(dt_s, target-physical_time)
        before = inventories(state, topologies, area)
        state, ledger = accepted_v24_mechanical_step(
            state, driving, support, systems, topologies, common, extensive,
            kinetics, requested, topology_route_enabled=False,
            mura_work_budget_mode="energy_limited_feasible_extents",
            feasible_family_extent_levels=12)
        accepted_dt = float(ledger["accepted_dt_s"])
        physical_time += accepted_dt
        capture = ledger["transport_capture"]["sign"]
        locking = ledger["locking_unlocking"]["sign"]
        ordering = ledger["ordering_topology"]["sign"]
        thermo = ledger["ordering_thermodynamics"]
        captured = sum(scalar_sum(capture[sign]["captured_line_m2"], area)
                       for sign in ("plus", "minus"))
        stretching = sum(scalar_sum(
            capture[sign]["mura_line_stretching_m2"], area)
            for sign in ("plus", "minus"))
        locked = sum(scalar_sum(locking[sign]["accepted_line_m2"], area)
                     for sign in ("plus", "minus"))
        ordered = sum(scalar_sum(
            ordering[sign]["accepted_tangle_to_ordered_m2"], area)
            for sign in ("plus", "minus"))
        turnover = sum(scalar_sum(
            thermo["turnover_m2_s"][sign]*accepted_dt, area)
            for sign in ("plus", "minus"))
        values = (captured, stretching, locked, ordered, turnover)
        for key, value in zip(cumulative, values):
            cumulative[key] += value
        if first_stress is None:
            first_stress = np.asarray(ledger["effective_stress_Pa"])
        steps.append({
            "accepted_dt_s": accepted_dt,
            "family_event_scales": [float(x) for x in
                                     ledger["mura_family_event_scales"]],
            "before": before, "after": inventories(state, topologies, area),
            **dict(zip(cumulative, values)),
        })

    # Freeze stress, temperature, and every non-ordering field. This isolates
    # the nonlinear reaction integrator from transport/capture differences.
    reaction_density = initial.density
    reaction_time = 0.0
    reaction_transfer = 0.0
    while reaction_time < horizon_s-32*np.finfo(float).eps*horizon_s:
        interval = min(dt_s, horizon_s-reaction_time)
        updated, ledger, _ = accepted_ordering_step(
            reaction_density, systems, topologies,
            initial.common.orientation_rad,
            np.zeros(initial.common.orientation_rad.shape+(3, 3)),
            first_stress, initial.common.temperature_K, extensive, interval)
        reaction_transfer += sum(scalar_sum(
            ledger["accepted_transfer_m2_s"][sign]*interval, area)
            for sign in ("plus", "minus"))
        reaction_density = updated
        reaction_time += interval
    return {
        "dt_s": dt_s, "accepted_intervals": len(steps),
        "elapsed_physical_time_s": physical_time-float(metadata["physical_time_s"]),
        "initial": inventories(initial, topologies, area),
        "final": inventories(state, topologies, area),
        "cumulative_channel_extents": cumulative,
        "reaction_only_frozen_fields": {
            "net_ordering_transfer_m_per_m_thickness": reaction_transfer,
            "final_ordered_line_m_per_m_thickness": sum(scalar_sum(
                getattr(reaction_density, f"wall_ordered_{sign}_m2"), area)
                for sign in ("plus", "minus")),
        },
        "steps": steps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon-s", type=float, default=4e-10)
    parser.add_argument("--dt-s", type=float, nargs="+", default=(1e-10, 5e-11))
    args = parser.parse_args()
    records = [advance_trace(args.checkpoint, dt, args.horizon_s)
               for dt in args.dt_s]
    coarse, fine = records[0], records[-1]
    comparisons = {}
    for key in coarse["cumulative_channel_extents"]:
        a = coarse["cumulative_channel_extents"][key]
        b = fine["cumulative_channel_extents"][key]
        comparisons[key] = {
            "coarse": a, "fine": b, "absolute_difference": abs(a-b),
            "relative_difference_to_coarse": abs(a-b)/max(abs(a), 1e-300),
        }
    result = {
        "schema": "asb-drx/v38/ordered-channel-audit/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "records": records, "channel_comparisons": comparisons,
        "diagnostic_scope": (
            "matched production trajectories plus a frozen-field reaction-only "
            "diagnostic; the latter is not a promoted physical trajectory"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
