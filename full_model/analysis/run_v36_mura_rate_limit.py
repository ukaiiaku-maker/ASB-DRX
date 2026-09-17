#!/usr/bin/env python3
"""Matched-time timestep and extent-resolution audit for the V36 Mura limiter."""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from full_model.analysis.run_v30_mura_tier_b1_case import (  # noqa: E402
    compact_metrics, create_case, load_checkpoint,
)
from full_model.production.common_tensorial_wall import (  # noqa: E402
    CommonWallDriving,
)
from full_model.production.v24_mechanical_wall import (  # noqa: E402
    accepted_v24_mechanical_step,
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def state_arrays(state):
    return {f"{group}.{field.name}": np.asarray(getattr(value, field.name))
            for group in ("common", "density", "reservoir_alignment")
            for value in (getattr(state, group),)
            for field in fields(value)}


def state_difference(left, right):
    result = {}
    for name, a in state_arrays(left).items():
        b = state_arrays(right)[name]
        denominator = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1.0)
        result[name] = float(np.linalg.norm(a-b)/denominator)
    density_names = [field.name for field in fields(left.density)]
    density_delta = sum(float(np.sum(np.abs(
        getattr(left.density, name)-getattr(right.density, name))))
        for name in density_names)
    density_scale = max(sum(float(np.sum(np.abs(
        getattr(left.density, name)))) for name in density_names), 1.0)
    alignment_names = [field.name for field in fields(left.reservoir_alignment)]
    alignment_delta = sum(float(np.sum(np.abs(
        getattr(left.reservoir_alignment, name)
        -getattr(right.reservoir_alignment, name))))
        for name in alignment_names)
    alignment_scale = max(sum(float(np.sum(np.abs(
        getattr(left.reservoir_alignment, name))))
        for name in alignment_names), 1.0)
    physical = {
        "density_inventory_relative_l1": density_delta/density_scale,
        "reservoir_alignment_relative_l1": alignment_delta/alignment_scale,
    }
    for name in ("beta_p", "slip", "family_nye_m1"):
        a = np.asarray(getattr(left.common, name))
        b = np.asarray(getattr(right.common, name))
        physical[name+"_relative_l2"] = float(np.linalg.norm(a-b)/max(
            float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1.0))
    physical["orientation_maximum_absolute_rad"] = float(np.max(np.abs(
        left.common.orientation_rad-right.common.orientation_rad)))
    physical["temperature_maximum_absolute_K"] = float(np.max(np.abs(
        left.common.temperature_K-right.common.temperature_K)))
    return {
        "maximum_relative_l2": max(result.values()),
        "per_field_relative_l2": result,
        "inventory_normalized_physical_differences": physical,
    }


def context(checkpoint):
    with np.load(checkpoint, allow_pickle=False) as archive:
        grid = int(archive["v24_common__orientation_rad"].shape[0])
    (_, fixed, support, systems, topologies, common, extensive, kinetics,
     spacing) = create_case(grid, "mechanical_heterogeneity", 42, 1e-5)
    state, metadata = load_checkpoint(checkpoint, systems, topologies)
    return (state, metadata, fixed, support, systems, topologies, common,
            extensive, kinetics, spacing)


def advance(checkpoint, trial_dt_s, extent_levels, horizon_s):
    (state, metadata, fixed, support, systems, topologies, common, extensive,
     kinetics, spacing) = context(checkpoint)
    strain_rate = float(metadata.get("strain_rate_s", 1e4))
    physical_time = float(metadata["physical_time_s"])
    target_time = physical_time+horizon_s
    ledgers = []
    start = time.perf_counter()
    while physical_time < target_time-32*np.finfo(float).eps*target_time:
        strain = strain_rate*physical_time
        driving = CommonWallDriving(
            mean_strain=np.array([[0.0, .5*strain], [.5*strain, 0.0]]),
            fixed_eigenstrain=fixed)
        requested = min(trial_dt_s, target_time-physical_time)
        state, ledger = accepted_v24_mechanical_step(
            state, driving, support, systems, topologies, common, extensive,
            kinetics, requested, topology_route_enabled=False,
            mura_work_budget_mode="energy_limited_feasible_extents",
            feasible_family_extent_levels=extent_levels)
        physical_time += ledger["accepted_dt_s"]
        ledgers.append(ledger)
    wall_s = time.perf_counter()-start
    last = ledgers[-1]
    return state, {
        "trial_dt_s": trial_dt_s,
        "extent_levels": extent_levels,
        "horizon_s": horizon_s,
        "accepted_intervals": len(ledgers),
        "elapsed_physical_time_s": physical_time-float(metadata["physical_time_s"]),
        "wall_seconds": wall_s,
        "family_event_scales": [[float(x) for x in
                                 ledger["mura_family_event_scales"]]
                                for ledger in ledgers],
        "family_trial_counts": [[len(candidate["extent_curve"])
                                 for candidate in ledger["mura_work_budget"][
                                     "family_candidate_audits"]]
                                for ledger in ledgers],
        "hard_invariants_passed": all(ledger["nye_suboperator_audit"][
            "accepted_step_hard_invariant_passed"] for ledger in ledgers),
        "minimum_heat_increment_J_m3": min(float(np.min(ledger[
            "mura_balance_ledger"]["deposited_heat_increment_J_m3"]))
            for ledger in ledgers),
        "final_metrics": compact_metrics(
            state, last, systems, topologies, spacing),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon-s", type=float, default=4e-10)
    parser.add_argument("--reference-dt-s", type=float, default=2e-9)
    parser.add_argument("--refined-dt-s", type=float, nargs="+",
                        default=(1e-10, 5e-11))
    parser.add_argument("--extent-levels", type=int, nargs="+",
                        default=(8, 12, 16))
    parser.add_argument("--measured-v35-wall-s", type=float)
    parser.add_argument("--measured-v36-wall-s", type=float)
    args = parser.parse_args()
    configurations = []
    states = []
    # Reference is shared by both comparisons, avoiding an uninformative
    # duplicate trajectory.
    settings = [(args.reference_dt_s, 12)]
    settings += [(trial_dt, 12) for trial_dt in args.refined_dt_s]
    settings += [(args.reference_dt_s, level) for level in args.extent_levels
                 if level != 12]
    for trial_dt, levels in settings:
        state, record = advance(
            args.checkpoint, trial_dt, levels, args.horizon_s)
        states.append(state); configurations.append(record)
    reference = configurations[0]
    comparisons = []
    for state, record in zip(states[1:], configurations[1:]):
        kind = ("timestep" if record["trial_dt_s"] != args.reference_dt_s
                else "extent_resolution")
        comparisons.append({
            "kind": kind,
            "candidate": {"trial_dt_s": record["trial_dt_s"],
                          "extent_levels": record["extent_levels"]},
            "matched_time_error_s": abs(record["elapsed_physical_time_s"]
                                         -reference["elapsed_physical_time_s"]),
            "state_difference_from_reference": state_difference(
                states[0], state),
        })
    timestep_indices = [index for index, record in enumerate(configurations)
                        if record["extent_levels"] == 12]
    timestep_pairwise = []
    for left, right in zip(timestep_indices[:-1], timestep_indices[1:]):
        timestep_pairwise.append({
            "coarser_trial_dt_s": configurations[left]["trial_dt_s"],
            "finer_trial_dt_s": configurations[right]["trial_dt_s"],
            "matched_time_error_s": abs(
                configurations[left]["elapsed_physical_time_s"]
                -configurations[right]["elapsed_physical_time_s"]),
            "state_difference": state_difference(states[left], states[right]),
        })
    valid = all(record["hard_invariants_passed"]
                and record["minimum_heat_increment_J_m3"] >= 0.0
                for record in configurations)
    profile = {
        "measurement_scope": (
            "same checkpoint; run_v35 two-mode audit including frozen "
            "energy-limited comparator and feasible-extent mode"),
        "v35_source_sha": "ce9d9ab",
        "v35_wall_seconds": args.measured_v35_wall_s,
        "v36_wall_seconds": args.measured_v36_wall_s,
        "speedup": (None if args.measured_v35_wall_s is None
                    or args.measured_v36_wall_s is None else
                    args.measured_v35_wall_s/args.measured_v36_wall_s),
        "preserved_checkpoint_single_step_family_trials": {
            "v35_exhaustive": 48,
            "v36_adaptive": sum(reference["family_trial_counts"][0]),
        },
    }
    result = {
        "schema": "asb-drx/v36-mura-rate-limit/v1",
        "source_sha": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True).strip(),
        "analysis_sha256": sha256(Path(__file__)),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "fixture_passed": bool(valid and all(
            row["matched_time_error_s"] <= 64*np.finfo(float).eps*args.horizon_s
            for row in comparisons)),
        "scientific_gate_passed": False,
        "scientific_gate_not_passed_reason": (
            "bounded preserved-checkpoint continuations establish rate-limit "
            "numerics, not long-time wall-pattern convergence"),
        "physics_changed": False,
        "rate_limit_classification": (
            "extent_resolution_exact_over_bounded_horizon; timestep "
            "convergence must be judged on inventory-normalized fields "
            "because individually tiny ordered reservoirs are ill-conditioned"),
        "profile": profile,
        "reference": reference,
        "configurations": configurations,
        "comparisons": comparisons,
        "timestep_pairwise_comparisons": timestep_pairwise,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"output": str(args.output),
                      "fixture_passed": result["fixture_passed"],
                      "configurations": len(configurations)}, sort_keys=True))


if __name__ == "__main__":
    main()
