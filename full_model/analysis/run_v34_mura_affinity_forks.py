#!/usr/bin/env python3
"""Bounded checkpoint forks for complete-affinity Mura selection."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from full_model.analysis.run_v30_mura_tier_b1_case import create_case, load_checkpoint
from full_model.production.common_tensorial_wall import (
    CommonWallDriving, resolved_driving_components,
)
from full_model.production.v24_mechanical_wall import (
    _resolved_elastic_energy_sum_J_m3_cells, accepted_v24_mechanical_step,
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head():
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def energy(raw, cells, spacing, dt=None):
    if raw is None:
        return None
    raw = float(raw); average = raw/cells
    result = {
        "raw_sum_J_m3_cells": raw,
        "volume_average_J_m3": average,
        "unit_thickness_total_J": raw*spacing**2,
    }
    if dt is not None:
        result.update({
            "raw_sum_rate_W_m3_cells": raw/dt,
            "volume_average_rate_W_m3": average/dt,
            "unit_thickness_total_rate_W": raw*spacing**2/dt,
        })
    return result


def compact_audit(audit, cells, spacing, dt):
    names = (
        "plastic_work_J_m3_cells",
        "recoverable_elastic_energy_release_J_m3_cells",
        "proposed_defect_energy_change_J_m3_cells",
        "proposed_line_creation_energy_J_m3_cells",
        "dissipative_drag_and_heat_J_m3_cells",
    )
    return {
        "admissible": bool(audit["admissible"]),
        "energies": {name.removesuffix("_J_m3_cells"): energy(
            audit[name], cells, spacing, dt) for name in names},
        "family_plastic_work_J_m3_cells": audit[
            "family_plastic_work_J_m3_cells"],
        "family_line_creation_energy_J_m3_cells": audit[
            "family_line_creation_energy_J_m3_cells"],
        "reaction_extents": audit["reaction_extents"],
    }


def compact_step(ledger, cells, spacing, requested_dt):
    budget = ledger["mura_work_budget"]
    accepted = budget["accepted"]
    accepted_dt = float(ledger["accepted_dt_s"])
    candidates = []
    for item in budget["family_candidate_audits"]:
        row = {key: item[key] for key in ("family", "result", "selected")}
        if item["result"] == "EVALUATED":
            row["complete_affinity"] = compact_audit(
                item["audit"], cells, spacing, accepted_dt)
            row["mechanical_work_J_m3_cells"] = item["audit"][
                "proposed_family_work_before_selection_J_m3_cells"][
                    item["family"]]
        else:
            row["error"] = item["error"]
        candidates.append(row)
    balance = ledger["mura_balance_ledger"]
    return {
        "requested_dt_s": float(requested_dt),
        "accepted_dt_s": accepted_dt,
        "dt_ratio": accepted_dt/float(requested_dt),
        "cfl_or_spin_limited": bool(accepted_dt < requested_dt*(1.0-1e-14)),
        "event_scale": float(ledger["mura_event_scale"]),
        "family_event_scales": [float(x) for x in
                                ledger["mura_family_event_scales"]],
        "selected_families": [int(i) for i, x in enumerate(
            ledger["mura_family_event_scales"]) if x > 0.0],
        "stalled_families": budget["stalled_families"],
        "joint_backtrack_trials": int(budget["trial_count"]),
        "physical_stall": bool(budget["physical_stall"]),
        "family_selection_rule": budget["family_selection_rule"],
        "unreacted_remainder_semantics": budget[
            "unreacted_remainder_semantics"],
        "family_unreacted_fractions": budget["family_unreacted_fractions"],
        "family_candidates": candidates,
        "joint_selected_family_full_candidate": compact_audit(
            budget["joint_selected_family_candidate"], cells, spacing,
            accepted_dt),
        "accepted_event": compact_audit(accepted, cells, spacing, accepted_dt),
        "first_law_residual": energy(
            balance["global_work_minus_heat_storage_residual_J_m3_cells"],
            cells, spacing, accepted_dt),
        "minimum_local_heat_increment_J_m3": float(np.min(
            balance["deposited_heat_increment_J_m3"])),
        "hard_invariant_passed": bool(ledger["nye_suboperator_audit"][
            "accepted_step_hard_invariant_passed"]),
        "post_step_projection_used": bool(ledger["nye_suboperator_audit"][
            "post_step_projection_used"]),
    }


def fork_case(checkpoint, grid, trial_dt, strain_rate, reduced=False):
    (_, fixed, support, systems, topologies, common, extensive, kinetics,
     spacing) = create_case(grid, "mechanical_heterogeneity", 42, 1e-5)
    state, metadata = load_checkpoint(checkpoint, systems, topologies)
    strain = float(metadata["applied_strain"])
    mean = np.array([[0.0, .5*strain], [.5*strain, 0.0]])
    hold = CommonWallDriving(mean_strain=mean, fixed_eigenstrain=fixed)
    cells = grid*grid
    scenarios = []

    def run(name, state_arg=state, driving=hold, parameters=common,
            requested_dt=trial_dt, external=None):
        _, ledger = accepted_v24_mechanical_step(
            state_arg, driving, support, systems, topologies, parameters,
            extensive, kinetics, requested_dt, topology_route_enabled=False,
            mura_work_budget_mode="energy_limited")
        row = {"name": name,
               **compact_step(ledger, cells, spacing, requested_dt)}
        if external is not None:
            row["external_constraint_switch_energy"] = energy(
                external, cells, spacing)
            row["external_energy_is_separate_from_mura_heat"] = True
        scenarios.append(row)

    run("fixed_total_strain_hold")
    warm = replace(state, common=replace(
        state.common, temperature_K=state.common.temperature_K+25.0))
    run("temperature_plus_25K", state_arg=warm,
        parameters=replace(common,
                           bath_temperature_K=common.bath_temperature_K+25.0))
    if not reduced:
        increment = strain_rate*trial_dt
        loaded = np.array([[0.0, .5*(strain+increment)],
                           [.5*(strain+increment), 0.0]])
        run("continued_load_plus_requested_increment",
            driving=CommonWallDriving(mean_strain=loaded,
                                      fixed_eigenstrain=fixed))
        release = CommonWallDriving(
            mean_strain=mean, fixed_eigenstrain=np.zeros_like(fixed))
        loaded_energy = _resolved_elastic_energy_sum_J_m3_cells(
            resolved_driving_components(
                state.common, hold, systems, topologies, common))
        released_energy = _resolved_elastic_energy_sum_J_m3_cells(
            resolved_driving_components(
                state.common, release, systems, topologies, common))
        run("mechanical_constraint_release", driving=release,
            external=released_energy-loaded_energy)
        for factor in (.5, .25, .125, .0625):
            run(f"dt_refinement_{factor:g}", requested_dt=trial_dt*factor)
    return {
        "grid": grid, "step": int(metadata["step"]),
        "applied_strain": strain,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_source_sha": metadata.get("source_sha", "UNRECORDED"),
        "spacing_m": spacing, "cell_count": cells,
        "scenarios": scenarios,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--near-64", type=Path, required=True)
    parser.add_argument("--later-64", type=Path, required=True)
    parser.add_argument("--selected-128", type=Path)
    parser.add_argument("--trial-dt-s", type=float, default=2e-9)
    parser.add_argument("--strain-rate-s", type=float, default=1e4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [
        fork_case(args.near_64, 64, args.trial_dt_s, args.strain_rate_s),
        fork_case(args.later_64, 64, args.trial_dt_s, args.strain_rate_s),
    ]
    if args.selected_128 is not None:
        cases.append(fork_case(
            args.selected_128, 128, args.trial_dt_s, args.strain_rate_s,
            reduced=True))
    scenarios = [scenario for case in cases for scenario in case["scenarios"]]
    fixture_passed = all(
        scenario["hard_invariant_passed"]
        and not scenario["post_step_projection_used"]
        and scenario["minimum_local_heat_increment_J_m3"] >= 0.0
        and scenario["accepted_event"]["admissible"]
        for scenario in scenarios)
    result = {
        "schema": "asb-drx/v34-mura-affinity-remainder/v1",
        "production_source_sha": git_head(),
        "diagnostic_script_sha256": sha256(Path(__file__)),
        "live_e7aa_trajectory_modified": False,
        "unreacted_remainder_policy": (
            "deterministic constrained instantaneous rate; no carried debt; "
            "recompute full proposal from the next accepted state and load"),
        "fixture_passed": bool(fixture_passed),
        "operator_qualified": bool(fixture_passed),
        "scientific_gate_passed": False,
        "scientific_gate_not_passed_reason": (
            "bounded operator forks do not establish a new long-trajectory or "
            "spatial-convergence result"),
        "broad_b2_authorized": False,
        "classification": (
            "COMPLETE_AFFINITY_OPERATOR_QUALIFIED_LONG_TRAJECTORY_PENDING"
            if fixture_passed else "COMPLETE_AFFINITY_OPERATOR_INVALID"),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"cases": len(cases), "output": str(args.output)},
                     sort_keys=True))


if __name__ == "__main__":
    main()
