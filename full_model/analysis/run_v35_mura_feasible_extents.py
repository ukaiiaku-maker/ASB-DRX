#!/usr/bin/env python3
"""Compare frozen V34 family screening with feasible complete-affinity extents."""

from __future__ import annotations

import argparse
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
from full_model.production.common_tensorial_wall import CommonWallDriving
from full_model.production.v24_mechanical_wall import accepted_v24_mechanical_step


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compact_audit(audit):
    keys = (
        "admissible", "admissibility_tolerance_J_m3_cells",
        "plastic_work_J_m3_cells",
        "recoverable_elastic_energy_release_J_m3_cells",
        "proposed_defect_energy_change_J_m3_cells",
        "proposed_line_creation_energy_J_m3_cells",
        "dissipative_drag_and_heat_J_m3_cells",
        "reaction_extents",
    )
    return {key: audit[key] for key in keys}


def run_mode(state, driving, support, systems, topologies, common, extensive,
             kinetics, trial_dt, mode):
    result, ledger = accepted_v24_mechanical_step(
        state, driving, support, systems, topologies, common, extensive,
        kinetics, trial_dt, topology_route_enabled=False,
        mura_work_budget_mode=mode)
    budget = ledger["mura_work_budget"]
    families = []
    for candidate in budget["family_candidate_audits"]:
        curve = []
        for point in candidate.get("extent_curve", []):
            row = {key: point[key] for key in (
                "extent", "result", "admissible") if key in point}
            if point["result"] == "EVALUATED":
                row.update({
                    "complete_affinity_J_m3_cells": point[
                        "complete_affinity_J_m3_cells"],
                    "sign_resolved": point["sign_resolved"],
                    "complete_budget": compact_audit(point["audit"]),
                })
            else:
                row["error"] = point["error"]
            curve.append(row)
        families.append({
            "family": candidate["family"],
            "selected": candidate["selected"],
            "selected_extent": candidate.get(
                "selected_extent", float(candidate["selected"])),
            "classification": candidate.get(
                "classification", "V34_FULL_EVENT_BOOLEAN"),
            "extent_curve": curve,
        })
    balance = ledger["mura_balance_ledger"]
    return result, {
        "mode": mode,
        "accepted_dt_s": ledger["accepted_dt_s"],
        "event_scale": ledger["mura_event_scale"],
        "family_event_scales": [float(x) for x in
                                ledger["mura_family_event_scales"]],
        "family_selection_rule": budget["family_selection_rule"],
        "families": families,
        "accepted_complete_budget": compact_audit(budget["accepted"]),
        "minimum_local_heat_increment_J_m3": float(np.min(
            balance["deposited_heat_increment_J_m3"])),
        "first_law_residual_J_m3_cells": balance[
            "global_work_minus_heat_storage_residual_J_m3_cells"],
        "hard_invariant_passed": ledger["nye_suboperator_audit"][
            "accepted_step_hard_invariant_passed"],
        "post_step_projection_used": ledger["nye_suboperator_audit"][
            "post_step_projection_used"],
    }


def checkpoint_case(path, trial_dt):
    grid = int(np.load(path, allow_pickle=False)[
        "v24_common__orientation_rad"].shape[0])
    (_, fixed, support, systems, topologies, common, extensive, kinetics,
     spacing) = create_case(grid, "mechanical_heterogeneity", 42, 1e-5)
    state, metadata = load_checkpoint(path, systems, topologies)
    strain = float(metadata["applied_strain"])
    driving = CommonWallDriving(
        mean_strain=np.array([[0.0, .5*strain], [.5*strain, 0.0]]),
        fixed_eigenstrain=fixed)
    comparisons = []
    for mode in ("energy_limited", "energy_limited_feasible_extents"):
        _, record = run_mode(
            state, driving, support, systems, topologies, common, extensive,
            kinetics, trial_dt, mode)
        comparisons.append(record)
    old, new = comparisons
    return {
        "checkpoint": str(path), "checkpoint_sha256": sha256(path),
        "checkpoint_source_sha": metadata.get("source_sha", "UNRECORDED"),
        "grid": grid, "spacing_m": spacing, "step": metadata["step"],
        "applied_strain": strain, "comparisons": comparisons,
        "operator_changes_accepted_state": bool(
            old["family_event_scales"] != new["family_event_scales"]
            or old["event_scale"] != new["event_scale"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, action="append",
                        required=True)
    parser.add_argument("--trial-dt-s", type=float, default=2e-9)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [checkpoint_case(path, args.trial_dt_s)
             for path in args.checkpoint]
    all_records = [record for case in cases
                   for record in case["comparisons"]]
    fixture_passed = all(
        record["hard_invariant_passed"]
        and not record["post_step_projection_used"]
        and record["minimum_local_heat_increment_J_m3"] >= 0.0
        for record in all_records)
    blocked = [family for case in cases
               for family in case["comparisons"][1]["families"]
               if not family["selected"]]
    result = {
        "schema": "asb-drx/v35-mura-feasible-extent/v1",
        "production_source_sha": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True).strip(),
        "script_sha256": sha256(Path(__file__)),
        "fixture_passed": fixture_passed,
        "scientific_gate_passed": False,
        "scientific_gate_not_passed_reason": (
            "bounded matched-state forks classify family directions but do "
            "not by themselves establish matched-time trajectory convergence"),
        "no_debt_policy": (
            "unaccepted instantaneous extent creates no pending line, energy, "
            "loading time, event clock, or numerical retry state"),
        "blocked_family_classifications": sorted(set(
            family["classification"] for family in blocked)),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"output": str(args.output), "cases": len(cases),
                      "fixture_passed": fixture_passed}, sort_keys=True))


if __name__ == "__main__":
    main()
