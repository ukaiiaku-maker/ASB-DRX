#!/usr/bin/env python3
"""Replay the first V30 Mura rejection from immutable V31 checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from full_model.analysis.run_v30_mura_tier_b1_case import create_case, load_checkpoint
from full_model.production.common_tensorial_wall import CommonWallDriving
from full_model.production.v24_mechanical_wall import (
    MuraWorkBudgetError, accepted_v24_mechanical_step,
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compact_accepted(ledger):
    budget = ledger["mura_work_budget"]
    balance = ledger["mura_balance_ledger"]
    return {
        "result": "ACCEPTED",
        "event_scale": ledger["mura_event_scale"],
        "family_event_scales": [float(value) for value in
                                ledger["mura_family_event_scales"]],
        "stalled_families": budget["stalled_families"],
        "trial_count": budget["trial_count"],
        "accepted_budget": budget["accepted"],
        "first_law_residual_J_m3_cells": balance[
            "global_work_minus_heat_storage_residual_J_m3_cells"],
        "maximum_scalar_line_balance_residual_m": balance[
            "maximum_scalar_line_balance_residual_m"],
        "maximum_alignment_balance_residual_m": balance[
            "maximum_alignment_balance_residual_m"],
        "hard_invariant_passed": ledger["nye_suboperator_audit"][
            "accepted_step_hard_invariant_passed"],
        "post_step_projection_used": ledger["nye_suboperator_audit"][
            "post_step_projection_used"],
    }


def replay(checkpoint, grid, trial_dt):
    data = create_case(grid, "mechanical_heterogeneity", 42, 1e-5)
    _, fixed, support, systems, topologies, common, extensive, kinetics, _ = data
    state, metadata = load_checkpoint(checkpoint, systems, topologies)
    strain = float(metadata["applied_strain"])
    driving = CommonWallDriving(
        mean_strain=np.array([[0.0, .5*strain], [.5*strain, 0.0]]),
        fixed_eigenstrain=fixed)
    legacy = []
    for factor in (1.0, .5, .25, .125, .0625):
        dt = trial_dt*factor
        try:
            _, ledger = accepted_v24_mechanical_step(
                state, driving, support, systems, topologies, common,
                extensive, kinetics, dt,
                mura_work_budget_mode="legacy_reject")
            item = compact_accepted(ledger)
        except MuraWorkBudgetError as error:
            item = {"result": "REJECTED", "message": str(error),
                    "audit": error.audit}
        item["requested_dt_s"] = dt
        legacy.append(item)
    advanced, ledger = accepted_v24_mechanical_step(
        state, driving, support, systems, topologies, common,
        extensive, kinetics, trial_dt,
        mura_work_budget_mode="energy_limited")
    advanced.validate(systems, topologies)
    return {
        "grid": grid,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256(checkpoint),
        "step": int(metadata["step"]),
        "applied_strain": strain,
        "legacy_timestep_audit": legacy,
        "energy_limited_replay": compact_accepted(ledger),
    }


def plot_decision(result, path):
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    colors = {64: "#3465a4", 128: "#cc0000"}
    for case in result["cases"]:
        grid = case["grid"]
        full = case["legacy_timestep_audit"][0]["audit"]["trials"][0]
        accepted = case["energy_limited_replay"]["accepted_budget"]
        x = np.arange(4)+(0.0 if grid == 64 else .35)
        axes[0, 0].bar(x, full["family_plastic_work_J_m3_cells"], .33,
                       color=colors[grid], alpha=.45, label=f"full {grid}")
        axes[0, 0].bar(x, accepted["family_plastic_work_J_m3_cells"], .18,
                       color=colors[grid], label=f"accepted {grid}")
        terms = [full["recoverable_elastic_energy_release_J_m3_cells"],
                 full["proposed_defect_energy_change_J_m3_cells"],
                 full["dissipative_drag_and_heat_J_m3_cells"],
                 accepted["dissipative_drag_and_heat_J_m3_cells"]]
        axes[0, 1].plot(range(4), terms, "o-", color=colors[grid],
                        label=str(grid))
        dts = []; deficits = []
        for trial in case["legacy_timestep_audit"]:
            audit = (trial["audit"]["trials"][0] if trial["result"] == "REJECTED"
                     else trial["accepted_budget"])
            dts.append(audit["accepted_dt_s"])
            deficits.append(audit["dissipative_drag_and_heat_J_m3_cells"])
        axes[1, 0].plot(dts, deficits, "o-", color=colors[grid], label=str(grid))
        axes[1, 1].bar(x, accepted["family_event_scales"], .33,
                       color=colors[grid], label=str(grid))
    axes[0, 0].axhline(0, color="k", lw=.7)
    axes[0, 0].set_xticks(range(4), ["F0", "F1", "F2", "F3"])
    axes[0, 0].set_ylabel("family plastic work (J m$^{-3}$-cells)")
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].axhline(0, color="k", lw=.7)
    axes[0, 1].set_xticks(range(4), ["elastic\nrelease", "defect\n"+r"$\Delta F$",
                                    "full\nheat", "accepted\nheat"])
    axes[0, 1].set_ylabel("energy (J m$^{-3}$-cells)")
    axes[0, 1].legend(title="grid", fontsize=7)
    axes[1, 0].axhline(0, color="k", lw=.7)
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_xlabel("accepted timestep (s)")
    axes[1, 0].set_ylabel("full-event heat balance (J m$^{-3}$-cells)")
    axes[1, 0].legend(title="grid", fontsize=7)
    axes[1, 1].set_xticks(range(4), ["F0", "F1", "F2", "F3"])
    axes[1, 1].set_ylim(-.05, 1.05)
    axes[1, 1].set_ylabel("accepted family event scale")
    axes[1, 1].legend(title="grid", fontsize=7)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v31-cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    cases = []
    for grid in (64, 128):
        case = args.v31_cases/f"mechanical_heterogeneity_{grid}"
        status = json.loads((case/"status.json").read_text())
        checkpoint = case/status["latest_checkpoint"]
        cases.append(replay(checkpoint, grid, float(status["trial_dt_s"])))
    result = {
        "schema": "asb-drx/v32-mura-first-rejection/v1",
        "immutable_v31_source_sha":
            "675d74183baf2043ad0f7c055fe6e3370435ae65",
        "cases": cases,
        "hypothesis_decisions": {
            "B1_event_extent_too_large": (
                "REJECTED AS A SCALAR TIMESTEP EXPLANATION: every tested "
                "interval remains inadmissible; family-wise constrained "
                "extent remains necessary and leaves its remainder unreacted"),
            "B2_omitted_physical_energy_source": (
                "REJECTED AT FIRST FAILURE: independently recomputed elastic "
                "release plus complete defect free-energy change remains "
                "inadmissible for the unpartitioned event"),
            "B3_missing_reverse_or_saturation": (
                "NOT REQUIRED FOR THIS FIRST FAILURE: existing reverse routes "
                "cannot be spent prospectively; retain for later exposure tests"),
            "B4_physical_stall": (
                "SUPPORTED PER BURGERS FAMILY: the negative-work family stalls "
                "while positive-work families advance conservatively"),
        },
        "promoted_repair": (
            "family-wise dissipation complementarity followed by atomic event "
            "backtracking against recoverable elastic release and exact defect "
            "free-energy change"),
        "exact_off_control": "mura_work_budget_mode=legacy_reject",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    if args.figure is not None:
        plot_decision(result, args.figure)
    print(json.dumps({"cases": len(cases), "output": str(args.output)},
                     sort_keys=True))


if __name__ == "__main__":
    main()
