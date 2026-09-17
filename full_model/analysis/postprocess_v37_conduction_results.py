#!/usr/bin/env python3
"""Classify fetched V37 finite-conduction response-family cases."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from full_model.analysis.postprocess_v32_asb_anchor import (
    relative_ledger_invariants, valid_checkpoints,
)
from full_model.analysis.run_v37_conduction_localization import endpoint


CASE_IDS = (
    "rate_low", "rate_high", "temperature_low", "temperature_high",
    "heterogeneity_short", "heterogeneity_long",
)


def locate_case(root: Path, name: str) -> Path | None:
    candidates = [path for path in root.rglob(name) if path.is_dir()
                  and (path/"v37_conduction_run_record.json").exists()]
    if len(candidates) > 1:
        raise ValueError(f"multiple attributable directories for {name}")
    return candidates[0] if candidates else None


def summarize_case(directory: Path) -> dict[str, object]:
    checkpoints = valid_checkpoints(directory)
    if not checkpoints:
        raise ValueError(f"no valid V37 checkpoint in {directory}")
    steps = sorted(checkpoints)
    latest = checkpoints[steps[-1]]
    trajectory = []
    invariant_rows = []
    for step in steps:
        item = endpoint(checkpoints[step])
        with np.load(checkpoints[step], allow_pickle=True) as data:
            invariants = relative_ledger_invariants(data)
        invariant_rows.append(invariants)
        trajectory.append({
            "step": step, "physical_time_s": item["physical_time_s"],
            "nominal_strain": item["nominal_strain"],
            "temperature_peak_minus_mean_K": item["temperature"]["peak_minus_mean"],
            "temperature_ipr_fraction": item["temperature"]["inverse_participation_fraction"],
            "activity_ipr_fraction": item["absolute_shear_rate"]["inverse_participation_fraction"],
            "activity_minor_fwhm_m": item["absolute_shear_rate"][
                "second_moment_widths"]["minor_gaussian_fwhm_m"],
        })
    record = json.loads((directory/"v37_conduction_run_record.json").read_text())
    terminal = bool(record.get("terminal", False))
    valid = terminal and all(row["passed"] for row in invariant_rows)
    candidate_flags = [
        row["activity_ipr_fraction"] <= .25
        and row["temperature_peak_minus_mean_K"] >= 50.0
        for row in trajectory]
    persistent = len(candidate_flags) >= 2 and all(candidate_flags[-2:])
    if not valid:
        classification = "INCOMPLETE_OR_HARD_INVALID"
    elif persistent:
        classification = "PARTIAL_LOCALIZATION_REQUIRES_MATCHED_CONTROL_AND_REFINEMENT"
    else:
        classification = "BROAD_OR_NONPERSISTENT_HEATING"
    final = endpoint(latest)
    return {
        "directory": str(directory.resolve()), "run_record": record,
        "checkpoint_steps": steps, "latest": final,
        "trajectory": trajectory, "terminal": terminal,
        "hard_invariants_passed": valid,
        "classification": classification,
        "candidate_persistent_last_two_checkpoints": persistent,
        "strict_asb_claimed": False,
        "strict_asb_blockers": [
            "no matched isothermal/frozen-flow control in screening family",
            "no promoted n192 refinement result",
            "frozen persistence conjunction not independently passed"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    cases = {}
    errors = {}
    for name in CASE_IDS:
        directory = locate_case(args.root, name)
        if directory is None:
            continue
        try:
            cases[name] = summarize_case(directory)
        except (OSError, ValueError, KeyError, EOFError) as error:
            errors[name] = str(error)
    valid_names = [name for name, row in cases.items()
                   if row["hard_invariants_passed"]]
    promotion = {"strongest_concentration": None, "broad_heating_negative": None,
                 "intermediate": None}
    if valid_names:
        ranked = sorted(valid_names, key=lambda name: cases[name]["latest"][
            "absolute_shear_rate"]["inverse_participation_fraction"])
        promotion["strongest_concentration"] = ranked[0]
        promotion["broad_heating_negative"] = ranked[-1]
        promotion["intermediate"] = ranked[len(ranked)//2]
    complete = len(cases) == len(CASE_IDS) and not errors
    result = {
        "schema": "asb-drx/v37/conduction-localization-results/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE" if complete else "PARTIAL",
        "classification": (
            "VALID_FINITE_CONDUCTION_RESPONSE_FAMILY" if complete and
            len(valid_names) == len(CASE_IDS) else
            "RUNNING_OR_PARTIAL_FINITE_CONDUCTION_RESPONSE_FAMILY"),
        "cases": cases, "errors": errors, "promotion_selection": promotion,
        "strict_asb_claimed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, row in cases.items():
        history = row["trajectory"]
        strain = [item["nominal_strain"] for item in history]
        axes[0].plot(strain, [item["activity_ipr_fraction"] for item in history],
                     label=name)
        axes[1].plot(strain, [item["temperature_peak_minus_mean_K"]
                              for item in history], label=name)
    axes[0].axhline(.25, color="k", linestyle="--", linewidth=.8)
    axes[0].set(xlabel="nominal strain", ylabel="activity IPR fraction")
    axes[1].axhline(50.0, color="k", linestyle="--", linewidth=.8)
    axes[1].set(xlabel="nominal strain", ylabel="Tmax - Tmean (K)")
    for axis in axes:
        axis.grid(alpha=.25); axis.legend(fontsize=7)
    figure.tight_layout(); figure.savefig(args.figure, dpi=180); plt.close(figure)


if __name__ == "__main__":
    main()
