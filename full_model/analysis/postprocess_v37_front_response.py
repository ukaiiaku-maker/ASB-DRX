#!/usr/bin/env python3
"""Postprocess the V37 continued-loading production front family."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CASES = ("slow_reference", "availability_mid", "shape_n_low")
INTERFACE_WIDTH_M = 4.0e-7


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(path: Path) -> dict[str, object]:
    result = json.loads(path.read_text())
    records = result["records"]
    displacements = np.asarray([
        row.get("accepted_contour_displacement_m", 0.0) for row in records],
        dtype=float)
    processed = np.asarray([
        row.get("processed_line_change_m", 0.0) for row in records], dtype=float)
    energy_passed = all(
        row["complete_energy_decision"] is not None
        and row["complete_energy_decision"]["accepted"]
        and abs(row["complete_energy_decision"]["first_law_residual_J"])
        <= row["complete_energy_decision"]["tolerance_J"]
        for row in records)
    common_clock_passed = all(
        row["mura"] is None or np.isclose(
            row["mura"]["accepted_dt_s"],
            result["configuration"]["dt_s"], rtol=0.0,
            atol=32*np.finfo(float).eps*result["configuration"]["dt_s"])
        for row in records)
    cumulative = float(displacements.sum())
    return {
        "path": str(path.resolve()), "sha256": digest(path),
        "source_commit": result["source_commit"],
        "configuration": result["configuration"],
        "completed_intervals": result["completed_intervals"],
        "physical_time_s": result["physical_time_s"],
        "accepted_front_intervals": result["accepted_front_intervals"],
        "cumulative_signed_sweep_m3": result["cumulative_signed_sweep_m3"],
        "cumulative_contour_displacement_m": cumulative,
        "cumulative_contour_displacement_cells": (
            cumulative/(3.2e-6/result["configuration"]["grid"])),
        "cumulative_contour_displacement_interface_widths": (
            cumulative/INTERFACE_WIDTH_M),
        "cumulative_processed_line_m": float(processed.sum()),
        "energy_acceptance_passed": energy_passed,
        "common_clock_passed": common_clock_passed,
        "target_quarter_width_reached": abs(cumulative) >= .25*INTERFACE_WIDTH_M,
        "time_s": [row["physical_time_end_s"] for row in records],
        "incremental_displacement_m": displacements.tolist(),
        "velocity_m_s": [row["net_velocity_m_s"] for row in records],
        "front_published": [row["front_published"] for row in records],
        "first_nonpublished_interval": next((
            index for index, row in enumerate(records)
            if not row["front_published"]), None),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    cases = {}
    for name in CASES:
        path = args.root/name/"result.json"
        if path.exists():
            cases[name] = summarize(path)
    complete = set(cases) == set(CASES) and all(
        row["completed_intervals"] >= 40 for row in cases.values())
    valid = complete and all(
        row["energy_acceptance_passed"] and row["common_clock_passed"]
        for row in cases.values())
    available_valid = bool(cases) and all(
        row["energy_acceptance_passed"] and row["common_clock_passed"]
        for row in cases.values())
    result = {
        "schema": "asb-drx/v37/recurrent-finite-amplitude/v2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": ("COMPLETE" if complete else "PARTIAL"),
        "classification": (
            "RUNNING_PARTIAL_RESPONSE_FAMILY" if not complete else
            "VALID_CONTINUED_LOADING_RESPONSE_FAMILY" if valid else
            "HARD_INVALID_RESPONSE_FAMILY"),
        "cases": cases, "available_cases_invariants_passed": available_valid,
        "complete": complete, "hard_invariants_passed": valid,
        "finite_amplitude_migration_claimed": bool(valid and any(
            row["target_quarter_width_reached"] for row in cases.values())),
        "drx_claimed": False,
        "claim_boundary": "existing-boundary migration during loading; no grain birth or DRX",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, row in cases.items():
        time = np.asarray(row["time_s"])*1e6
        displacement = np.cumsum(row["incremental_displacement_m"])*1e9
        axes[0].plot(time, displacement, label=name)
        axes[1].plot(time, np.asarray(row["velocity_m_s"])*1e9, label=name)
    axes[0].set(xlabel="physical time (us)", ylabel="accepted displacement (nm)")
    axes[1].set(xlabel="physical time (us)", ylabel="raw velocity (nm/s)")
    for axis in axes:
        axis.grid(alpha=.25); axis.legend(fontsize=8)
    figure.tight_layout(); figure.savefig(args.figure, dpi=180); plt.close(figure)


if __name__ == "__main__":
    main()
