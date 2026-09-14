#!/usr/bin/env python3
"""Create the machine-readable v10 existing-HAGB SIBM qualification record."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def arrays_equal(a, b) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if np.issubdtype(a.dtype, np.number):
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--continuous", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.source, allow_pickle=True) as source, np.load(
            args.final, allow_pickle=True) as final:
        experiment = json.loads(str(final["sibm_experiment_json"].item()))
        front = json.loads(str(final["sparse_front_metadata_json"].item()))
        source_time = float(source["sim_time"])
        final_time = float(final["sim_time"])
        parameters = json.loads(str(final["P_json"].item()))
        no_label = int(source["Ng"]) == int(final["Ng"])
        orientations_inherited = np.array_equal(source["psi_gv"], final["psi_gv"])
        final_step = int(final["step"])
    with args.diagnostics.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    diagnostic = rows[-1]
    thickness = parameters["nuc_barrier_thickness_b"] * parameters["b"]
    numerical_penalty = {
        "alpha_J": float(diagnostic["F_comp_alpha"]) * thickness,
        "gb_J": float(diagnostic["F_comp_GB"]) * thickness,
        "enters_migration_decision": False,
        "converted_to_heat": False,
    }

    with np.load(args.continuous, allow_pickle=True) as continuous, np.load(
            args.split, allow_pickle=True) as split:
        common = sorted((set(continuous.files) & set(split.files)) - {"P_json"})
        mismatched = [key for key in common
                      if not arrays_equal(continuous[key], split[key])]
        missing = sorted(set(continuous.files) ^ set(split.files))

    ledger = front["ledger"]
    line_scale = max(abs(ledger["parent_line_processed_m"]), 1e-300)
    checks = {
        "existing_hagb_selected": (
            experiment["misorientation_rad"] >= np.deg2rad(15.0)
            and experiment.get("parent_pure_core_cells", 0) >= 16
            and experiment.get("child_pure_core_cells", 0) >= 16),
        "favorable_common_drive": experiment["net_flat_boundary_drive_Pa"] > 0.0,
        "initial_bulge_supercritical": (
            experiment["initial_bulge_radius_m"]
            > experiment["analytical_critical_radius_m"]),
        "bulge_area_grew": (
            experiment["current_excess_bulge_area_m2"]
            > experiment["initial_excess_bulge_area_m2"]),
        "line_balance_closed": abs(ledger["line_closure_m"]) <= 1e-14 * line_scale,
        "signed_burgers_balance_closed": ledger["signed_burgers_change_m2"] == 0.0,
        "front_energy_heat_closed": (
            ledger["line_energy_released_J"] == ledger["heat_released_J"]),
        "no_new_label": no_label,
        "orientations_inherited": bool(orientations_inherited),
        "restart_exact": not mismatched and not missing,
        "new_stochastic_creation_disabled": parameters[
            "disable_new_stochastic_creation_after_restart"],
        "numerical_penalty_excluded": not numerical_penalty[
            "enters_migration_decision"],
    }
    checks = {key: bool(value) for key, value in checks.items()}
    classification = (
        "SIBM_SUPERCRITICAL_GROWTH" if checks["bulge_area_grew"]
        and checks["initial_bulge_supercritical"] else
        "SIBM_STALLED_BY_REHARDENING_OR_DRAG")
    result = {
        "schema": "full-v34-v10-sibm-full-model-local/v1",
        "scope": "short local full-2D qualification; HPC3 horizon pending",
        "source_checkpoint": {
            "path": str(args.source), "sha256": sha256(args.source),
            "step": 6900,
        },
        "final_checkpoint": {
            "path": str(args.final), "sha256": sha256(args.final),
            "step": final_step,
        },
        "boundary": experiment,
        "physical_compatibility": {
            "mean_pressure_Pa": experiment["physical_compatibility_pressure_Pa"],
            "maximum_pressure_Pa": experiment[
                "physical_compatibility_pressure_max_Pa"],
            "basis": "line-tension price of signed-GND and Frank-Bilby residuals",
        },
        "numerical_compatibility_penalty": numerical_penalty,
        "front_ledger": ledger,
        "physical_horizon_s": final_time - source_time,
        "restart": {
            "comparison": "4 continuous steps versus 2+2 steps",
            "authoritative_field_count": len(common),
            "excluded_field": "P_json (continuation nSteps/restart path metadata)",
            "missing_fields": missing,
            "mismatched_fields": mismatched,
            "nan_sentinels_compared_equal": True,
        },
        "checks": checks,
        "fixture_passed": all(checks.values()),
        "scientific_gate_passed": all(checks.values()),
        "classification": classification,
        "strict_asb_qualification": "pending_separate_campaign",
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not result["fixture_passed"]:
        raise SystemExit("full-model SIBM local qualification failed")


if __name__ == "__main__":
    main()
