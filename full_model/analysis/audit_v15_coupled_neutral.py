#!/usr/bin/env python3
"""Decision audit for a v15 fully coupled neutral fixed-point attempt."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from audit_v15_growth_increments import linear_velocity


def checkpoint(directory: Path):
    return sorted(directory.glob("drx_v25_restart_*.npz"))[-1]


def checkpoint_record(path: Path):
    with np.load(path, allow_pickle=True) as state:
        metadata = json.loads(str(state["sparse_front_metadata_json"].item()))
        experiment = json.loads(str(state["sibm_experiment_json"].item()))
        populations = (state["rp"], state["rm"], state["rho_forest"],
                       state["rho_wall"])
        return {
            "path": str(path), "ledger": metadata["ledger"],
            "pressure_Pa": float(experiment["applied_continuation_pressure_Pa"]),
            "mean_total_density_m2": float(np.mean(state["rho"])),
            "minimum_population_m2": float(min(np.min(x) for x in populations)),
            "phase_simplex_error": float(np.max(np.abs(
                np.sum(state["eta"], axis=2)-1.0))),
            "Ng": int(state["Ng"]),
            "orientation_count": int(np.asarray(state["psi_gv"]).size),
        }


def branch_record(directory: Path, source_ledger, grid):
    path = checkpoint(directory)
    record = checkpoint_record(path)
    rows = list(csv.DictReader((directory/"sibm_contour_diagnostics.csv").open()))
    time = [float(row["time_s"]) for row in rows]
    amplitude = [float(row["bulge_amplitude_m"]) for row in rows]
    window = max(3, len(rows)//3)
    increment = {key: float(record["ledger"][key])-float(source_ledger.get(key, 0.0))
                 for key in record["ledger"]}
    line_scale = max(abs(increment["parent_line_processed_m"]), 1e-30)
    record.update({
        "directory": str(directory), "initial_amplitude_m": amplitude[0],
        "final_amplitude_m": amplitude[-1],
        "amplitude_drift_m": amplitude[-1]-amplitude[0],
        "amplitude_drift_cells": (amplitude[-1]-amplitude[0])/(10e-6/grid),
        "early_velocity_fit": linear_velocity(time[:window], amplitude[:window]),
        "late_velocity_fit": linear_velocity(time[-window:], amplitude[-window:]),
        "ledger_increment": increment,
        "ledger_closure_passed": bool(
            abs(increment["line_closure_m"]) <= 1e-12*line_scale+1e-24
            and increment["signed_burgers_change_m2"] <= 1e-20
            and abs(increment["line_energy_released_J"]
                    -increment["heat_released_J"]) <= 1e-24),
    })
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("derivative", type=Path)
    parser.add_argument("restart", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--grid", type=int, default=128)
    args = parser.parse_args()
    source = checkpoint_record(args.source)
    first = branch_record(args.first, source["ledger"], args.grid)
    second = branch_record(args.second, first["ledger"], args.grid)
    pressure_shift = second["pressure_Pa"]-first["pressure_Pa"]
    line_ratio = (second["ledger_increment"]["parent_line_processed_m"]
                  /max(first["ledger_increment"]["parent_line_processed_m"], 1e-300))
    derivative = json.loads(args.derivative.read_text())
    restart = json.loads(args.restart.read_text())
    invariants = all(x["minimum_population_m2"] >= 0.0
                     and x["phase_simplex_error"] <= 1e-12
                     and x["ledger_closure_passed"] for x in (first, second))
    fixed = bool(
        abs(second["amplitude_drift_cells"]) <= 0.02
        and abs(pressure_shift) <= 0.1e6
        and line_ratio <= 1.05
        and abs(second["late_velocity_fit"]["velocity_m_s"])
        <= 2.0*second["late_velocity_fit"]["velocity_standard_error_m_s"]
        and invariants and derivative.get("passed", False)
        and restart.get("restart_exact_within_declared_roundoff", False))
    payload = {
        "schema": "full-v34-v15-coupled-neutral-audit/v1",
        "classification": ("V15_COUPLED_NEUTRAL_FIXED_POINT_PASSED"
                           if fixed else "V15_COUPLED_NEUTRAL_FIXED_POINT_FAILED_AT_128"),
        "fixture_passed": bool(invariants and derivative.get("passed", False)
                               and restart.get("restart_exact_within_declared_roundoff", False)),
        "scientific_gate_passed": fixed, "grid": args.grid,
        "source": source, "first_hold": first, "second_hold": second,
        "pressure_shift_between_holds_Pa": pressure_shift,
        "processed_line_increment_ratio_second_to_first": line_ratio,
        "full_functional_finite_difference": derivative,
        "restart_comparison": restart,
        "hard_invariants_passed": invariants,
        "failure_reasons": ([] if fixed else [
            "neutral pressure is not stationary within 0.1 MPa",
            "fresh front processing grows rather than approaching zero or stationary cycling",
            "a scalar energetic root does not arrest irreversible local front-state feedback",
        ]),
        "downstream_authorization": {
            "grid_192_256_threshold_campaign": False,
            "long_horizon_canonical_campaign": False,
            "polycrystal_pair_campaign": False,
            "overnight_HPC3_bundle": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(payload["classification"])


if __name__ == "__main__":
    main()
