#!/usr/bin/env python3
"""Classify the zero-pressure, stored-energy-driven V22 SIBM fallback."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

PRODUCTION = Path(__file__).resolve().parents[1] / "production"
if str(PRODUCTION) not in sys.path:
    sys.path.insert(0, str(PRODUCTION))
from moving_front import phase_total_line_densities, state_from_checkpoint  # noqa: E402


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def final_checkpoint(directory: Path) -> Path:
    paths = sorted(directory.glob("drx_v25_restart_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no checkpoint in {directory}")
    return paths[-1]


def case_record(directory: Path) -> dict:
    checkpoint = final_checkpoint(directory)
    rows = list(csv.DictReader((directory/"sibm_contour_diagnostics.csv").open()))
    with np.load(checkpoint, allow_pickle=True) as state:
        experiment = json.loads(str(state["sibm_experiment_json"].item()))
        metadata = json.loads(str(state["sparse_front_metadata_json"].item()))
        arrays = {name.removeprefix("sparse_front__"): state[name]
                  for name in state.files if name.startswith("sparse_front__")}
        front = state_from_checkpoint(
            str(state["sparse_front_metadata_json"].item()), arrays)
        parent_density, child_density = phase_total_line_densities(front)
        parent, child = front.parent_label, front.child_label
        # The canonical contour measurement after the first accepted step is
        # the comparison origin.  The legacy metadata field can refer to the
        # pre-displacement reference contour rather than the imposed cap.
        initial = float(rows[0]["bulge_amplitude_m"])
        final = float(rows[-1]["bulge_amplitude_m"])
        line = metadata["ledger"]
        line_scale = max(abs(float(line["parent_line_processed_m"])), 1e-30)
        closure = bool(
            abs(float(line["line_closure_m"])) <= 1e-12*line_scale+1e-24
            and abs(float(line["signed_burgers_change_m2"])) <= 1e-20
            and abs(float(line["line_energy_released_J"])
                    -float(line["heat_released_J"])) <= 1e-24)
        no_allocation = bool(
            int(state["Ng"]) == int(experiment["initial_phase_count"])
            if "initial_phase_count" in experiment else True)
        return {
            "name": directory.name, "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest(checkpoint),
            "parent_label": parent, "child_label": child,
            "initial_parent_mean_density_m2": float(
                experiment["parent_mean_density_m2"]),
            "initial_child_mean_density_m2": float(
                experiment["child_mean_density_m2"]),
            "final_parent_mean_density_m2": float(np.mean(parent_density)),
            "final_child_mean_density_m2": float(np.mean(child_density)),
            "initial_amplitude_m": initial, "final_amplitude_m": final,
            "amplitude_change_m": final-initial,
            "initial_stored_energy_drive_Pa": float(
                experiment.get("initial_stored_energy_drive_Pa",
                               experiment.get("stored_energy_drive_Pa", np.nan))),
            "final_stored_energy_drive_Pa": float(
                experiment.get("stored_energy_drive_Pa", np.nan)),
            "applied_external_pressure_Pa": float(
                experiment.get("applied_continuation_pressure_Pa", 0.0)),
            "final_strain": float(rows[-1]["strain"]),
            "front_ledger": line, "ledger_passed": closure,
            "no_label_allocation": no_allocation,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    names = (
        "equal_energy_curved", "parent_high_child_low",
        "reversed_density_contrast", "favorable_small_radius",
        "favorable_large_radius", "continued_deformation_rehardening")
    records = {name: case_record(args.root/"cases"/name) for name in names}
    equal = records["equal_energy_curved"]
    favorable = records["parent_high_child_low"]
    reverse = records["reversed_density_contrast"]
    small = records["favorable_small_radius"]
    large = records["favorable_large_radius"]
    deformation = records["continued_deformation_rehardening"]
    zero_pressure = all(abs(item["applied_external_pressure_Pa"]) <= 1e-12
                        for item in records.values())
    hard_invariants = all(item["ledger_passed"] and item["no_label_allocation"]
                          for item in records.values())
    directional = bool(
        equal["amplitude_change_m"] < 0.0
        and favorable["amplitude_change_m"] > equal["amplitude_change_m"]
        and reverse["amplitude_change_m"] < favorable["amplitude_change_m"])
    curvature = large["amplitude_change_m"] > small["amplitude_change_m"]
    rehardening = (deformation["final_child_mean_density_m2"]
                   > 1.01*deformation["initial_child_mean_density_m2"])
    passed = bool(zero_pressure and hard_invariants and directional
                  and curvature and rehardening)
    result = {
        "schema": "asb-drx/v22-zero-pressure-stored-energy-sibm/v1",
        "classification": (
            "ZERO_PRESSURE_STORED_ENERGY_SIBM_PATHWAY_SUPPORTED"
            if passed else "ZERO_PRESSURE_STORED_ENERGY_SIBM_PATHWAY_UNRESOLVED"),
        "fixture_passed": hard_invariants,
        "scientific_gate_passed": passed,
        "external_pressure_is_exactly_zero": zero_pressure,
        "stored_energy_directionality_passed": directional,
        "curvature_competition_passed": curvature,
        "continued_deformation_child_rehardening_passed": rehardening,
        "cases": list(records.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(result["classification"])


if __name__ == "__main__":
    main()
