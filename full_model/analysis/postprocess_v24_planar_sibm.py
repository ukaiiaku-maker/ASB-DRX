#!/usr/bin/env python3
"""Decision-grade classification of the V24 full-driver planar SIBM matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import numpy as np

from full_model.production.moving_front import state_from_checkpoint


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def checkpoints(path):
    return sorted(path.glob("drx_v25_restart_*.npz"), key=lambda item: int(
        re.search(r"(\d+)$", item.stem).group(1)))


def stable_slopes(time, displacement):
    indices = np.array_split(np.arange(len(time)//2, len(time)), 3)
    slopes = [float(np.polyfit(time[index], displacement[index], 1)[0])
              for index in indices if len(index) >= 2]
    same_sign = bool(slopes and (all(x > 0 for x in slopes)
                                or all(x < 0 for x in slopes)))
    return slopes, same_sign


def case_record(directory, interface_width_m):
    rows = list(csv.DictReader((directory/"sibm_contour_diagnostics.csv").open()))
    if len(rows) < 2:
        raise ValueError(f"insufficient contour history in {directory}")
    time = np.array([float(row["time_s"]) for row in rows])
    displacement = np.array([
        float(row["signed_normal_displacement_mean_m"]) for row in rows])
    pressure = np.array([float(row["local_normal_pressure_Pa"]) for row in rows])
    slopes, stable = stable_slopes(time, displacement)
    final = checkpoints(directory)[-1]
    with np.load(final, allow_pickle=True) as data:
        experiment = json.loads(str(data["sibm_experiment_json"].item()))
        metadata = json.loads(str(data["sparse_front_metadata_json"].item()))
        arrays = {name.removeprefix("sparse_front__"): data[name]
                  for name in data.files if name.startswith("sparse_front__")}
        front = state_from_checkpoint(
            str(data["sparse_front_metadata_json"].item()), arrays)
        simplex = float(np.max(np.abs(np.sum(data["eta"], axis=2)-1.0)))
        current_phase_count = int(data["Ng"])
    ledger = metadata["ledger"]
    line_scale = max(abs(float(ledger["parent_line_processed_m"])), 1e-30)
    closure = bool(
        abs(float(ledger["line_closure_m"])) <= 1e-12*line_scale+1e-24
        and float(ledger["signed_burgers_change_m2"]) == 0.0
        and abs(float(ledger["line_energy_released_J"])
                -float(ledger["heat_released_J"])) <= 1e-22)
    delta = float(displacement[-1]-displacement[0])
    return {
        "case": directory.name, "checkpoint": str(final),
        "checkpoint_sha256": digest(final),
        "initial_pressure_Pa": float(pressure[0]),
        "median_pressure_Pa": float(np.median(pressure)),
        "normal_displacement_m": delta,
        "normal_displacement_interface_widths": float(delta/interface_width_m),
        "stable_window_velocities_m_s": slopes,
        "stable_velocity_sign": stable,
        "resolved_quarter_width": bool(
            abs(delta) >= .25*interface_width_m),
        "declared_initial_curvature_relative_to_inverse_width": float(abs(
            float(experiment.get("seed_tip_curvature_m_1", np.inf)))
            *interface_width_m),
        "first_observed_curvature_relative_to_inverse_width": float(abs(
            float(rows[0]["tip_curvature_m-1"]))*interface_width_m),
        "applied_external_pressure_Pa": float(
            experiment.get("applied_continuation_pressure_Pa", 0.0)),
        "front_ledger_passed": closure,
        "phase_simplex_error": simplex,
        "phase_count_unchanged": (
            current_phase_count == int(experiment["initial_phase_count"])),
        "parent_label": front.parent_label, "child_label": front.child_label,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    common = json.loads((args.root/"common_state"/"manifest.json").read_text())
    # Frozen full-model values: sqrt(kappa_eta/W_eta).
    width = np.sqrt(5e-7/5e6)
    names = ("equal", "parent_high_child_low", "reversed", "mobility_off")
    records = {name: case_record(args.root/"cases"/name, width) for name in names}
    equal = records["equal"]; favorable = records["parent_high_child_low"]
    reverse = records["reversed"]; off = records["mobility_off"]
    zero_pressure = all(item["applied_external_pressure_Pa"] == 0.0
                        for item in records.values())
    planar = all(item["declared_initial_curvature_relative_to_inverse_width"] < .01
                 for item in records.values())
    invariants = all(item["front_ledger_passed"] and item["phase_count_unchanged"]
                     and item["phase_simplex_error"] <= 1e-12
                     for item in records.values())
    equal_stationary = bool(
        abs(equal["normal_displacement_interface_widths"]) < .1)
    mobility_stationary = bool(
        abs(off["normal_displacement_interface_widths"]) < .02)
    favorable_advance = bool(
        favorable["initial_pressure_Pa"] > 0.0
        and favorable["normal_displacement_m"] > 0.0
        and favorable["resolved_quarter_width"]
        and favorable["stable_velocity_sign"])
    reverse_retreat = bool(
        reverse["initial_pressure_Pa"] < 0.0
        and reverse["normal_displacement_m"] < 0.0
        and reverse["resolved_quarter_width"]
        and reverse["stable_velocity_sign"])
    sequential_dynamic_activation_completed = False
    passed = bool(zero_pressure and planar and invariants and equal_stationary
                  and mobility_stationary and favorable_advance and reverse_retreat
                  and sequential_dynamic_activation_completed)
    fixture = bool(invariants and zero_pressure and planar
                   and common["zero_cumulative_front_sweep_on_assignment"])
    sign_coupling_failure = bool(
        fixture and (not favorable_advance or not reverse_retreat))
    classification = (
        "FULL_DYNAMIC_ZERO_PRESSURE_PLANAR_SIBM_SUPPORTED" if passed else
        "FULL_DYNAMIC_PLANAR_SIBM_SIGN_OR_COUPLING_FAILURE"
        if sign_coupling_failure else
        "FULL_DYNAMIC_PLANAR_SIBM_FIXTURE_INVALID")
    result = {
        "schema": "asb-drx/v24-full-dynamic-planar-sibm/v1",
        "common_phase_geometry_sha256": common["phase_geometry_sha256"],
        "geometry_identical_across_branches": len({
            item["eta_sha256"] for item in common["variants"]}) == 1,
        "zero_external_pressure": zero_pressure,
        "zero_curvature_within_discrete_tolerance": planar,
        "hard_invariants_passed": invariants,
        "equal_energy_stationary": equal_stationary,
        "mobility_off_stationary": mobility_stationary,
        "favorable_low_defect_child_advances": favorable_advance,
        "reversed_contrast_retreats": reverse_retreat,
        "sign_or_coupling_failure": sign_coupling_failure,
        "sequential_dynamic_activation_completed": (
            sequential_dynamic_activation_completed),
        "cases": list(records.values()),
        "fixture_passed": fixture,
        "scientific_gate_passed": passed,
        "classification": classification,
        "qualification_limit": (
            "The initial equilibration disables front transfer but the legacy "
            "full driver has no exact all-defect freeze switch. The submitted "
            "matrix tests the final all-channel state; the preceding sequential "
            "handoffs remain V23 algebraic fixtures rather than full-driver "
            "stages. These are explicit qualification limitations even if "
            "directionality passes."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(result["classification"])


if __name__ == "__main__":
    main()
