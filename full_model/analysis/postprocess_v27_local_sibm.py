#!/usr/bin/env python3
"""Force/history/resolution classification for local V27 SIBM matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from full_model.production.asb_grid_audit import relative_difference


CASES = ("equal", "favorable", "reversed", "mobility_off", "label_swapped")


def fit_velocity(time, displacement, count=3):
    count = min(int(count), len(time))
    if count < 2:
        return 0.0
    return float(np.polyfit(time[:count], displacement[:count], 1)[0])


def case_record(directory, grid):
    contour = pd.read_csv(directory/"sibm_contour_diagnostics.csv")
    time = contour["time_s"].to_numpy(float)
    displacement = contour["signed_normal_displacement_mean_m"].to_numpy(float)
    pressure = contour["local_normal_pressure_Pa"].to_numpy(float)
    velocity = contour["area_equivalent_normal_velocity_m_s"].to_numpy(float)
    terminal_path = directory/"sibm_terminal_event.json"
    terminal = json.loads(terminal_path.read_text()) if terminal_path.exists() else None
    checkpoint = sorted(directory.glob("drx_v25_restart_*.npz"))[-1]
    with np.load(checkpoint, allow_pickle=True) as data:
        p = json.loads(str(data["P_json"].item()))
        front = json.loads(str(data["sparse_front_metadata_json"].item()))
        experiment = json.loads(str(data["sibm_experiment_json"].item()))
    ledger = front["ledger"]
    interface_width = float(np.sqrt(float(p["kappa_eta"])/float(p["W_eta"])))
    dx = float(p["L_phys"])/grid
    signed_area = float(np.sum(
        contour["parent_to_child_contour_area_m2"]
        -contour["child_to_parent_contour_area_m2"]))
    length = np.maximum(
        contour["pair_local_interface_length_m"].to_numpy(float), dx)
    material_displacement = np.cumsum(
        contour["parent_to_child_contour_area_m2"].to_numpy(float)
        -contour["child_to_parent_contour_area_m2"].to_numpy(float))/length
    active = (np.abs(pressure) > max(1e-6*np.max(np.abs(pressure)), 1e-6))
    active &= np.abs(velocity) > max(1e-6*np.max(np.abs(velocity)), 1e-12)
    sign_match = (float(np.mean(np.sign(pressure[active]) == np.sign(velocity[active])))
                  if np.any(active) else 1.0)
    reversals = np.flatnonzero(np.signbit(pressure[1:]) != np.signbit(pressure[:-1]))+1
    swept = float(ledger["swept_volume_m3"])
    return {
        "grid": grid, "interface_width_m": interface_width,
        "cell_size_m": dx, "points_per_interface_width": interface_width/dx,
        "resolved_at_four_points": bool(interface_width/dx >= 4.0),
        "time_s": time.tolist(), "accepted_timestep_s": np.diff(time).tolist(),
        "fixed_frame_displacement_m": displacement.tolist(),
        "material_frame_displacement_m": material_displacement.tolist(),
        "complete_variational_pressure_Pa": pressure.tolist(),
        "area_velocity_m_s": velocity.tolist(),
        "initial_velocity_fit_m_s": fit_velocity(time, material_displacement),
        "late_velocity_m_s": float(np.median(velocity[-min(3, len(velocity)):])),
        "cumulative_signed_contour_area_m2": signed_area,
        "drive_reversal_times_s": time[reversals].tolist(),
        "simultaneous_force_velocity_sign_match_fraction": sign_match,
        "line_processed_per_swept_volume_m2": (
            float(ledger["parent_line_processed_m"])/swept if swept > 0.0 else 0.0),
        "heat_per_swept_volume_J_m3": (
            float(ledger["heat_released_J"])/swept if swept > 0.0 else 0.0),
        "front_ledger": ledger, "terminal_event": terminal,
        "equal_state_projection_activations": int(
            experiment.get("equal_state_projection_activations", 0)),
        "equal_state_suppressed_abs_volume_m3": float(
            experiment.get("equal_state_suppressed_abs_volume_m3", 0.0)),
        "equal_state_suppressed_signed_volume_m3": float(
            experiment.get("equal_state_suppressed_signed_volume_m3", 0.0)),
        "complete_derivative_decomposition": {
            "stored_energy_Pa": experiment.get("stored_energy_drive_Pa"),
            "applied_pressure_Pa": experiment.get("applied_continuation_pressure_Pa"),
            "interface_owned_compatibility_Pa": experiment.get(
                "physical_compatibility_pressure_Pa"),
            "interface_owned_drag_Pa": experiment.get("physical_drag_pressure_Pa"),
            "net_flat_boundary_drive_Pa": experiment.get("net_flat_boundary_drive_Pa"),
        },
    }


def grid_record(root, grid):
    return {name: case_record(root/f"s5-{name}", grid) for name in CASES}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid64", type=Path)
    parser.add_argument("--grid128", type=Path, required=True)
    parser.add_argument("--grid192", type=Path)
    parser.add_argument("--symmetry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    grids = {"128": grid_record(args.grid128, 128)}
    if args.grid192 is not None:
        grids["192"] = grid_record(args.grid192, 192)
    if args.grid64 is not None:
        grids["64"] = grid_record(args.grid64, 64)
    symmetry = json.loads(args.symmetry.read_text())
    metric_names = ("initial_velocity_fit_m_s", "late_velocity_m_s",
                    "line_processed_per_swept_volume_m2",
                    "heat_per_swept_volume_J_m3")
    convergence = {}
    if "192" in grids:
        for case in CASES:
            convergence[case] = {name: relative_difference(
                grids["128"][case][name], grids["192"][case][name])
                for name in metric_names}
            convergence[case]["all_within_5pct"] = all(
                convergence[case][name] <= .05 for name in metric_names)
    controls = bool(
        grids["128"]["equal"]["front_ledger"]["swept_volume_m3"] == 0.0
        and grids["128"]["favorable"]["cumulative_signed_contour_area_m2"] > 0.0
        and grids["128"]["reversed"]["cumulative_signed_contour_area_m2"] < 0.0
        and grids["128"]["mobility_off"]["front_ledger"]["swept_volume_m3"] == 0.0
        and ("192" not in grids or (
            grids["192"]["equal"]["front_ledger"]["swept_volume_m3"] == 0.0
            and grids["192"]["favorable"]["cumulative_signed_contour_area_m2"] > 0.0
            and grids["192"]["reversed"]["cumulative_signed_contour_area_m2"] < 0.0
            and grids["192"]["mobility_off"]["front_ledger"]["swept_volume_m3"] == 0.0)))
    resolved_converged = bool("192" in grids and controls and all(
        row["all_within_5pct"] for row in convergence.values()))
    projection_localized = bool(symmetry.get(
        "equal_state_projection_is_machine_symmetry_only", False))
    odd = bool(symmetry.get("near_equal_response_continuous_and_odd", False))
    sign_consistent = all(
        grids[g][case]["simultaneous_force_velocity_sign_match_fraction"] >= .95
        for g in ("128", "192") if g in grids
        for case in ("favorable", "reversed"))
    if not projection_localized or not odd:
        classification = "SIBM_EQUAL_STATE_PROJECTION_DEPENDENT"
    elif not sign_consistent:
        classification = "SIBM_FRONT_MEMORY_OR_COUPLING_SIGN_FAILURE"
    elif not resolved_converged:
        classification = "SIBM_GRID_RESOLUTION_UNQUALIFIED"
    else:
        classification = "GENERIC_ZERO_PRESSURE_STORED_ENERGY_SIBM_MECHANISM_QUALIFIED"
    result = {
        "schema": "asb-drx/v27-local-sibm-decision/v1",
        "minimum_points_per_interface_width": 4.0,
        "grids": grids, "near_equal_symmetry": symmetry,
        "resolved_128_192_relative_differences": convergence,
        "resolved_192_executed": "192" in grids,
        "directional_controls_passed": controls,
        "resolved_grids_converged_5pct": resolved_converged,
        "force_velocity_sign_consistent": sign_consistent,
        "classification": classification,
        "fixture_passed": True,
        "scientific_gate_passed": bool(classification.startswith("GENERIC_")),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(classification)


if __name__ == "__main__":
    main()
