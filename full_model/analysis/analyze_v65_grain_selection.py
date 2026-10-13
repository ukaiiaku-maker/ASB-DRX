#!/usr/bin/env python3
"""Track a prepared grain with one registered geometric convention."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from full_model.analysis.grain_response_metrics import grain_response
from full_model.analysis.run_v58_three_grain_production import _load_checkpoint
from full_model.analysis.run_v64_adaptive_production import (
    _context, _controller_from_checkpoint, _physical_configuration,
)
from full_model.production.complete_multigrain_energy import (
    evaluate_complete_multigrain_energy,
)
from full_model.production.multigrain_production import advance_multigrain_front


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--history", type=Path, action="append", default=[])
    parser.add_argument("--grain-id", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--joint-front-diagnostics", action="store_true")
    args = parser.parse_args()
    initial, initial_runtime, initial_step, initial_load, _, config, _ = (
        _load_checkpoint(args.origin))
    physical = (config["physical_configuration"] if config and config.get(
        "schema") == "asb-drx-v64-adaptive-production-v1" else config)
    spacing = float(physical["length_m"])/int(physical["n"])
    thickness = 2.0*2.48e-10
    context = _context(_physical_configuration(physical))
    try:
        grain_index = tuple(initial.grain_ids).index(args.grain_id)
    except ValueError as error:
        raise ValueError("requested grain is absent from origin") from error
    rows = []
    for checkpoint in args.checkpoint:
        state, runtime, step, load, *_ = _load_checkpoint(checkpoint)
        if tuple(state.grain_ids) != tuple(initial.grain_ids):
            raise ValueError("grain identities changed across tracked checkpoints")
        response = grain_response(
            initial.supports, state.supports, spacing_m=spacing,
            represented_thickness_m=thickness)
        energy = evaluate_complete_multigrain_energy(
            state, spacing_m=spacing, represented_thickness_m=thickness,
            wall_parameters=context.wall_parameters,
            **dict(context.energy_kwargs, mean_strain=np.array(
                [[0.0, .5*load], [.5*load, 0.0]])))
        energy_components = asdict(energy)
        energy_components.update(
            helmholtz_J=energy.helmholtz_J, internal_J=energy.internal_J)
        owner = state.owners[grain_index]
        total_density = sum(
            np.sum(np.asarray(getattr(owner, plus))
                   +np.asarray(getattr(owner, minus)), axis=2)
            for plus, minus in (
                ("mobile_plus_m2", "mobile_minus_m2"),
                ("forest_plus_m2", "forest_minus_m2"),
                ("wall_plus_m2", "wall_minus_m2")))
        support = np.asarray(state.supports[grain_index], dtype=float)
        supported_density = float(np.sum(support*total_density)
                                  /max(np.sum(support), 1e-300))
        joint_front = None
        if args.joint_front_diagnostics:
            controller = _controller_from_checkpoint(checkpoint)
            diagnostic_dt = (controller.proposed_step_s if controller else
                             float(physical["dt_s"]))
            _, _, decision = advance_multigrain_front(
                state, runtime, kinetics=context.front_kinetics,
                dt_s=diagnostic_dt, spacing_m=context.spacing_m,
                represented_thickness_m=context.represented_thickness_m,
                wall_parameters=context.wall_parameters,
                energy_kwargs=dict(context.energy_kwargs, mean_strain=np.array(
                    [[0.0, .5*load], [.5*load, 0.0]])),
                applied_shear_rate_s=context.shear_rate_s,
                systems=context.systems,
                heat_deposition_mode=context.heat_deposition_mode)
            incident = [key for key, interface in (
                (item.component_id, item) for item in runtime.interfaces)
                if args.grain_id in (interface.grain_a_id,
                                     interface.grain_b_id)]
            joint_front = {
                "diagnostic_interval_s": diagnostic_dt,
                "classification": decision.classification,
                "incident_interfaces": incident,
                "direction": {key: decision.direction_by_interface.get(key)
                              for key in incident},
                "independent_pressure_Pa": {
                    key: decision.independent_pressure_by_interface_Pa.get(key)
                    for key in incident},
                "joint_pressure_Pa": {
                    key: decision.pressure_by_interface_Pa.get(key)
                    for key in incident},
                "selected_velocity_m_s": {
                    key: decision.selected_velocity_by_interface_m_s.get(key)
                    for key in incident},
                "accepted_fraction": {
                    key: decision.accepted_fraction_by_interface.get(key, 0.0)
                    for key in incident},
                "joint_pressure_factor": decision.joint_pressure_factor,
                "backtracks": decision.backtracks,
                "force_rate_closure": (
                    decision.force_rate_closure_classification),
            }
        rows.append({
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256(checkpoint),
            "adaptive_interval": int(step),
            "physical_time_s": runtime.ledger.physical_time_s,
            "applied_shear_strain": float(load),
            "grain_id": args.grain_id,
            "volume_growth_fraction": float(
                response["volume_growth_fraction"][grain_index]),
            "equivalent_contour_displacement_m": float(
                response["equivalent_contour_displacement_m"][grain_index]),
            "equivalent_contour_displacement_cells": float(
                response["equivalent_contour_displacement_m"][grain_index]
                /spacing),
            "initial_perimeter_only_displacement_cells": float(
                response["initial_perimeter_only_displacement_m"][grain_index]
                /spacing),
            "initial_perimeter_m": float(
                response["initial_perimeter_m"][grain_index]),
            "final_perimeter_m": float(
                response["final_perimeter_m"][grain_index]),
            "fresh_sweep_volume_m3": float(
                state.ledger.fresh_sweep_fraction*spacing**2*thickness),
            "revisit_sweep_volume_m3": float(
                state.ledger.revisit_sweep_fraction*spacing**2*thickness),
            "cumulative_processed_volume_m3": float(
                state.ledger.accepted_material_fraction*spacing**2*thickness),
            "target_owner_unweighted_density_mean_m2": float(
                np.mean(total_density)),
            "target_owner_support_weighted_density_mean_m2": supported_density,
            "complete_energy_J": energy_components,
            "joint_front_diagnostic": joint_front,
        })
    rows.sort(key=lambda row: row["physical_time_s"])
    for index, row in enumerate(rows):
        if index == 0:
            row["incremental_displacement_velocity_m_s"] = None
            row["growth_trend"] = "ORIGIN"
            continue
        previous = rows[index-1]
        dt = row["physical_time_s"]-previous["physical_time_s"]
        velocity = ((row["equivalent_contour_displacement_m"]
                     -previous["equivalent_contour_displacement_m"])/dt)
        row["incremental_displacement_velocity_m_s"] = velocity
        row["growth_trend"] = ("ADVANCING" if velocity > 0.0 else
                               "RECEDING" if velocity < 0.0 else "STATIONARY")
    history_by_interval = {}
    for path in args.history:
        for item in json.loads(path.read_text()):
            index = int(item["adaptive_interval"])
            if index in history_by_interval and history_by_interval[index] != item:
                raise ValueError(f"conflicting history at interval {index}")
            history_by_interval[index] = item
    volume_history = []
    initial_volume = float(
        np.sum(initial.supports[grain_index])*spacing**2*thickness)
    for index in sorted(history_by_interval):
        item = history_by_interval[index]
        volume = float(item["grain_volume_m3"][grain_index])
        volume_history.append({
            "adaptive_interval": index,
            "physical_time_s": float(item["physical_time_s"]),
            "volume_m3": volume,
            "volume_growth_fraction": (volume-initial_volume)/initial_volume,
        })
    volume_extrema = None
    if volume_history:
        maximum = max(volume_history, key=lambda item: item["volume_m3"])
        passages = [item for item in volume_history
                    if item["volume_growth_fraction"] >= .01]
        volume_extrema = {
            "history_row_count": len(volume_history),
            "first_one_percent_growth": passages[0] if passages else None,
            "maximum_attained": maximum,
            "last_history_row": volume_history[-1],
        }
    payload = {
        "schema": "asb-drx-v65-grain-selection-history-v1",
        "origin": {"path": str(args.origin.resolve()),
                   "sha256": sha256(args.origin),
                   "step": int(initial_step),
                   "physical_time_s": initial_runtime.ledger.physical_time_s,
                   "applied_shear_strain": float(initial_load)},
        "grain_id": args.grain_id,
        "spacing_m": spacing,
        "displacement_convention": response["convention"],
        "volume_history_extrema": volume_extrema,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output),
                      "rows": len(rows)}))


if __name__ == "__main__":
    main()
