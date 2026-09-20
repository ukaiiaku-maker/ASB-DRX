#!/usr/bin/env python3
"""Extract physical response observables from the retained V46 trajectory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from full_model.analysis.run_v34_finite_coupled_response import resolved_bicrystal
from full_model.analysis.run_v36_recurrent_physical_response import driving_at_time
from full_model.analysis.run_v39_common_horizon import load_stage, stage_diagnostics
from full_model.production.common_tensorial_wall import resolved_driving_components


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_record(state, context, time_s):
    driving = driving_at_time(context["spacing_m"] and state.eta.shape[0],
                              .01, "hold", 0.0, time_s)
    diagnostic = stage_diagnostics(state, context, driving)
    drive = resolved_driving_components(
        state.mechanical.common, driving, context["systems"],
        context["topologies"], context["wall_parameters"])
    beta = state.mechanical.common.beta_p
    symmetric_plastic = .5*(beta+np.swapaxes(beta, -1, -2))
    child = state.eta[..., 1]
    return {
        "physical_time_s": time_s,
        "imposed_mean_shear_tensor_component": float(driving.mean_strain[0, 1]),
        "mean_plastic_shear_tensor_component": float(np.mean(
            symmetric_plastic[..., 0, 1])),
        "maximum_abs_plastic_distortion": float(np.max(np.abs(beta))),
        "mean_shear_stress_Pa": float(np.mean(
            drive["stress_tensor_Pa"][..., 0, 1])),
        "rms_shear_stress_Pa": float(np.sqrt(np.mean(
            drive["stress_tensor_Pa"][..., 0, 1]**2))),
        "child_fraction_mean": float(np.mean(child)),
        "temperature_mean_K": float(np.mean(
            state.mechanical.common.temperature_K)),
        "temperature_minimum_K": diagnostic["temperature_minimum_K"],
        "temperature_maximum_K": diagnostic["temperature_maximum_K"],
        "temperature_peak_minus_mean_K": float(
            diagnostic["temperature_maximum_K"]-np.mean(
                state.mechanical.common.temperature_K)),
        "physical_energy_J": diagnostic["physical_energy"],
        "line_inventory": diagnostic["line_inventory"],
        "nye": diagnostic["nye"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v46-results", type=Path, required=True)
    parser.add_argument("--grid", type=int, choices=(128, 192), default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    n = args.grid
    first_root = args.v46_results/"v46-checkpointed"/f"n{n}"
    long_root = args.v46_results/"v46-original-horizon"/f"n{n}"
    initial_path = first_root/"operation_000_initial.npz"
    final_manifest_path = long_root/"run_manifest.json"
    manifest = json.loads(final_manifest_path.read_text())
    final_path = Path(manifest["latest_checkpoint"])
    if (manifest["terminal_state"] != "COMPLETE"
            or manifest["completed_operation_count"] != 119
            or digest(final_path) != manifest["latest_checkpoint_sha256"]):
        raise ValueError("V46 final checkpoint is incomplete or corrupt")
    context = resolved_bicrystal(
        grid=n, length_m=3.2e-6, interface_width_m=4e-7,
        temperature_K=1100.0, child_line_fraction=.35)
    initial, initial_metadata = load_stage(initial_path, context)
    final, final_metadata = load_stage(final_path, context)
    before = state_record(initial, context, float(initial_metadata["physical_time_s"]))
    after = state_record(final, context, float(final_metadata["physical_time_s"]))
    dx = context["spacing_m"]
    initial_components = initial.front_runtime.topology.components
    final_components = final.front_runtime.topology.components
    displacements = []
    for old, new in zip(initial_components, final_components, strict=True):
        axis = final.front_runtime.normal_axis
        displacements.append(float(
            (new.centroid_grid[axis]-old.centroid_grid[axis])*dx))
    ledger = final.front_runtime.ledger
    energy_delta = {
        key: float(after["physical_energy_J"][key]
                   -before["physical_energy_J"][key])
        for key in before["physical_energy_J"]}
    temperature = final.mechanical.common.temperature_K
    heat_capacity = context["wall_parameters"].volumetric_heat_capacity_J_m3_K
    thickness = context["represented_thickness_m"]
    thermal_rise_J = float(np.sum(
        heat_capacity*(temperature-1100.0))*dx*dx*thickness)
    payload = {
        "schema": "asb-drx/v47/v46-physical-response/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "retained_scientific_source_sha": manifest["configuration"][
            "parent_scientific_source_sha"],
        "grid": n,
        "domain_m": 3.2e-6,
        "represented_thickness_m": thickness,
        "initial": before,
        "final": after,
        "changes": {
            "mean_plastic_shear_tensor_component": (
                after["mean_plastic_shear_tensor_component"]
                -before["mean_plastic_shear_tensor_component"]),
            "mean_shear_stress_Pa": (
                after["mean_shear_stress_Pa"]-before["mean_shear_stress_Pa"]),
            "child_fraction_mean": (
                after["child_fraction_mean"]-before["child_fraction_mean"]),
            "physical_energy_J": energy_delta,
            "thermal_rise_from_temperature_field_J": thermal_rise_J,
        },
        "front_response": {
            "attempts": ledger.attempts,
            "accepted": ledger.accepted,
            "component_contour_displacements_m": displacements,
            "maximum_abs_component_contour_displacement_m": max(
                abs(x) for x in displacements),
            "interface_width_m": 4e-7,
            "maximum_displacement_over_interface_width": max(
                abs(x) for x in displacements)/4e-7,
            "a_to_b_swept_volume_m3": ledger.a_to_b_swept_volume_m3,
            "b_to_a_swept_volume_m3": ledger.b_to_a_swept_volume_m3,
            "processed_line_m": (
                ledger.a_to_b_line_processed_m+ledger.b_to_a_line_processed_m),
            "transmitted_line_m": ledger.transmitted_line_m,
            "boundary_line_m": ledger.boundary_line_m,
            "annihilated_line_m": ledger.annihilated_line_m,
            "sink_line_m": ledger.sink_line_m,
            "irreversible_front_heat_J": ledger.heat_J,
            "maximum_abs_line_closure_m": ledger.maximum_abs_line_closure_m,
        },
        "classification": (
            "VALID_TINY_EXISTING_BOUNDARY_MIGRATION_WITH_BULK_RECOVERY"
            if ledger.accepted == 8 and max(abs(x) for x in displacements) < .01*4e-7
            else "PHYSICAL_RESPONSE_REQUIRES_REVIEW"),
        "spontaneous_nucleation_claimed": False,
        "lagb_claimed": False,
        "drx_claimed": False,
        "strict_asb_claimed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps({
        "classification": payload["classification"],
        "maximum_displacement_m": payload["front_response"][
            "maximum_abs_component_contour_displacement_m"],
        "temperature_rise_K": after["temperature_mean_K"]-1100.0,
        "plastic_shear": payload["changes"][
            "mean_plastic_shear_tensor_component"],
        "sha256": digest(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
