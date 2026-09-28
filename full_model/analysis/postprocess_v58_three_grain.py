#!/usr/bin/env python3
"""Outcome-neutral physical classification of a V58 three-grain trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from full_model.analysis.run_v58_three_grain_production import (
    _load_checkpoint, initialize_state,
)
from full_model.production.multigrain_common_state import (
    audit_multigrain_nye, reconstruct_multigrain_common,
)
from full_model.production.common_tensorial_wall import (
    CommonWallDriving, CommonWallParameters,
)
from full_model.production.multigrain_production import (
    multigrain_instantaneous_dissipation_fields,
)
from full_model.production.tensorial_nye import (
    bcc_four_family_systems, spectral_derivatives,
)
from full_model.analysis.spatial_localization import spatial_localization_metrics


def _perimeter_m(support, spacing):
    gx, gy = spectral_derivatives(support, spacing)
    return float(np.sum(np.sqrt(gx*gx+gy*gy), dtype=np.longdouble)*spacing**2)


def _total_density(owner):
    return np.sum(
        owner.mobile_plus_m2+owner.mobile_minus_m2
        +owner.forest_plus_m2+owner.forest_minus_m2
        +owner.wall_plus_m2+owner.wall_minus_m2, axis=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--source-commit")
    parser.add_argument("--parent-run")
    args = parser.parse_args()
    root = Path(args.run)
    history = json.loads((root/"history.json").read_text())
    if args.parent_run:
        parent = json.loads((Path(args.parent_run)/"history.json").read_text())
        first_step = int(history[0]["step"]) if history else 10**30
        history = [item for item in parent if int(item["step"]) < first_step]+history
    complete = (root/"result.json").exists()
    if complete:
        run_result = json.loads((root/"result.json").read_text())
        checkpoint = Path(run_result["checkpoint"])
    else:
        checkpoint = sorted(root.glob("checkpoint_*.npz"))[-1]
        (_, _, _, _, _, configuration, provenance) = _load_checkpoint(checkpoint)
        if configuration is None:
            raise ValueError("partial checkpoint has no bound configuration")
        n = int(configuration["n"])
        spacing = float(configuration["length_m"])/n
        run_result = {
            "case": configuration["case"], "n": n,
            "temperature_K": configuration["temperature_K"],
            "spacing_m": spacing,
            "represented_thickness_m": 2.0*2.48e-10,
            "interface_width_m": configuration["interface_width_m"],
            "source_commit": (args.source_commit
                              or (provenance or {}).get("source_commit")
                              or "UNRECORDED_PARTIAL_SOURCE"),
            "final": history[-1], "checkpoint": str(checkpoint),
        }
    (state, runtime, step, gamma, initial_volume,
     checkpoint_configuration, checkpoint_provenance) = _load_checkpoint(checkpoint)
    spacing = float(run_result["spacing_m"])
    thickness = float(run_result["represented_thickness_m"])
    initial = initialize_state(
        run_result["n"], run_result["temperature_K"],
        run_result["case"] == "equal_density", spacing_m=spacing,
        interface_width_m=run_result["interface_width_m"])
    cell_volume = spacing**2*thickness
    final_volume = np.sum(state.supports, axis=(1, 2))*cell_volume
    support_delta = state.supports-initial.supports
    gross_transformed = float(
        .5*np.sum(np.abs(support_delta), dtype=np.longdouble)*cell_volume)
    fresh = state.ledger.fresh_sweep_fraction*cell_volume
    revisit = state.ledger.revisit_sweep_fraction*cell_volume
    common, _ = reconstruct_multigrain_common(state, spacing)
    total_density = _total_density(common)
    temperature = np.asarray(common.temperature_K)
    orientation = np.asarray(common.orientation_rad)
    audit = audit_multigrain_nye(state, spacing)
    gnd = np.linalg.norm(audit.exact_reconstructed_m1, axis=(-2, -1))
    owner_mismatch = np.linalg.norm(
        audit.owner_reservoir_mismatch_m1, axis=(-2, -1))
    systems = bcc_four_family_systems()
    flow_mode = ((checkpoint_configuration or {}).get(
        "flow_temperature_mode", "physical"))
    wall = CommonWallParameters(
        spacing_m=spacing, elastic_iterations=2,
        mobile_correlation_diffusivity_m2_s=0.0,
        transport_scheme="upwind", maximum_fraction_per_step=.75,
        flow_temperature_override_K=(
            run_result["temperature_K"] if flow_mode == "frozen" else None),
        volumetric_heat_capacity_J_m3_K=3.8e6,
        thermal_diffusivity_m2_s=(checkpoint_configuration or {}).get(
            "thermal_diffusivity_m2_s", 0.15/3.8e6), bath_rate_s=0.0)
    strain = np.array([[0.0, .5*gamma], [.5*gamma, 0.0]])
    dissipation = multigrain_instantaneous_dissipation_fields(
        state, driving=CommonWallDriving(mean_strain=strain), systems=systems,
        topologies=(), wall_parameters=wall)
    perimeters_initial = np.asarray([
        _perimeter_m(value, spacing) for value in initial.supports])
    perimeters_final = np.asarray([
        _perimeter_m(value, spacing) for value in state.supports])
    equivalent_displacement = np.divide(
        final_volume-initial_volume,
        thickness*.5*(perimeters_initial+perimeters_final),
        out=np.zeros_like(final_volume),
        where=(perimeters_initial+perimeters_final) > 0.0)
    newly_swept_density = []
    for index, owner in enumerate(state.owners):
        growth = np.maximum(support_delta[index], 0.0)
        density = _total_density(owner)
        newly_swept_density.append(float(np.sum(growth*density)
            /max(float(np.sum(growth)), 1e-300)))
    low_index = 1
    low_growth_fraction = float(
        (final_volume[low_index]-initial_volume[low_index])
        /max(initial_volume[low_index], 1e-300))
    substantial_drx = bool(
        low_growth_fraction >= .01
        and equivalent_displacement[low_index] >= .5*spacing
        and fresh > 0.0)
    thermal_localization_ratio = float(
        (np.max(temperature)-np.min(temperature))
        /max(np.mean(temperature)-run_result["temperature_K"], 1e-12))
    peak_stress = max(item["shear_stress_Pa"] for item in history)
    consecutive = maximum_consecutive = 0
    candidate_steps = []
    for item in history:
        spatial = item.get("localization", {})
        plastic = spatial.get("plastic_power", {})
        heat = spatial.get("irreversible_heat_rate", {})
        candidate = bool(
            plastic.get("band_like", False)
            and heat.get("band_like", False)
            and plastic.get("maximum_to_mean", 0.0) >= 5.0
            and item["mechanical_energy"]["relative_first_law_residual"] <= .05)
        if candidate:
            consecutive += 1; candidate_steps.append(int(item["step"]))
            maximum_consecutive = max(maximum_consecutive, consecutive)
        else:
            consecutive = 0
    thermal_threshold_met = any(
        item["temperature_contrast_K"] >= 100.0 for item in history)
    post_peak_softening = any(
        item["shear_stress_Pa"] < peak_stress for item in history
        if int(item["step"]) >= candidate_steps[0]) if candidate_steps else False
    persistent_candidate = bool(
        maximum_consecutive >= 3 and thermal_threshold_met
        and post_peak_softening)
    classification = {
        "schema": "asb-drx-v58-three-grain-classification-v1",
        "trajectory_complete": complete,
        "source_commit": run_result["source_commit"],
        "checkpoint": str(checkpoint),
        "step": step, "physical_time_s": runtime.ledger.physical_time_s,
        "applied_shear_strain": gamma,
        "initial_volume_m3": initial_volume.tolist(),
        "final_volume_m3": final_volume.tolist(),
        "net_volume_change_m3": (final_volume-initial_volume).tolist(),
        "gross_transformed_volume_m3": gross_transformed,
        "fresh_sweep_volume_m3": fresh,
        "revisit_sweep_volume_m3": revisit,
        "equivalent_contour_displacement_m": equivalent_displacement.tolist(),
        "newly_swept_owner_density_m2": newly_swept_density,
        "temperature_mean_K": float(np.mean(temperature)),
        "temperature_contrast_K": float(np.max(temperature)-np.min(temperature)),
        "thermal_localization_ratio": thermal_localization_ratio,
        "maximum_gnd_m1": float(np.max(gnd)),
        "maximum_owner_nye_mismatch_m1": float(np.max(owner_mismatch)),
        "instantaneous_plastic_power": spatial_localization_metrics(
            dissipation["plastic_power_W_m3"]),
        "instantaneous_irreversible_heat_rate": spatial_localization_metrics(
            dissipation["irreversible_heat_rate_W_m3"]),
        "peak_shear_stress_Pa": peak_stress,
        "asb_candidate_steps": candidate_steps,
        "maximum_consecutive_asb_candidate_steps": maximum_consecutive,
        "asb_temperature_contrast_threshold_met": thermal_threshold_met,
        "post_peak_softening_observed": post_peak_softening,
        "persistent_asb_trajectory_candidate": persistent_candidate,
        "maximum_energy_closure_relative": (
            runtime.ledger.maximum_relative_energy_closure),
        "energy_qualified_multigrain_production": bool(
            runtime.ledger.accepted_events > 0
            and runtime.ledger.rejected_events == 0
            and runtime.ledger.maximum_relative_energy_closure <= .05),
        "substantial_existing_boundary_drx": substantial_drx,
        "strict_asb": False,
        "strict_asb_missing_requirements": [
            item for item, missing in (
                ("three-step persistence", not persistent_candidate),
                ("completed physical horizon", not complete),
                ("matched frozen-flow/front controls", True),
                ("selected spatial/time refinement", True),
            ) if missing],
        "spontaneous_grain_birth": False,
        "claim_limit": (
            "Prepared three-grain existing-boundary trajectory; no spontaneous "
            "nucleation claim. Strict ASB also requires persistence and "
            "observable-specific refinement beyond this endpoint screen."),
    }
    classification_name = ("classification.json" if complete
                           else "classification_partial.json")
    (root/classification_name).write_text(json.dumps(classification, indent=2))

    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    fields = (
        (np.argmax(state.supports, axis=0), "dominant grain", "tab10"),
        (state.supports[1], "low-defect grain support", "viridis"),
        (total_density, "total dislocation density [m$^{-2}$]", "magma"),
        (gnd, "Nye/GND norm [m$^{-1}$]", "inferno"),
        (orientation, "orientation [rad]", "twilight"),
        (temperature, "temperature [K]", "plasma"),
    )
    for axis, (field, title, cmap) in zip(axes.flat, fields):
        image = axis.imshow(field, origin="lower", cmap=cmap)
        axis.set_title(title); axis.set_xticks(()); axis.set_yticks(())
        figure.colorbar(image, ax=axis, shrink=.75)
    figure.savefig(root/("final_fields.png" if complete
                         else "latest_fields.png"), dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    dissipation_fields = (
        (dissipation["plastic_power_W_m3"], "plastic power [W m$^{-3}$]", "magma"),
        (dissipation["irreversible_heat_rate_W_m3"],
         "irreversible heat rate [W m$^{-3}$]", "inferno"),
        (np.sum(np.abs(common.slip), axis=2), "accumulated |slip|", "viridis"),
        (common.wall_order, "wall order", "cividis"),
        (owner_mismatch, "owner Nye mismatch [m$^{-1}$]", "plasma"),
        (temperature, "temperature [K]", "coolwarm"),
    )
    for axis, (field, title, cmap) in zip(axes.flat, dissipation_fields):
        image = axis.imshow(field, origin="lower", cmap=cmap)
        axis.set_title(title); axis.set_xticks(()); axis.set_yticks(())
        figure.colorbar(image, ax=axis, shrink=.75)
    figure.savefig(root/("final_dissipation.png" if complete
                         else "latest_dissipation.png"), dpi=180)
    plt.close(figure)

    strain_history = np.asarray(
        [item["applied_shear_strain"] for item in history])
    stress_history = np.asarray([item["shear_stress_Pa"] for item in history])
    mean_temperature_history = np.asarray(
        [item["temperature_mean_K"] for item in history])
    contrast_history = np.asarray(
        [item["temperature_contrast_K"] for item in history])
    low_growth_history = np.asarray([
        item["grain_volume_change_m3"][low_index]/initial_volume[low_index]
        for item in history])
    energy_error_history = np.asarray([
        item["mechanical_energy"]["relative_first_law_residual"]
        for item in history])
    power_ratio_history = np.asarray([
        item.get("localization", {}).get("plastic_power", {}).get(
            "maximum_to_mean", np.nan) for item in history])
    band_history = np.asarray([
        item.get("localization", {}).get("plastic_power", {}).get(
            "band_like", False) for item in history], dtype=float)
    figure, axes = plt.subplots(3, 2, figsize=(11, 10), constrained_layout=True)
    series = (
        (stress_history/1e9, "shear stress [GPa]"),
        (mean_temperature_history, "mean temperature [K]"),
        (contrast_history, "temperature contrast [K]"),
        (low_growth_history, "low-defect grain volume change / initial"),
        (power_ratio_history, "plastic-power max / mean"),
        (energy_error_history, "relative first-law residual"),
    )
    for axis, (values, label) in zip(axes.flat, series):
        axis.plot(strain_history, values, marker="o", ms=2)
        if label == "plastic-power max / mean":
            axis.scatter(strain_history[band_history > 0.5],
                         values[band_history > 0.5], color="tab:red", s=12,
                         label="band-like")
            axis.legend(loc="best", fontsize=8)
        axis.set_xlabel("applied shear strain")
        axis.set_ylabel(label); axis.grid(alpha=.25)
    figure.savefig(root/("trajectory.png" if complete
                         else "trajectory_partial.png"), dpi=180)
    plt.close(figure)
    print(json.dumps(classification, indent=2))


if __name__ == "__main__":
    main()
