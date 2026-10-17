#!/usr/bin/env python3
"""Continue a connected high-rate power band with the qualified RX pathway."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from full_model.analysis.run_v58_three_grain_production import (
    _load_checkpoint, _save_checkpoint,
)
from full_model.analysis.run_v66_coupled_rx_asb import _physics
from full_model.analysis.run_v67_interface_structural_bridge import (
    _args_from_configuration,
)
from full_model.analysis.spatial_localization import spatial_localization_metrics
from full_model.production.coupled_multigrain_interval import (
    advance_complete_multigrain_interval, make_endpoint,
)
from full_model.production.interface_structural_absorption import (
    InterfaceStructuralAbsorptionParameters,
)
from full_model.production.structural_interface_network import (
    JunctionReactionParameters,
)
from full_model.production.lattice_interface_absorption import (
    LatticeInterfaceAbsorptionParameters,
)
from full_model.production.wall_orientation_relaxation import (
    WallOrientationRelaxationParameters,
)
from full_model.production.implicit_wall_order_relaxation import (
    ImplicitWallOrderRelaxationParameters,
)
from full_model.production.conservative_wall_glide import (
    ConservativeWallGlideParameters,
)
from full_model.production.conservative_dipole_wall_organization import (
    ConservativeDipoleWallParameters,
)
from full_model.production.arrhenius_dislocation_sources import (
    ArrheniusSourceParameters,
)
from full_model.production.multigrain_source_transaction import (
    NeutralBoundarySinkParameters,
)


ROOT = Path(__file__).resolve().parents[2]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _largest_periodic_component(mask):
    value = np.asarray(mask, dtype=bool)
    unseen = set(map(tuple, np.argwhere(value)))
    largest = set()
    nx, ny = value.shape
    while unseen:
        component = {unseen.pop()}
        queue = list(component)
        while queue:
            i, j = queue.pop()
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (1, -1), (-1, 1), (-1, -1)):
                neighbor = ((i+di) % nx, (j+dj) % ny)
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        if len(component) > len(largest):
            largest = component
    result = np.zeros_like(value)
    for cell in largest:
        result[cell] = True
    return result


def _band(endpoint):
    power = np.maximum(np.asarray(
        endpoint.observables["instantaneous_plastic_power_W_m3"]), 0.0)
    threshold = 2.0*float(np.mean(power))
    mask = power >= threshold if threshold > 0.0 else np.zeros_like(power, bool)
    return power, _largest_periodic_component(mask), spatial_localization_metrics(
        power, threshold_multiple=2.0)


def _flow_exposure(endpoint, power, shear_rate_s):
    """Fraction of imposed shear power carried by resolved plastic flow."""
    stress = np.asarray(endpoint.observables["common_stress_tensor_Pa"])
    imposed = abs(float(np.mean(stress[..., 0, 1]))*float(shear_rate_s))
    return float(np.mean(np.maximum(power, 0.0))/max(
        imposed, np.finfo(float).tiny))


def _physical_band(metrics, flow_exposure_fraction, threshold=.01):
    return bool(metrics["band_like"] and flow_exposure_fraction >= threshold)


def _production_source_mode(production_rx, requested_mode):
    """Resolve source geography without changing the source kinetics."""
    if not production_rx:
        if requested_mode != "gb_tj_only":
            raise ValueError(
                "a source-geography control requires --production-rx")
        return "disabled"
    return requested_mode


def _interface_emission_mode(production_rx, enabled):
    if enabled and not production_rx:
        raise ValueError("interface mobile emission requires --production-rx")
    return "arrhenius_neutral_pair" if enabled else "disabled"


def _response_snapshot(endpoint, shear_rate_s):
    """Return scalar mechanics/thermal diagnostics on one accepted state."""
    power, _, metrics = _band(endpoint)
    flow = _flow_exposure(endpoint, power, shear_rate_s)
    stress = np.asarray(endpoint.observables["common_stress_tensor_Pa"])
    temperature = np.asarray(endpoint.observables["common_temperature_K"])
    return {
        "load": float(endpoint.load),
        "mean_shear_stress_Pa": float(np.mean(stress[..., 0, 1])),
        "temperature_mean_K": float(np.mean(temperature)),
        "temperature_range_K": float(np.ptp(temperature)),
        "plastic_power_mean_W_m3": float(np.mean(power)),
        "plastic_power_maximum_W_m3": float(np.max(power)),
        "plastic_power_maximum_to_mean": float(metrics["maximum_to_mean"]),
        "plastic_to_imposed_power_fraction": flow,
        "connected_physical_band": _physical_band(metrics, flow),
    }


def _temperature_seed(state, peak_K, width_m, spacing_m, normal="x"):
    """Apply one owner-independent, mean-zero thermal imperfection."""
    amplitude = float(peak_K)
    if amplitude == 0.0:
        return state
    if amplitude < 0.0 or width_m <= 0.0:
        raise ValueError("temperature seed amplitude/width must be nonnegative/positive")
    grid = state.supports.shape[1:]
    x, y = np.meshgrid(np.arange(grid[0]), np.arange(grid[1]), indexing="ij")
    if normal == "x":
        coordinate = x
        period = grid[0]
    elif normal == "y":
        coordinate = y
        period = grid[1]
    elif normal == "diagonal":
        if grid[0] != grid[1]:
            raise ValueError("diagonal seed requires a square grid")
        coordinate = np.mod(x-y, grid[0])
        period = grid[0]
    else:
        raise ValueError("unknown thermal seed normal")
    distance = np.minimum(
        np.abs(coordinate-.5*period),
        period-np.abs(coordinate-.5*period))
    profile = np.exp(-.5*(distance/(width_m/spacing_m))**2)
    profile -= np.mean(profile)
    profile /= max(float(np.max(profile)), np.finfo(float).tiny)
    owners = tuple(replace(
        owner, temperature_K=np.asarray(owner.temperature_K)+amplitude*profile)
        for owner in state.owners)
    # Preserve explicit front-interface reservoirs and their ledger. Rebuilding
    # only from grain IDs/supports/owners would silently erase the exact
    # moving-front history carried by a restart checkpoint.
    seeded = replace(state, owners=owners)
    seeded.validate()
    return seeded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--intervals", type=int, default=50)
    parser.add_argument("--capture-distance", type=float, default=5e-7)
    parser.add_argument("--disable-rx", action="store_true")
    parser.add_argument("--interface-neutral-sink", action="store_true")
    parser.add_argument("--wall-orientation-relaxation", action="store_true")
    parser.add_argument(
        "--production-rx", action="store_true",
        help=("enable the qualified complete-clock source, junction, wall "
              "order/glide, and dipole organization channels"))
    parser.add_argument(
        "--source-mode", choices=("gb_tj_only", "matched_random"),
        default="gb_tj_only",
        help=("Hall--Petch source geography; matched_random preserves the "
              "exact GB/TJ opportunity distribution and changes only its "
              "spatial placement"))
    parser.add_argument(
        "--interface-mobile-emission", action="store_true",
        help=("enable energy-qualified neutral-pair emission from the explicit "
              "mobile-channel GB reservoir under the same Hall--Petch hazard"))
    parser.add_argument(
        "--front-transmission-fraction", type=float,
        help=("fixed front dislocation transmission fraction; omission "
            "preserves the inherited transfer law"))
    parser.add_argument("--override-dt", type=float)
    parser.add_argument("--override-shear-rate", type=float)
    parser.add_argument(
        "--thermal-control", choices=("inherited", "adiabatic", "strong_bath"),
        default="inherited")
    parser.add_argument("--bath-rate", type=float, default=5e9)
    parser.add_argument("--temperature-seed", type=float, default=0.0)
    parser.add_argument("--temperature-seed-width", type=float, default=6e-7)
    parser.add_argument(
        "--temperature-seed-normal", choices=("x", "y", "diagonal"),
        default="x")
    args = parser.parse_args()
    source_status = subprocess.check_output(
        ("git", "status", "--porcelain", "--untracked-files=no"),
        cwd=ROOT, text=True)
    args.out.mkdir(parents=True, exist_ok=True)
    state, runtime, step, load, initial_volume, configuration, provenance = (
        _load_checkpoint(args.checkpoint))
    physics_args = _args_from_configuration(configuration)
    physics_args.shear_rate = float(
        configuration["shear_rate_s"] if args.override_shear_rate is None
        else args.override_shear_rate)
    physics_args.dt = float(
        configuration["dt_s"] if args.override_dt is None else args.override_dt)
    if args.thermal_control != "inherited":
        physics_args.thermal_control = args.thermal_control
        physics_args.bath_rate = args.bath_rate
    spacing, _, _, _, _, _, base_context = _physics(physics_args)
    state = _temperature_seed(
        state, args.temperature_seed, args.temperature_seed_width,
        spacing, args.temperature_seed_normal)
    rx_enabled = not args.disable_rx
    wall_parameters = base_context.wall_parameters
    if args.production_rx:
        wall_parameters = replace(
            wall_parameters, wall_nye_match_coefficient_J_m=(
                wall_parameters.c44_Pa*(1.5e-7)**2))
    elif args.wall_orientation_relaxation:
        wall_parameters = replace(
            wall_parameters, wall_nye_match_coefficient_J_m=2.0e-5)
    orientation_enabled = bool(
        args.wall_orientation_relaxation or args.production_rx)
    source_mode = _production_source_mode(
        args.production_rx, args.source_mode)
    interface_emission_mode = _interface_emission_mode(
        args.production_rx, args.interface_mobile_emission)
    context = replace(
        base_context, wall_parameters=wall_parameters,
        shear_rate_s=physics_args.shear_rate,
        front_enabled=True,
        source_mode=source_mode,
        source_parameters=(
            ArrheniusSourceParameters(grain_size_m=2.4e-6)
            if args.production_rx else None),
        source_random_seed=67, sink_mode="disabled",
        interface_mobile_emission_mode=(
            interface_emission_mode),
        interface_sink_mode=("neutral_pair" if args.interface_neutral_sink
                             else "disabled"),
        interface_sink_parameters=(NeutralBoundarySinkParameters()
                                   if args.interface_neutral_sink else None),
        structural_absorption_mode=("common_axis" if rx_enabled else "disabled"),
        structural_absorption_parameters=(
            InterfaceStructuralAbsorptionParameters(complete_physical_clock=True)
            if rx_enabled else None),
        structural_junction_mode=(
            "arrhenius_cycle" if args.production_rx and rx_enabled
            else "disabled"),
        structural_junction_parameters=(
            JunctionReactionParameters(complete_physical_clock=True)
            if args.production_rx and rx_enabled else None),
        lattice_absorption_mode=("one_flux_mura" if rx_enabled else "disabled"),
        lattice_absorption_parameters=(
            LatticeInterfaceAbsorptionParameters(
                capture_distance_m=args.capture_distance,
                complete_physical_clock=True)
            if rx_enabled else None),
        wall_orientation_relaxation_mode=(
            "ordered_nye" if orientation_enabled else "disabled"),
        wall_orientation_relaxation_parameters=(
            WallOrientationRelaxationParameters(boundary_clearance_m=2.0e-7)
            if orientation_enabled else None),
        wall_order_relaxation_mode=(
            "implicit_arrhenius" if args.production_rx else "disabled"),
        wall_order_relaxation_parameters=(
            ImplicitWallOrderRelaxationParameters(
                owner_support_threshold=1.0e-8,
                optimizer_max_iterations=50)
            if args.production_rx else None),
        conservative_wall_glide_mode=(
            "chemical_edge_glide" if args.production_rx else "disabled"),
        conservative_wall_glide_parameters=(
            ConservativeWallGlideParameters(maximum_edge_fraction=.02)
            if args.production_rx else None),
        conservative_dipole_wall_mode=(
            "neutral_pair_expansion" if args.production_rx else "disabled"),
        conservative_dipole_wall_parameters=(
            ConservativeDipoleWallParameters()
            if args.production_rx else None))
    if args.front_transmission_fraction is not None:
        transmission = float(args.front_transmission_fraction)
        if not np.isfinite(transmission) or not 0.0 <= transmission <= 1.0:
            raise ValueError("front transmission fraction must lie in [0,1]")
        context = replace(context, front_kinetics=replace(
            context.front_kinetics, transfer_law=replace(
                context.front_kinetics.transfer_law,
                transmission_fraction=transmission)))
    endpoint = make_endpoint(state, runtime, load, context)
    initial_supports = np.asarray(state.supports).copy()
    initial_orientations = np.asarray([
        np.mean(owner.orientation_rad) for owner in state.owners])
    initial_power, initial_band, initial_metrics = _band(endpoint)
    initial_flow_exposure = _flow_exposure(
        endpoint, initial_power, context.shear_rate_s)
    initial_physical_band = _physical_band(
        initial_metrics, initial_flow_exposure)
    commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()
    output_configuration = dict(configuration)
    output_configuration.update(
        schema="v67-drx-asb-overlap-v1", target_steps=args.intervals,
        dt_s=physics_args.dt, shear_rate_s=physics_args.shear_rate,
        thermal_control=physics_args.thermal_control,
        thermal_bath_rate_s=(
            0.0 if physics_args.thermal_control == "adiabatic"
            else physics_args.bath_rate),
        temperature_seed_peak_K=args.temperature_seed,
        temperature_seed_width_m=args.temperature_seed_width,
        temperature_seed_normal=args.temperature_seed_normal,
        temperature_seed_mean_zero=True,
        parent_checkpoint=str(args.checkpoint.resolve()),
        rx_enabled=rx_enabled,
        interface_neutral_sink_enabled=bool(args.interface_neutral_sink),
        production_rx_enabled=bool(args.production_rx),
        source_mode=source_mode,
        source_random_seed=(67 if source_mode != "disabled" else None),
        interface_mobile_emission_mode=(
            context.interface_mobile_emission_mode),
        front_transmission_fraction=float(
            context.front_kinetics.transfer_law.transmission_fraction),
        lattice_absorption_mode=("one_flux_mura" if rx_enabled else "disabled"),
        lattice_capture_distance_m=args.capture_distance,
        structural_absorption_mode=("common_axis" if rx_enabled else "disabled"))
    output_configuration.update(
        wall_orientation_relaxation_mode=(
            "ordered_nye" if orientation_enabled else "disabled"),
        wall_nye_match_coefficient_J_m=(
            wall_parameters.wall_nye_match_coefficient_J_m))
    growth_inside = np.zeros(len(state.owners))
    growth_outside = np.zeros(len(state.owners))
    active_band_intervals = 0
    active_band_area_fraction_sum = 0.0
    lattice_heat_inside_band_J = 0.0
    lattice_heat_outside_band_J = 0.0
    first_law_audit_keys = (
        "mechanical_relative_first_law_residual",
        "source_relative_first_law_residual",
        "wall_order_relaxation_relative_first_law_residual",
        "conservative_dipole_wall_relative_first_law_residual",
        "conservative_wall_glide_relative_first_law_residual",
        "structural_absorption_relative_first_law_residual",
        "structural_absorption_post_relative_first_law_residual",
        "structural_junction_relative_first_law_residual",
        "structural_junction_post_relative_first_law_residual",
        "lattice_absorption_relative_first_law_residual",
        "wall_orientation_relaxation_relative_first_law_residual",
    )
    maximum_first_law_residuals = {key: 0.0 for key in first_law_audit_keys}
    maximum_lattice_mura_residual = 0.0
    records = []
    response_history = [{
        "interval": 0,
        **_response_snapshot(endpoint, context.shear_rate_s),
    }]
    terminal_band_persistence_intervals = 0
    dt = physics_args.dt
    for index in range(1, args.intervals+1):
        power_before, band_before, metrics_before = _band(endpoint)
        flow_exposure_before = _flow_exposure(
            endpoint, power_before, context.shear_rate_s)
        physical_band_before = _physical_band(
            metrics_before, flow_exposure_before)
        active_mask = (band_before if physical_band_before
                       else np.zeros_like(band_before))
        before_supports = np.asarray(endpoint.state.supports)
        interval = advance_complete_multigrain_interval(endpoint, dt, context)
        endpoint = interval.endpoint
        maximum_lattice_mura_residual = max(
            maximum_lattice_mura_residual,
            abs(float(interval.audit[
                "lattice_absorption_mura_relative_residual"])))
        for key in first_law_audit_keys:
            maximum_first_law_residuals[key] = max(
                maximum_first_law_residuals[key],
                abs(float(interval.audit[key])))
        lattice_heat = np.asarray(
            interval.integrals["lattice_absorption_heat_J_by_cell"])
        lattice_heat_inside_band_J += float(np.sum(
            lattice_heat*active_mask, dtype=np.longdouble))
        lattice_heat_outside_band_J += float(np.sum(
            lattice_heat*(~active_mask), dtype=np.longdouble))
        positive = np.maximum(
            np.asarray(endpoint.state.supports)-before_supports, 0.0)
        growth_inside += np.sum(
            positive*active_mask[None, ...], axis=(1, 2))*spacing**2
        growth_outside += np.sum(
            positive*(~active_mask)[None, ...], axis=(1, 2))*spacing**2
        active_band_intervals += int(physical_band_before)
        active_band_area_fraction_sum += float(np.mean(active_mask))
        if index == 1 or index == args.intervals or index % 10 == 0:
            records.append({
                "interval": index, "load": endpoint.load,
                "band_before": metrics_before,
                "flow_exposure_fraction_before": flow_exposure_before,
                "physical_band_before": physical_band_before,
                "lattice_classification": interval.audit[
                    "lattice_absorption_classification"],
                "structural_classification": interval.audit[
                    "structural_absorption_classification"],
                "front_directions": interval.audit["front_directions"],
                "wall_orientation_classification": interval.audit[
                    "wall_orientation_relaxation_classification"],
                "wall_orientation_fractions": interval.audit[
                    "wall_orientation_relaxation_fractions"],
            })
        response = {
            "interval": index,
            **_response_snapshot(endpoint, context.shear_rate_s),
        }
        response_history.append(response)
        terminal_band_persistence_intervals = (
            terminal_band_persistence_intervals + 1
            if response["connected_physical_band"] else 0)
        temporary = args.out/"checkpoint_latest.tmp.npz"
        _save_checkpoint(
            temporary, endpoint.state, endpoint.runtime, step+index,
            endpoint.load, initial_volume, output_configuration,
            {"source_commit": commit, "parent_provenance": provenance},
            diagnostic_arrays={
                "diagnostic_v67_overlap_records_json": np.asarray(
                    json.dumps(records)),
                "diagnostic_v67_overlap_response_json": np.asarray(
                    json.dumps(response_history))})
        os.replace(temporary, args.out/"checkpoint_latest.npz")
    final_power, final_band, final_metrics = _band(endpoint)
    final_flow_exposure = _flow_exposure(
        endpoint, final_power, context.shear_rate_s)
    final_physical_band = _physical_band(final_metrics, final_flow_exposure)
    final_supports = np.asarray(endpoint.state.supports)
    final_orientations = np.asarray([
        np.mean(owner.orientation_rad) for owner in endpoint.state.owners])
    area_before = np.sum(initial_supports, axis=(1, 2))*spacing**2
    area_after = np.sum(final_supports, axis=(1, 2))*spacing**2
    net_growth = area_after-area_before
    total_growth = growth_inside+growth_outside
    overlap_fraction = np.divide(
        growth_inside, total_growth, out=np.zeros_like(total_growth),
        where=total_growth > 0.0)
    mean_active_band_fraction = active_band_area_fraction_sum/args.intervals
    overlap_enrichment = overlap_fraction/max(mean_active_band_fraction, 1e-300)
    lattice_heat_total_J = (
        lattice_heat_inside_band_J+lattice_heat_outside_band_J)
    lattice_heat_inside_fraction = (
        lattice_heat_inside_band_J/lattice_heat_total_J
        if lattice_heat_total_J > 0.0 else 0.0)
    lattice_heat_band_enrichment = (
        lattice_heat_inside_fraction/max(mean_active_band_fraction, 1e-300))
    stress_history = np.abs(np.asarray([
        row["mean_shear_stress_Pa"] for row in response_history]))
    peak_stress_index = int(np.argmax(stress_history))
    peak_stress = float(stress_history[peak_stress_index])
    endpoint_softening = float(max(
        peak_stress-stress_history[-1], 0.0)/max(peak_stress, 1e-300))
    checkpoint = args.out/"checkpoint_latest.npz"
    temporary = args.out/"checkpoint_latest.tmp.npz"
    _save_checkpoint(
        temporary, endpoint.state, endpoint.runtime, step+args.intervals,
        endpoint.load, initial_volume, output_configuration,
        {"source_commit": commit, "parent_provenance": provenance})
    os.replace(temporary, checkpoint)
    owner_results = {
        str(grain): {
            "area_before_m2": float(area_before[i]),
            "area_after_m2": float(area_after[i]),
            "net_area_change_m2": float(net_growth[i]),
            "relative_area_change": float(net_growth[i]/area_before[i]),
            "positive_front_growth_inside_active_band_m2": float(growth_inside[i]),
            "positive_front_growth_outside_active_band_m2": float(growth_outside[i]),
            "positive_growth_inside_band_fraction": float(overlap_fraction[i]),
            "growth_inside_band_enrichment": float(overlap_enrichment[i]),
            "mean_orientation_change_rad": float(
                final_orientations[i]-initial_orientations[i]),
        } for i, grain in enumerate(state.grain_ids)}
    growing = [row for row in owner_results.values()
               if row["net_area_change_m2"] > 0.0]
    payload = {
        "schema": "v67-drx-asb-overlap-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": commit, "source_tree_clean": not bool(source_status),
        "parent_checkpoint": str(args.checkpoint.resolve()),
        "parent_checkpoint_sha256": _sha(args.checkpoint),
        "parent_provenance": provenance,
        "configuration": output_configuration,
        "initial_power_localization": initial_metrics,
        "final_power_localization": final_metrics,
        "flow_regime": {
            "minimum_plastic_power_fraction": .01,
            "initial_plastic_to_imposed_power_fraction": initial_flow_exposure,
            "final_plastic_to_imposed_power_fraction": final_flow_exposure,
            "initial_physical_band": initial_physical_band,
            "final_physical_band": final_physical_band,
        },
        "initial_final_band_jaccard": float(
            np.sum(initial_band & final_band)/max(
                np.sum(initial_band | final_band), 1)),
        "owner_results": owner_results,
        "active_band_intervals": active_band_intervals,
        "mean_active_largest_band_area_fraction": mean_active_band_fraction,
        "lattice_absorption_heat_overlap": {
            "inside_active_band_J": lattice_heat_inside_band_J,
            "outside_active_band_J": lattice_heat_outside_band_J,
            "inside_active_band_fraction": lattice_heat_inside_fraction,
            "inside_band_enrichment": lattice_heat_band_enrichment,
        },
        "sampled_interval_records": records,
        "response_history": response_history,
        "response_summary": {
            "peak_absolute_mean_shear_stress_Pa": peak_stress,
            "peak_stress_interval": int(
                response_history[peak_stress_index]["interval"]),
            "endpoint_post_peak_softening_fraction": endpoint_softening,
            "maximum_temperature_range_K": float(max(
                row["temperature_range_K"] for row in response_history)),
            "terminal_band_persistence_intervals": int(
                terminal_band_persistence_intervals),
        },
        "ledger": asdict(endpoint.runtime.ledger),
        "hard_balance_maxima": {
            "lattice_mura_relative_residual": (
                maximum_lattice_mura_residual),
            "first_law_relative_residuals": maximum_first_law_residuals,
        },
        "decision": {
            "initial_connected_band": initial_physical_band,
            "initial_relative_threshold_band": bool(initial_metrics["band_like"]),
            "rx_pathway_enabled": rx_enabled,
            "production_rx_channels_enabled": bool(args.production_rx),
            "source_mode": source_mode,
            "interface_mobile_emission_mode": (
                context.interface_mobile_emission_mode),
            "front_transmission_fraction": float(
                context.front_kinetics.transfer_law.transmission_fraction),
            "interface_neutral_sink_enabled": bool(
                args.interface_neutral_sink),
            "final_connected_band": final_physical_band,
            "final_relative_threshold_band": bool(final_metrics["band_like"]),
            "band_persisted_every_interval": active_band_intervals == args.intervals,
            "terminal_band_persisted_at_least_20_intervals": bool(
                terminal_band_persistence_intervals >= 20),
            "any_owner_net_growth": bool(growing),
            "growing_owner_intersects_active_band": bool(any(
                row["positive_front_growth_inside_active_band_m2"] > 0.0
                for row in growing)),
            "lattice_absorption_occurs_inside_active_band": bool(
                lattice_heat_inside_band_J > 0.0),
            "all_hard_absorption_balances_passed": bool(
                not rx_enabled or (
                    endpoint.runtime.ledger.lattice_absorption_rejected_intervals
                    == 0)),
            "mura_hard_balance_passed": bool(
                maximum_lattice_mura_residual < 2.0e-11),
            "first_law_hard_balance_passed": bool(all(
                value < .05
                for value in maximum_first_law_residuals.values())),
            "grain_ids_unchanged": endpoint.state.grain_ids == state.grain_ids,
        },
        "artifacts": {"checkpoint_latest.npz": _sha(checkpoint)},
        "claim_boundary": (
            "Prepared high-rate localization endpoint with existing grain "
            "owners; this tests spatial coexistence, not spontaneous nucleation "
            "or grid convergence of the inherited band."),
    }
    (args.out/"result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
