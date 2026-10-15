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
from full_model.production.lattice_interface_absorption import (
    LatticeInterfaceAbsorptionParameters,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--intervals", type=int, default=50)
    parser.add_argument("--capture-distance", type=float, default=5e-7)
    parser.add_argument("--disable-rx", action="store_true")
    parser.add_argument("--interface-neutral-sink", action="store_true")
    args = parser.parse_args()
    source_status = subprocess.check_output(
        ("git", "status", "--porcelain"), cwd=ROOT, text=True)
    args.out.mkdir(parents=True, exist_ok=True)
    state, runtime, step, load, initial_volume, configuration, provenance = (
        _load_checkpoint(args.checkpoint))
    physics_args = _args_from_configuration(configuration)
    physics_args.shear_rate = float(configuration["shear_rate_s"])
    spacing, _, _, _, _, _, base_context = _physics(physics_args)
    rx_enabled = not args.disable_rx
    context = replace(
        base_context, shear_rate_s=float(configuration["shear_rate_s"]),
        front_enabled=True, source_mode="disabled", sink_mode="disabled",
        interface_sink_mode=("neutral_pair" if args.interface_neutral_sink
                             else "disabled"),
        interface_sink_parameters=(NeutralBoundarySinkParameters()
                                   if args.interface_neutral_sink else None),
        structural_absorption_mode=("common_axis" if rx_enabled else "disabled"),
        structural_absorption_parameters=(
            InterfaceStructuralAbsorptionParameters() if rx_enabled else None),
        lattice_absorption_mode=("one_flux_mura" if rx_enabled else "disabled"),
        lattice_absorption_parameters=(
            LatticeInterfaceAbsorptionParameters(
                capture_distance_m=args.capture_distance)
            if rx_enabled else None))
    endpoint = make_endpoint(state, runtime, load, context)
    initial_supports = np.asarray(state.supports).copy()
    initial_orientations = np.asarray([
        np.mean(owner.orientation_rad) for owner in state.owners])
    initial_power, initial_band, initial_metrics = _band(endpoint)
    growth_inside = np.zeros(len(state.owners))
    growth_outside = np.zeros(len(state.owners))
    active_band_intervals = 0
    active_band_area_fraction_sum = 0.0
    records = []
    dt = float(configuration["dt_s"])
    for index in range(1, args.intervals+1):
        _, band_before, metrics_before = _band(endpoint)
        before_supports = np.asarray(endpoint.state.supports)
        interval = advance_complete_multigrain_interval(endpoint, dt, context)
        endpoint = interval.endpoint
        positive = np.maximum(
            np.asarray(endpoint.state.supports)-before_supports, 0.0)
        growth_inside += np.sum(
            positive*band_before[None, ...], axis=(1, 2))*spacing**2
        growth_outside += np.sum(
            positive*(~band_before)[None, ...], axis=(1, 2))*spacing**2
        active_band_intervals += int(metrics_before["band_like"])
        active_band_area_fraction_sum += float(np.mean(band_before))
        if index == 1 or index == args.intervals or index % 10 == 0:
            records.append({
                "interval": index, "load": endpoint.load,
                "band_before": metrics_before,
                "lattice_classification": interval.audit[
                    "lattice_absorption_classification"],
                "structural_classification": interval.audit[
                    "structural_absorption_classification"],
                "front_directions": interval.audit["front_directions"],
            })
    final_power, final_band, final_metrics = _band(endpoint)
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
    commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()
    output_configuration = dict(configuration)
    output_configuration.update(
        schema="v67-drx-asb-overlap-v1", target_steps=args.intervals,
        parent_checkpoint=str(args.checkpoint.resolve()),
        rx_enabled=rx_enabled,
        interface_neutral_sink_enabled=bool(args.interface_neutral_sink),
        lattice_absorption_mode=("one_flux_mura" if rx_enabled else "disabled"),
        lattice_capture_distance_m=args.capture_distance,
        structural_absorption_mode=("common_axis" if rx_enabled else "disabled"))
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
        "initial_final_band_jaccard": float(
            np.sum(initial_band & final_band)/max(
                np.sum(initial_band | final_band), 1)),
        "owner_results": owner_results,
        "active_band_intervals": active_band_intervals,
        "mean_active_largest_band_area_fraction": mean_active_band_fraction,
        "sampled_interval_records": records,
        "ledger": asdict(endpoint.runtime.ledger),
        "decision": {
            "initial_connected_band": bool(initial_metrics["band_like"]),
            "rx_pathway_enabled": rx_enabled,
            "interface_neutral_sink_enabled": bool(
                args.interface_neutral_sink),
            "final_connected_band": bool(final_metrics["band_like"]),
            "band_persisted_every_interval": active_band_intervals == args.intervals,
            "any_owner_net_growth": bool(growing),
            "growing_owner_intersects_active_band": bool(any(
                row["positive_front_growth_inside_active_band_m2"] > 0.0
                for row in growing)),
            "all_hard_absorption_balances_passed": bool(
                not rx_enabled or (
                    endpoint.runtime.ledger.lattice_absorption_rejected_intervals
                    == 0)),
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
