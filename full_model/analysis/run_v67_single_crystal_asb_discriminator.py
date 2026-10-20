#!/usr/bin/env python3
"""Matched adiabatic/bath discriminator for thermal localization.

The calculation uses one periodic material owner so grain-boundary geometry
cannot seed a power band.  A declared, mean-zero temperature imperfection is
the only spatial seed.  Both controls use the same EXP-floor Arrhenius
Peierls--Taylor mechanics and differ only by physical heat extraction.
"""

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
    _save_checkpoint, initialize_network_state,
)
from full_model.analysis.run_v66_coupled_rx_asb import _physics
from full_model.analysis.run_v67_drx_asb_overlap import (
    _multiplication_parameters,
)
from full_model.analysis.spatial_localization import spatial_localization_metrics
from full_model.production.coupled_multigrain_interval import (
    advance_complete_multigrain_interval, make_endpoint,
)
from full_model.production.multigrain_common_state import (
    multigrain_state_digest,
)
from full_model.production.multigrain_production import MultiGrainProductionRuntime


ROOT = Path(__file__).resolve().parents[2]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def initial_state(args):
    """Return the common matched initial state used by both heat controls."""
    return initialize_network_state(
        args.n, args.temperature, 1,
        spacing_m=args.length/args.n,
        interface_width_m=args.interface_width,
        initial_temperature_band_K=args.temperature_seed,
        initial_temperature_band_width_m=args.seed_width,
        single_crystal_band_normal=args.band_normal)


def _power_metrics(endpoint):
    power = np.maximum(np.asarray(
        endpoint.observables["instantaneous_plastic_power_W_m3"]), 0.0)
    return power, spatial_localization_metrics(power, threshold_multiple=2.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--thermal-control", choices=("adiabatic", "strong_bath"),
                        required=True)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--length", type=float, default=4.8e-6)
    parser.add_argument("--interface-width", type=float, default=6e-7)
    parser.add_argument("--temperature", type=float, default=1000.0)
    parser.add_argument("--temperature-seed", type=float, default=0.25)
    parser.add_argument("--seed-width", type=float, default=6e-7)
    parser.add_argument("--band-normal", choices=("x", "y", "diagonal"),
                        default="x")
    parser.add_argument("--shear-rate", type=float, default=8e4)
    parser.add_argument("--initial-shear-strain", type=float, default=0.012)
    parser.add_argument("--dt", type=float, default=2e-9)
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--bath-rate", type=float, default=5e9)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--maximum-mechanical-subdivisions", type=int, default=10)
    parser.add_argument(
        "--multiplication-law",
        choices=("legacy_v21_mobile", "v34_km_forest", "disabled"),
        default="legacy_v21_mobile",
        help="explicit common-wall multiplication schema")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    source_commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()
    source_dirty = bool(subprocess.check_output(
        ("git", "status", "--porcelain", "--untracked-files=no"),
        cwd=ROOT, text=True).strip())
    state = initial_state(args)
    initial_digest = multigrain_state_digest(state)
    spacing = args.length/args.n
    thickness = 2.0*2.48e-10
    _, _, _, _, _, _, context = _physics(args)
    wall_parameters = _multiplication_parameters(
        context.wall_parameters, args.multiplication_law, {})
    # There is one owner and therefore no front, source boundary, or label
    # transaction.  All spatial response comes from the common mechanics and
    # physical heat equation.
    context = replace(
        context, front_enabled=False, wall_parameters=wall_parameters)
    runtime = MultiGrainProductionRuntime(())
    endpoint = make_endpoint(state, runtime, args.initial_shear_strain, context)
    initial_volume = np.asarray([args.length**2*thickness])
    configuration = {
        "schema": "v67-single-crystal-asb-discriminator-v1",
        "n": args.n, "length_m": args.length,
        "interface_width_m": args.interface_width,
        "temperature_K": args.temperature,
        "temperature_seed_peak_K": args.temperature_seed,
        "temperature_seed_width_m": args.seed_width,
        "temperature_seed_mean_zero": True,
        "band_normal": args.band_normal,
        "shear_rate_s": args.shear_rate,
        "initial_shear_strain": args.initial_shear_strain,
        "dt_s": args.dt, "target_steps": args.steps,
        "thermal_control": args.thermal_control,
        "thermal_bath_rate_s": (
            0.0 if args.thermal_control == "adiabatic" else args.bath_rate),
        "maximum_mechanical_subdivisions": args.maximum_mechanical_subdivisions,
        "immutable_flow_law": (
            "EXP-floor Arrhenius Peierls-Taylor common mechanics"),
        "spatial_seed": "mean-zero temperature Gaussian only",
        "front_enabled": False,
        "multiplication_law": wall_parameters.multiplication_law,
        "multiplication_coefficient_m1": (
            wall_parameters.multiplication_coefficient_m1),
        "multiplication_storage_work_fraction": (
            wall_parameters.multiplication_storage_work_fraction),
        "multiplication_destination": (
            "forest_plus_minus"
            if wall_parameters.multiplication_law == "v34_km_forest_storage_v1"
            else "mobile_plus_minus"),
    }
    initial_temperature = np.asarray(
        endpoint.observables["common_temperature_K"])
    initial_power, initial_metrics = _power_metrics(endpoint)
    history = []
    maximum_mechanical_residual = 0.0
    for index in range(1, args.steps+1):
        interval = advance_complete_multigrain_interval(endpoint, args.dt, context)
        endpoint = interval.endpoint
        raw_mult = np.asarray(interval.integrals[
            "multiplication_raw_line_increment_m2"])
        accepted_mult = np.asarray(interval.integrals[
            "multiplication_accepted_line_increment_m2"])
        power, metrics = _power_metrics(endpoint)
        temperature = np.asarray(endpoint.observables["common_temperature_K"])
        stress = np.asarray(endpoint.observables["common_stress_tensor_Pa"])
        residual = abs(float(interval.audit[
            "mechanical_relative_first_law_residual"]))
        maximum_mechanical_residual = max(maximum_mechanical_residual, residual)
        history.append({
            "step": index,
            "physical_time_s": endpoint.runtime.ledger.physical_time_s,
            "applied_shear_strain": endpoint.load,
            "mean_shear_stress_Pa": float(np.mean(stress[..., 0, 1])),
            "temperature_mean_K": float(np.mean(temperature)),
            "temperature_contrast_K": float(np.ptp(temperature)),
            "temperature_peak_minus_mean_K": float(
                np.max(temperature)-np.mean(temperature)),
            "power_mean_W_m3": float(np.mean(power)),
            "power_max_W_m3": float(np.max(power)),
            "power_localization": metrics,
            "mechanical_relative_first_law_residual": residual,
            "mechanical_external_work_J": float(interval.integrals[
                "mechanical_external_work_J"]),
            "mechanical_irreversible_heat_J": float(interval.integrals[
                "mechanical_irreversible_heat_J"]),
            "thermal_bath_exchange_J": float(np.sum(
                interval.integrals["thermal_bath_exchange_J_m3"])
                * spacing**2*thickness),
            "thermal_conduction_exchange_J": float(np.sum(
                interval.integrals["thermal_conduction_J_m3"])
                * spacing**2*thickness),
            "multiplication_raw_physical_line_increment_m": float(np.sum(
                raw_mult, dtype=np.longdouble)*spacing**2*thickness),
            "multiplication_accepted_physical_line_increment_m": float(
                np.sum(accepted_mult, dtype=np.longdouble)
                *spacing**2*thickness),
            "multiplication_budget_active_cell_time_fraction": float(
                interval.audit[
                    "multiplication_budget_active_cell_time_fraction"]),
        })
        if index == 1 or index % args.checkpoint_every == 0 or index == args.steps:
            temporary = args.out/"checkpoint_latest.tmp.npz"
            _save_checkpoint(
                temporary, endpoint.state, endpoint.runtime, index,
                endpoint.load, initial_volume, configuration,
                {"source_commit": source_commit})
            os.replace(temporary, args.out/"checkpoint_latest.npz")
            _atomic_json(args.out/"history.json", history)
            print(json.dumps({
                "step": index, "target": args.steps,
                "strain": endpoint.load,
                "stress_MPa": history[-1]["mean_shear_stress_Pa"]/1e6,
                "temperature_contrast_K": history[-1]["temperature_contrast_K"],
                "power_maximum_to_mean": metrics["maximum_to_mean"],
                "band_like": metrics["band_like"],
            }, sort_keys=True), flush=True)

    checkpoint = args.out/"checkpoint_latest.npz"
    stresses = np.asarray([row["mean_shear_stress_Pa"] for row in history])
    band_flags = np.asarray([
        row["power_localization"]["band_like"] for row in history], dtype=bool)
    final_temperature = np.asarray(endpoint.observables["common_temperature_K"])
    final_power, final_metrics = _power_metrics(endpoint)
    peak_index = int(np.argmax(stresses))
    payload = {
        "schema": "v67-single-crystal-asb-discriminator-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit, "source_tree_clean": not source_dirty,
        "configuration": configuration,
        "initial_state_digest": initial_digest,
        "initial_temperature": {
            "mean_K": float(np.mean(initial_temperature)),
            "contrast_K": float(np.ptp(initial_temperature)),
        },
        "initial_power_localization": initial_metrics,
        "final_power_localization": final_metrics,
        "result": {
            "final_temperature_mean_K": float(np.mean(final_temperature)),
            "final_temperature_contrast_K": float(np.ptp(final_temperature)),
            "maximum_temperature_contrast_K": float(max(
                row["temperature_contrast_K"] for row in history)),
            "peak_stress_Pa": float(stresses[peak_index]),
            "peak_stress_step": peak_index+1,
            "final_stress_Pa": float(stresses[-1]),
            "post_peak_stress_drop_fraction": float(
                (stresses[peak_index]-stresses[-1])
                / max(abs(stresses[peak_index]), np.finfo(float).tiny)),
            "band_intervals": int(np.sum(band_flags)),
            "terminal_band_persistence_intervals": int(next(
                (offset for offset, active in enumerate(band_flags[::-1])
                 if not active), len(band_flags))),
            "maximum_mechanical_relative_first_law_residual": (
                maximum_mechanical_residual),
            "grain_ids_unchanged": endpoint.state.grain_ids == (10,),
            "multiplication_raw_physical_line_m": float(sum(
                row["multiplication_raw_physical_line_increment_m"]
                for row in history)),
            "multiplication_accepted_physical_line_m": float(sum(
                row["multiplication_accepted_physical_line_increment_m"]
                for row in history)),
            "multiplication_maximum_budget_active_cell_time_fraction": float(
                max(row[
                    "multiplication_budget_active_cell_time_fraction"]
                    for row in history)),
        },
        "ledger": asdict(endpoint.runtime.ledger),
        "artifacts": {"checkpoint_latest.npz": _sha(checkpoint)},
        "claim_boundary": (
            "Prepared one-owner thermal-imperfection discriminator; no grain "
            "boundary, grain birth, moving front, or rate-dependent switch."),
    }
    _atomic_json(args.out/"result.json", payload)
    print(json.dumps(payload["result"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
