#!/usr/bin/env python3
"""Advance a matured RX owner in the complete DRX/ASB production map.

The run deliberately starts from the V66 inherited-subgrain handoff.  It does
not claim spontaneous nucleus creation: the question is whether that retained
orientation can migrate while the very same state carries Arrhenius plasticity,
physical dissipation, heat, Nye content, and load.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np

from full_model.analysis.run_v58_three_grain_production import (
    _load_checkpoint, _save_checkpoint, network_interfaces,
)
from full_model.analysis.run_v66_multigrain_handoff import _advance, _initial
from full_model.production.arrhenius_kinetics import ActivatedProcess, EV_J
from full_model.production.arrhenius_dislocation_sources import (
    ArrheniusSourceParameters, boundary_tj_site_field,
)
from full_model.production.common_tensorial_wall import CommonWallParameters
from full_model.production.coupled_multigrain_interval import (
    CoupledMultigrainContext, advance_complete_multigrain_interval, make_endpoint,
)
from full_model.production.multigrain_common_state import PhysicalTransferLaw
from full_model.production.multigrain_common_state import reconstruct_multigrain_common
from full_model.production.multigrain_production import (
    MultiGrainFrontKinetics, MultiGrainProductionRuntime,
)
from full_model.production.tensorial_nye import bcc_four_family_systems


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT/"full_model"/"verification"/"v66_coupled_rx_asb"


def _atomic_json(path: Path, payload) -> None:
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    os.replace(temporary, path)


def _configuration(args) -> dict:
    return {
        "schema": "v66-coupled-rx-asb-configuration-v1",
        "n": args.n,
        "length_m": args.length,
        "temperature_K": args.temperature,
        "shear_rate_s": args.shear_rate,
        "dt_s": args.dt,
        "target_steps": args.steps,
        "front_maximum_substep_s": args.dt,
        "thermal_diffusivity_m2_s": .15/3.8e6,
        "initial_state": "V66 inherited orientation-gradient matured owner",
        "immutable_flow_law": "EXP-floor Arrhenius with Peierls and Taylor resistance",
        "source_mode": args.source_mode,
        "source_grain_size_m": args.source_grain_size,
    }


def _physics(args):
    spacing = args.length/args.n
    thickness = 2.0*2.48e-10
    systems = bcc_four_family_systems()
    wall = CommonWallParameters(
        spacing_m=spacing, elastic_iterations=2,
        mobile_correlation_diffusivity_m2_s=0.0,
        transport_scheme="upwind", maximum_fraction_per_step=.10,
        volumetric_heat_capacity_J_m3_K=3.8e6,
        thermal_diffusivity_m2_s=.15/3.8e6, bath_rate_s=0.0)
    kinetics = MultiGrainFrontKinetics(
        ActivatedProcess("V66 coupled matured HAGB", 1e10, 0.0, 1e10),
        PhysicalTransferLaw(.55, .08, .02), .35*EV_J, 1e9,
        2.0, 1.5, .10, wall.burgers_m**3, wall.burgers_m,
        virtual_fraction=2e-4, maximum_fraction_per_step=.01,
        closure_fraction=.05, maximum_backtracks=14,
        temperature_resolution="local_interface")
    energy = {
        "topologies": (), "systems": systems,
        "phase_barrier_J_m3": 5e6, "phase_gradient_J_m": 5e-7,
        "boundary_line_energy_J_m": wall.line_energy_J_m,
        "boundary_junction_energy_J_m": wall.junction_energy_J_m,
        "reference_temperature_K": args.temperature,
    }
    context = CoupledMultigrainContext(
        systems, (), wall, kinetics, spacing, thickness, args.shear_rate,
        args.dt, 10, energy, heat_deposition_mode="local_realized_event",
        front_enabled=True, warm_start_front_factor=True)
    return spacing, thickness, systems, wall, kinetics, energy, context


def _prepared_state(args, physics):
    spacing, thickness, systems, wall, kinetics, energy, _ = physics
    reduced = json.loads((
        ROOT/"full_model"/"verification"/"v66_subgrain_maturation"/"result.json"
    ).read_text())
    contrast = reduced["cases"]["T1000_rate1"]["stored_energy_contrast_Pa"]
    state, high, low = _initial(
        args.n, args.length, args.temperature, contrast)
    runtime = MultiGrainProductionRuntime(network_interfaces(state))
    state, runtime, _, _ = _advance(
        state, runtime, 20, 2e-8, spacing=spacing, thickness=thickness,
        wall=wall, kinetics=kinetics, energy=dict(energy, mean_strain=np.zeros((2, 2))),
        systems=systems)
    return state, runtime, {"contrast_Pa": contrast,
                            "high_density_m2_each": high,
                            "low_density_m2_each": low}


def _record(step, endpoint, interval, initial_volume, context, wall_time_s):
    temperature = np.asarray(endpoint.observables["common_temperature_K"])
    power = np.maximum(
        np.asarray(endpoint.observables["instantaneous_plastic_power_W_m3"]), 0.0)
    volumes = (np.sum(endpoint.state.supports, axis=(1, 2))
               *context.spacing_m**2*context.represented_thickness_m)
    positive = power[power > 0.0]
    mean_positive = float(np.mean(positive)) if positive.size else 0.0
    temperature_excess = temperature-float(np.min(temperature))
    power_threshold = np.quantile(power, .9)
    temperature_threshold = np.quantile(temperature_excess, .9)
    power_hot = power >= power_threshold
    temperature_hot = temperature_excess >= temperature_threshold
    overlap = float(np.sum(power_hot & temperature_hot)/max(np.sum(power_hot), 1))
    stress = np.asarray(endpoint.observables["common_stress_tensor_Pa"])
    return {
        "step": step,
        "physical_time_s": endpoint.runtime.ledger.physical_time_s,
        "coupled_time_s": step*interval.duration_s,
        "applied_shear_strain": endpoint.load,
        "mean_shear_stress_Pa": float(np.mean(stress[..., 0, 1])),
        "temperature_mean_K": float(np.mean(temperature)),
        "temperature_max_K": float(np.max(temperature)),
        "temperature_contrast_K": float(np.ptp(temperature)),
        "positive_power_mean_W_m3": mean_positive,
        "positive_power_max_W_m3": float(np.max(power)),
        "power_localization_ratio": float(
            np.max(power)/max(mean_positive, np.finfo(float).tiny)),
        "power_temperature_top_decile_overlap": overlap,
        "target_growth_fraction": float(
            (volumes[1]-initial_volume[1])/initial_volume[1]),
        "grain_volumes_m3": volumes.tolist(),
        "mechanical_relative_first_law_residual": interval.audit[
            "mechanical_relative_first_law_residual"],
        "maximum_front_relative_energy_closure": (
            endpoint.runtime.ledger.maximum_relative_energy_closure),
        "mechanical_external_work_J": interval.integrals[
            "mechanical_external_work_J"],
        "mechanical_irreversible_heat_J": interval.integrals[
            "mechanical_irreversible_heat_J"],
        "front_heat_J": interval.integrals["front_heat_J"],
        "front_directions": interval.audit["front_directions"],
        "source_mode": interval.audit["source_mode"],
        "source_classification": interval.audit["source_classification"],
        "source_event_count": interval.audit["source_event_count"],
        "source_created_line_cell_sum_m2": interval.audit[
            "source_created_line_cell_sum_m2"],
        "source_heat_J": interval.integrals["source_heat_J"],
        "source_relative_first_law_residual": interval.audit[
            "source_relative_first_law_residual"],
        "mechanical_subdivision_depth": interval.audit[
            "mechanical_subdivision_depth"],
        "wall_time_s": wall_time_s,
    }


def _summarize(history, endpoint, initial_volume, preparation, configuration):
    stress = np.asarray([row["mean_shear_stress_Pa"] for row in history])
    peak = int(np.argmax(stress))
    final = history[-1]
    after = stress[peak:]
    softening = float((stress[peak]-np.min(after))/max(abs(stress[peak]), 1e-300))
    target = np.asarray([row["target_growth_fraction"] for row in history])
    temperatures = np.asarray([row["temperature_contrast_K"] for row in history])
    powers = np.asarray([row["power_localization_ratio"] for row in history])
    common, _ = reconstruct_multigrain_common(
        endpoint.state, configuration["length_m"]/configuration["n"])
    total_density = sum(
        np.sum(np.asarray(getattr(common, name)), axis=2)
        for name in ("mobile_plus_m2", "mobile_minus_m2",
                     "forest_plus_m2", "forest_minus_m2",
                     "wall_plus_m2", "wall_minus_m2"))
    boundary = boundary_tj_site_field(endpoint.state.supports, tj_site_weight=2.0)
    def correlation(a, b):
        aa = np.ravel(a)-np.mean(a); bb = np.ravel(b)-np.mean(b)
        scale = np.sqrt(np.sum(aa*aa)*np.sum(bb*bb))
        return 0.0 if scale == 0.0 else float(np.sum(aa*bb)/scale)
    return {
        "schema": "v66-coupled-rx-asb-result-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration,
        "preparation": preparation,
        "result": {
            "steps_completed": len(history),
            "peak_step": int(history[peak]["step"]),
            "peak_stress_Pa": float(stress[peak]),
            "post_peak_softening_fraction": softening,
            "final_temperature_mean_K": final["temperature_mean_K"],
            "maximum_temperature_contrast_K": float(np.max(temperatures)),
            "maximum_power_localization_ratio": float(np.max(powers)),
            "final_target_growth_fraction": float(target[-1]),
            "target_growth_monotone": bool(np.all(np.diff(target) >= -1e-12)),
            "maximum_mechanical_relative_first_law_residual": (
                endpoint.runtime.ledger.maximum_mechanical_relative_first_law_residual),
            "maximum_front_relative_energy_closure": (
                endpoint.runtime.ledger.maximum_relative_energy_closure),
            "accepted_front_events": endpoint.runtime.ledger.accepted_events,
            "rejected_front_events": endpoint.runtime.ledger.rejected_events,
            "source_intervals": endpoint.runtime.ledger.source_intervals,
            "source_inactive_intervals": (
                endpoint.runtime.ledger.source_inactive_intervals),
            "source_event_count": endpoint.runtime.ledger.source_event_count,
            "source_created_line_cell_sum_m2": (
                endpoint.runtime.ledger.source_created_line_cell_sum_m2),
            "source_generated_heat_J": (
                endpoint.runtime.ledger.cumulative_source_generated_heat_J),
            "maximum_source_relative_first_law_residual": (
                endpoint.runtime.ledger.maximum_source_relative_first_law_residual),
            "terminal_density_boundary_correlation": correlation(
                total_density, boundary),
            "cumulative_mechanical_external_work_J": (
                endpoint.runtime.ledger.cumulative_mechanical_external_work_J),
            "cumulative_mechanical_first_law_residual_J": (
                endpoint.runtime.ledger.cumulative_mechanical_first_law_residual_J),
        },
        "decision": {
            "same_common_state_carries_drx_and_asb_channels": True,
            "sustained_target_growth": bool(target[-1] > 0.0),
            "stress_peak_observed": bool(peak < len(history)-1),
            "thermal_localization_observed": bool(np.max(temperatures) > 1.0),
            "strict_asb_qualified": False,
            "reason": (
                "This common-state exposure tests simultaneous mechanisms; strict ASB "
                "still requires persistence/refinement and matched adiabatic/isothermal "
                "controls, while RX remains an inherited-subgrain maturation pathway."),
        },
        "initial_target_volume_m3": float(initial_volume[1]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--length", type=float, default=4.8e-6)
    parser.add_argument("--temperature", type=float, default=1000.0)
    parser.add_argument("--shear-rate", type=float, default=4e4)
    parser.add_argument("--dt", type=float, default=2e-9)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument(
        "--source-mode", choices=("disabled", "gb_tj_only", "matched_random"),
        default="disabled")
    parser.add_argument("--source-grain-size", type=float, default=2.4e-6)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    configuration = _configuration(args)
    physics = _physics(args)
    context = physics[-1]
    if args.source_mode != "disabled":
        context = replace(
            context,
            source_parameters=ArrheniusSourceParameters(
                grain_size_m=args.source_grain_size),
            source_mode=args.source_mode)
    provenance = {"campaign": "V66", "source_commit": "6b104d5",
                  "scientific_scope": "inherited-subgrain coupled exposure"}
    history_path = args.out/"history.json"
    if args.resume is None:
        state, runtime, preparation = _prepared_state(args, physics)
        endpoint = make_endpoint(state, runtime, 0.0, context)
        initial_volume = (np.sum(state.supports, axis=(1, 2))
                          *context.spacing_m**2*context.represented_thickness_m)
        history = []
        start_step = 0
        _atomic_json(args.out/"preparation.json", preparation)
    else:
        state, runtime, start_step, load, initial_volume, saved, saved_provenance = (
            _load_checkpoint(args.resume))
        if saved != configuration or saved_provenance != provenance:
            raise ValueError("resume configuration/provenance mismatch")
        endpoint = make_endpoint(state, runtime, load, context)
        preparation = json.loads((args.out/"preparation.json").read_text())
        history = json.loads(history_path.read_text())
        if len(history) != start_step:
            raise ValueError("history/checkpoint step mismatch")
    for step in range(start_step+1, args.steps+1):
        started = time.perf_counter()
        interval = advance_complete_multigrain_interval(endpoint, args.dt, context)
        endpoint = interval.endpoint
        history.append(_record(
            step, endpoint, interval, initial_volume, context,
            time.perf_counter()-started))
        if step == 1 or step % args.checkpoint_every == 0 or step == args.steps:
            checkpoint = args.out/"checkpoint_latest.npz"
            temporary = args.out/"checkpoint_latest.tmp.npz"
            _save_checkpoint(temporary, endpoint.state, endpoint.runtime, step,
                             endpoint.load, initial_volume, configuration, provenance)
            os.replace(temporary, checkpoint)
            _atomic_json(history_path, history)
            partial = _summarize(
                history, endpoint, initial_volume, preparation, configuration)
            partial["decision"]["run_complete"] = step == args.steps
            _atomic_json(args.out/"result.json", partial)
            print(json.dumps({
                "step": step, "target": args.steps,
                "strain": endpoint.load,
                "stress_MPa": history[-1]["mean_shear_stress_Pa"]/1e6,
                "temperature_contrast_K": history[-1]["temperature_contrast_K"],
                "target_growth_fraction": history[-1]["target_growth_fraction"],
                "mean_interval_wall_s": float(np.mean(
                    [row["wall_time_s"] for row in history[-args.checkpoint_every:]])),
            }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
