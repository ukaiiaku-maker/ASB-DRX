"""Audit the production front's global-temperature approximation on a state.

This is a read-only rate audit.  It reprices each declared interface from the
accepted checkpoint, evaluates the same bidirectional EXP-floor event at the
production global mean temperature and at a geometric interface-weighted
temperature, and never publishes a material transaction.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from full_model.analysis.run_v58_three_grain_production import _load_checkpoint
from full_model.production.arrhenius_kinetics import ActivatedProcess, EV_J
from full_model.production.coupled_front_event import (
    propose_bidirectional_front_event,
)
from full_model.production.common_tensorial_wall import CommonWallParameters
from full_model.production.multigrain_common_state import PhysicalTransferLaw
from full_model.production.multigrain_production import (
    MultiGrainFrontKinetics, _defect, _virtual_channel,
)
from full_model.production.tensorial_nye import bcc_four_family_systems


def _operators(configuration: dict):
    spacing = float(configuration["length_m"])/int(configuration["n"])
    temperature = float(configuration["temperature_K"])
    wall = CommonWallParameters(
        spacing_m=spacing, elastic_iterations=2,
        mobile_correlation_diffusivity_m2_s=0.0,
        transport_scheme="upwind",
        maximum_fraction_per_step=float(configuration["maximum_fraction_per_step"]),
        flow_temperature_override_K=(
            temperature if configuration["flow_temperature_mode"] == "frozen"
            else None),
        recovery_temperature_override_K=(
            temperature if configuration["recovery_temperature_mode"] == "frozen"
            else None),
        volumetric_heat_capacity_J_m3_K=3.8e6,
        thermal_diffusivity_m2_s=float(
            configuration["thermal_diffusivity_m2_s"]),
        bath_rate_s=0.0)
    kinetics = MultiGrainFrontKinetics(
        ActivatedProcess(
            "V58 multi-grain HAGB",
            float(configuration["front_attempt_frequency_s"]), 0.0,
            float(configuration["front_attempt_frequency_s"])),
        PhysicalTransferLaw(.55, .08, .02), .35*EV_J, 1e9,
        2.0, 1.5, .10, wall.burgers_m**3, wall.burgers_m,
        virtual_fraction=2e-4,
        maximum_fraction_per_step=float(
            configuration["front_maximum_fraction_per_step"]),
        closure_fraction=float(configuration["front_closure_fraction"]),
        maximum_backtracks=14,
        temperature_override_K=(
            temperature if configuration["front_temperature_mode"] == "frozen"
            else None))
    return spacing, wall, kinetics


def audit(checkpoint: Path) -> dict:
    (state, runtime, step, gamma, _initial_volume, configuration,
     provenance) = _load_checkpoint(checkpoint)
    spacing, wall, kinetics = _operators(configuration)
    thickness = 2.0*wall.burgers_m
    systems = bcc_four_family_systems()
    common_temperature = sum(
        np.asarray(support, dtype=float)*np.asarray(owner.temperature_K, dtype=float)
        for support, owner in zip(state.supports, state.owners))
    global_temperature = float(np.mean(common_temperature))
    index = {grain_id: position
             for position, grain_id in enumerate(state.grain_ids)}
    strain = np.array([[0.0, .5*gamma], [.5*gamma, 0.0]])
    energy_kwargs = dict(
        mean_strain=strain, topologies=(), systems=systems,
        phase_barrier_J_m3=5e6, phase_gradient_J_m=5e-7,
        boundary_line_energy_J_m=wall.line_energy_J_m,
        boundary_junction_energy_J_m=wall.junction_energy_J_m,
        reference_temperature_K=float(configuration["temperature_K"]))
    rows = []
    for interface in runtime.interfaces:
        a = interface.grain_a_id; b = interface.grain_b_id
        ab, weight_ab = _virtual_channel(
            state, interface, a, b, kinetics=kinetics, spacing_m=spacing,
            represented_thickness_m=thickness, wall_parameters=wall,
            energy_kwargs=energy_kwargs,
            interval_s=float(configuration["dt_s"]), systems=systems)
        ba, weight_ba = _virtual_channel(
            state, interface, b, a, kinetics=kinetics, spacing_m=spacing,
            represented_thickness_m=thickness, wall_parameters=wall,
            energy_kwargs=energy_kwargs,
            interval_s=float(configuration["dt_s"]), systems=systems)
        weight = np.asarray(weight_ab)+np.asarray(weight_ba)
        weight_sum = float(np.sum(weight, dtype=np.longdouble))
        if weight_sum <= 0.0 or (ab is None and ba is None):
            continue
        local_temperature = float(np.sum(
            weight*common_temperature, dtype=np.longdouble)/weight_sum)
        variance = float(np.sum(
            weight*(common_temperature-local_temperature)**2,
            dtype=np.longdouble)/weight_sum)
        delta_ab = 0.0 if ab is None else ab[0]*kinetics.event_volume_m3/ab[1]
        delta_ba = 0.0 if ba is None else ba[0]*kinetics.event_volume_m3/ba[1]

        def event_at(temperature_K):
            return propose_bidirectional_front_event(
                _defect(state.owners[index[a]]),
                _defect(state.owners[index[b]]),
                event_volume_m3=kinetics.event_volume_m3,
                event_length_m=kinetics.event_length_m,
                line_energy_J_m=wall.line_energy_J_m,
                temperature_K=temperature_K, process=kinetics.process,
                h0_J=kinetics.h0_J,
                critical_pressure_Pa=kinetics.critical_pressure_Pa,
                exp_a=kinetics.exp_a, exp_n=kinetics.exp_n,
                exp_floor=kinetics.exp_floor,
                transmission_fraction=0.0, boundary_storage_fraction=0.0,
                neutral_sink_fraction=0.0,
                kinetic_free_energy_a_to_b_J=delta_ab,
                kinetic_free_energy_b_to_a_J=delta_ba,
                actual_reverse_edge=False,
                deterministic_rate_law="complete_dissipation")

        global_event = event_at(global_temperature)
        local_event = event_at(local_temperature)
        global_velocity = float(global_event.net_velocity_a_to_b_m_s)
        local_velocity = float(local_event.net_velocity_a_to_b_m_s)
        rows.append({
            "interface": interface.component_id,
            "grain_ids": [a, b],
            "global_temperature_K": global_temperature,
            "interface_weighted_temperature_K": local_temperature,
            "interface_weighted_temperature_std_K": variance**.5,
            "temperature_offset_K": local_temperature-global_temperature,
            "global_mean_rate_velocity_m_s": global_velocity,
            "interface_weighted_rate_velocity_m_s": local_velocity,
            "relative_velocity_change": abs(local_velocity-global_velocity)/max(
                abs(global_velocity), 1e-300),
            "direction_changed": bool(np.signbit(local_velocity)
                                      != np.signbit(global_velocity)),
            "global_event": asdict(global_event),
            "local_event": asdict(local_event),
        })
    return {
        "schema": "asb-drx-v61-front-temperature-locality-audit-v1",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "source_commit": (provenance or {}).get("source_commit"),
        "step": step, "physical_time_s": runtime.ledger.physical_time_s,
        "applied_shear_strain": gamma,
        "production_temperature_semantics": (
            "unweighted spatial mean of the common temperature; all owner "
            "temperature fields are synchronized in the accepted state"),
        "local_temperature_semantics": (
            "common temperature weighted by the sum of both directional "
            "level-set interface sweep measures"),
        "interfaces": rows,
        "maximum_absolute_temperature_offset_K": max(
            (abs(row["temperature_offset_K"]) for row in rows), default=0.0),
        "maximum_relative_velocity_change": max(
            (row["relative_velocity_change"] for row in rows), default=0.0),
        "any_direction_change": any(row["direction_changed"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({key: result[key] for key in (
        "step", "physical_time_s", "maximum_absolute_temperature_offset_K",
        "maximum_relative_velocity_change", "any_direction_change")}, indent=2))


if __name__ == "__main__":
    main()
