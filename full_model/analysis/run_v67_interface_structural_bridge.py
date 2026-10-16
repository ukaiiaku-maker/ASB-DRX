#!/usr/bin/env python3
"""Exercise the signed-reservoir/common-axis bridge on a retained V67 state."""

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
from full_model.analysis.run_v67_growth_reversal_audit import _directional_audit
from full_model.production.coupled_multigrain_interval import (
    advance_complete_multigrain_interval, make_endpoint,
)
from full_model.production.interface_structural_absorption import (
    InterfaceStructuralAbsorptionParameters,
)
from full_model.production.structural_multigrain_runtime import (
    StructuralMultigrainRuntime, advance_structural_junction_interval,
)


ROOT = Path(__file__).resolve().parents[2]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _args_from_configuration(configuration):
    c = configuration
    return argparse.Namespace(
        n=int(c["n"]), length=float(c["length_m"]),
        interface_width=float(c["interface_width_m"]),
        temperature=float(c["temperature_K"]), shear_rate=0.0,
        dt=float(c["dt_s"]),
        maximum_mechanical_subdivisions=int(
            c["maximum_mechanical_subdivisions"]),
        source_mode="disabled", source_grain_size=2.4e-6,
        sink_mode="disabled", sink_activation_enthalpy=1.4,
        sink_stress_assisted=False, interface_sink_mode="disabled",
        plastic_history_mode=c.get("plastic_history_mode", "receiver_dormant"),
        thermal_control=c.get("thermal_control", "adiabatic"),
        bath_rate=float(c.get("thermal_bath_rate_s", 0.0)))


def _edge_20_30(audit):
    return [row for row in audit["directional_prices"]
            if row["interface"] == "edge-20-30"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--intervals", type=int, default=50)
    parser.add_argument("--dt", type=float)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    state, runtime, step, load, initial_volume, configuration, provenance = (
        _load_checkpoint(args.checkpoint))
    physics_args = _args_from_configuration(configuration)
    spacing, thickness, _, _, _, _, context = _physics(physics_args)
    dt = float(configuration["dt_s"] if args.dt is None else args.dt)
    context = replace(
        context, shear_rate_s=0.0, front_enabled=False,
        interface_sink_mode="disabled", interface_sink_parameters=None,
        structural_absorption_mode="common_axis",
        structural_absorption_parameters=(
            InterfaceStructuralAbsorptionParameters()))
    endpoint = make_endpoint(state, runtime, load, context)
    before = _directional_audit(endpoint, context, fraction=2e-4)
    records = []
    for interval_index in range(1, args.intervals+1):
        interval = advance_complete_multigrain_interval(endpoint, dt, context)
        endpoint = interval.endpoint
        records.append({
            "interval": interval_index,
            "classification": interval.audit[
                "structural_absorption_classification"],
            "kinetic_fraction": interval.audit[
                "structural_absorption_fraction"],
            "structural_content_change": interval.audit[
                "structural_absorption_content_change"],
            "generated_heat_J": interval.integrals[
                "structural_absorption_heat_J"],
            "relative_first_law_residual": interval.audit[
                "structural_absorption_relative_first_law_residual"],
        })
    after = _directional_audit(endpoint, context, fraction=2e-4)
    strain = np.array([[0.0, .5*endpoint.load], [.5*endpoint.load, 0.0]])
    _, _, cycle = advance_structural_junction_interval(
        endpoint.state,
        StructuralMultigrainRuntime(endpoint.runtime.structural_interfaces),
        dt_s=dt, spacing_m=spacing,
        represented_thickness_m=thickness,
        wall_parameters=context.wall_parameters,
        energy_kwargs=dict(context.energy_kwargs, mean_strain=strain))
    output_checkpoint = args.out/"checkpoint_after_bridge.npz"
    temporary = args.out/"checkpoint_after_bridge.tmp.npz"
    source_commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()
    bridge_configuration = dict(configuration)
    bridge_configuration.update(
        schema="v67-interface-structural-bridge-v1",
        parent_checkpoint=str(args.checkpoint.resolve()),
        stationary_intervals=args.intervals, stationary_interval_s=dt,
        structural_absorption_mode="common_axis")
    _save_checkpoint(
        temporary, endpoint.state, endpoint.runtime, step+args.intervals,
        endpoint.load, initial_volume, bridge_configuration,
        {"source_commit": source_commit,
         "parent_provenance": provenance})
    os.replace(temporary, output_checkpoint)
    accepted = sum(row["classification"].startswith("ACCEPTED")
                   for row in records)
    payload = {
        "schema": "v67-interface-structural-bridge-discriminator-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "parent_checkpoint": str(args.checkpoint.resolve()),
        "parent_checkpoint_sha256": _sha(args.checkpoint),
        "configuration": bridge_configuration,
        "before_edge_20_30": _edge_20_30(before),
        "after_edge_20_30": _edge_20_30(after),
        "interval_records": records,
        "ledger": asdict(endpoint.runtime.ledger),
        "cycle_redistribution_discriminator": {
            "classification": cycle.classification,
            "accepted": cycle.accepted,
            "kinetic_fraction": cycle.kinetic_proposal.reacted_fraction,
            "absolute_proposed_content": (
                cycle.kinetic_proposal.absolute_reacted_content),
            "complete_delta_helmholtz_J": (
                cycle.complete_energy_candidate.helmholtz_J
                -cycle.complete_energy_before.helmholtz_J),
            "low_level_delta_helmholtz_J": (
                cycle.kinetic_proposal.delta_helmholtz_J),
            "generated_heat_J": cycle.generated_heat_J,
        },
        "decision": {
            "accepted_absorption_intervals": accepted,
            "hard_balance_passed": bool(all(
                row["relative_first_law_residual"] < .05
                for row in records if row["classification"].startswith("ACCEPTED"))),
            "supports_unchanged": bool(np.array_equal(
                endpoint.state.supports, state.supports)),
            "grain_ids_unchanged": bool(
                endpoint.state.grain_ids == state.grain_ids),
            "post_reversal_neighbor_competition_reversed": False,
        },
        "artifacts": {
            "checkpoint_after_bridge.npz": _sha(output_checkpoint),
        },
        "claim_boundary": (
            "Stationary common-axis signed-reservoir absorption discriminator; "
            "it does not by itself qualify sustained dynamic DRX."),
    }
    before_by_direction = {
        row["direction"]: row for row in payload["before_edge_20_30"]}
    after_by_direction = {
        row["direction"]: row for row in payload["after_edge_20_30"]}
    # 20 -> 30 is canonical a_to_b and consumes the target; 30 -> 20 is b_to_a.
    before_bias = (before_by_direction["b_to_a"][
        "structural_directional_energy_density_Pa"]
        -before_by_direction["a_to_b"][
            "structural_directional_energy_density_Pa"])
    after_bias = (after_by_direction["b_to_a"][
        "structural_directional_energy_density_Pa"]
        -after_by_direction["a_to_b"][
            "structural_directional_energy_density_Pa"])
    payload["decision"].update(
        before_target_growth_minus_shrink_pressure_Pa=before_bias,
        after_target_growth_minus_shrink_pressure_Pa=after_bias,
        post_reversal_neighbor_competition_reversed=bool(
            np.sign(after_bias) != np.sign(before_bias)),
        incoming_boundary_storage_penalty_change_Pa=(
            after_by_direction["b_to_a"]["structural_energy_component_change_J"][
                "boundary_excess_J"]
            /max(after_by_direction["b_to_a"]["swept_volume_m3"], 1e-300)
            -before_by_direction["b_to_a"]["structural_energy_component_change_J"][
                "boundary_excess_J"]
            /max(before_by_direction["b_to_a"]["swept_volume_m3"], 1e-300)))
    result_path = args.out/"result.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
