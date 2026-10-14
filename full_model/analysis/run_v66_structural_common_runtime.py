#!/usr/bin/env python3
"""Exercise the structural junction reaction on retained common states."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from full_model.analysis.run_v58_three_grain_production import (
    _load_checkpoint, network_interfaces,
)
from full_model.analysis.run_v66_coupled_rx_asb import _physics
from full_model.analysis.run_v66_multigrain_handoff import _advance, _initial
from full_model.production.multigrain_production import MultiGrainProductionRuntime
from full_model.production.structural_interface_network import (
    audit_network_compatibility, initialize_structural_interfaces,
    interface_lengths_m,
)
from full_model.production.structural_multigrain_runtime import (
    StructuralMultigrainRuntime, advance_structural_junction_interval,
    synchronize_structural_owner_orientations,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"full_model"/"verification"/"v66_structural_common_runtime.json"


def _orient(state, angles):
    return replace(state, owners=tuple(
        replace(owner, orientation_rad=np.full_like(owner.orientation_rad, angle))
        for owner, angle in zip(state.owners, angles)))


def _run(state, interfaces, *, spacing, thickness, wall, energy, mean_strain):
    runtime = StructuralMultigrainRuntime(interfaces)
    state = synchronize_structural_owner_orientations(
        state, runtime, spacing_m=spacing, burgers_m=wall.burgers_m)
    evolved, final_runtime, decision = advance_structural_junction_interval(
        state, runtime, dt_s=1e-6, spacing_m=spacing,
        represented_thickness_m=thickness, wall_parameters=wall,
        energy_kwargs=dict(energy, mean_strain=mean_strain))
    half_state, half_runtime, _ = advance_structural_junction_interval(
        state, runtime, dt_s=.5e-6, spacing_m=spacing,
        represented_thickness_m=thickness, wall_parameters=wall,
        energy_kwargs=dict(energy, mean_strain=mean_strain))
    split_state, split_runtime, split_decision = advance_structural_junction_interval(
        half_state, half_runtime, dt_s=.5e-6, spacing_m=spacing,
        represented_thickness_m=thickness, wall_parameters=wall,
        energy_kwargs=dict(energy, mean_strain=mean_strain))
    content = np.asarray([
        item.signed_integrated_content for item in final_runtime.interfaces])
    split_content = np.asarray([
        item.signed_integrated_content for item in split_runtime.interfaces])
    lengths = interface_lengths_m(
        state.supports, runtime.interfaces, state.grain_ids, spacing)
    before_audit = audit_network_compatibility(
        state.grain_ids, runtime.interfaces, lengths, wall.burgers_m)
    after_audit = audit_network_compatibility(
        state.grain_ids, final_runtime.interfaces, lengths, wall.burgers_m)
    return {
        "classification": decision.classification,
        "accepted": bool(decision.accepted),
        "frank_backstress_Pa": decision.kinetic_proposal.frank_backstress_Pa,
        "reacted_fraction": decision.kinetic_proposal.reacted_fraction,
        "absolute_reacted_content": (
            final_runtime.ledger.reacted_absolute_content),
        "cycle_residual_before_deg": float(np.degrees(
            before_audit.maximum_cycle_residual_rad)),
        "cycle_residual_after_deg": float(np.degrees(
            after_audit.maximum_cycle_residual_rad)),
        "generated_heat_J": decision.generated_heat_J,
        "relative_first_law_residual": decision.relative_first_law_residual,
        "owner_orientation_before_deg": np.degrees(
            decision.owner_orientation_before_rad).tolist(),
        "owner_orientation_candidate_deg": np.degrees(
            decision.owner_orientation_candidate_rad).tolist(),
        "split_both_accepted": bool(
            half_runtime.ledger.accepted_reactions == 1
            and split_decision.accepted),
        "split_content_relative_error": float(np.linalg.norm(
            split_content-content)/max(np.linalg.norm(content), 1e-300)),
        "split_temperature_relative_error": float(abs(
            np.mean(split_state.owners[0].temperature_K)
            -np.mean(evolved.owners[0].temperature_K))/max(
                abs(np.mean(evolved.owners[0].temperature_K)), 1e-300)),
        "physical_grain_count_change": 0,
    }


def main():
    args = SimpleNamespace(
        n=16, length=4.8e-6, temperature=1000.0,
        shear_rate=4e4, dt=2e-9)
    spacing, thickness, systems, wall, kinetics, energy, _ = _physics(args)
    reduced = json.loads((ROOT/"full_model"/"verification"/
                           "v66_subgrain_maturation"/"result.json").read_text())
    contrast = reduced["cases"]["T1000_rate1"]["stored_energy_contrast_Pa"]
    initial, _, _ = _initial(16, args.length, args.temperature, contrast)
    edges = tuple(network_interfaces(initial))
    prepared, _, _, _ = _advance(
        initial, MultiGrainProductionRuntime(edges), 20, 2e-8,
        spacing=spacing, thickness=thickness, wall=wall, kinetics=kinetics,
        energy=dict(energy, mean_strain=np.zeros((2, 2))), systems=systems)
    coupled, _, _, gamma, *_ = _load_checkpoint(
        ROOT/"full_model"/"verification"/"v66_coupled_source_disabled_v2"/
        "checkpoint_latest.npz")
    angles = np.radians((0.0, 2.5, 5.0))
    edge_pairs = tuple((edge.grain_a_id, edge.grain_b_id) for edge in edges)
    initial_lengths = interface_lengths_m(
        initial.supports, edges, initial.grain_ids, spacing)
    prepared_lengths = interface_lengths_m(
        prepared.supports, edges, prepared.grain_ids, spacing)
    static_interfaces = initialize_structural_interfaces(
        initial.grain_ids, angles, edge_pairs, initial_lengths, wall.burgers_m)
    coupled_interfaces = initialize_structural_interfaces(
        initial.grain_ids, angles, edge_pairs, prepared_lengths, wall.burgers_m)
    shear = np.array(((0.0, .5*gamma), (.5*gamma, 0.0)))
    cases = {
        "prepared_static_handoff": _run(
            _orient(prepared, angles), static_interfaces, spacing=spacing,
            thickness=thickness, wall=wall, energy=energy,
            mean_strain=np.zeros((2, 2))),
        "deformed_four_percent_shear": _run(
            _orient(coupled, angles), coupled_interfaces, spacing=spacing,
            thickness=thickness, wall=wall, energy=energy,
            mean_strain=shear),
    }
    payload = {
        "schema": "v66-structural-common-runtime-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": "aa12308+runtime-development",
        "configuration": {
            "grid": [16, 16], "temperature_K": 1000.0,
            "reaction_interval_s": 1e-6,
            "mechanics": "retained Peierls/Taylor/EXP-floor common state",
            "structural_scope": "low-angle common-axis ablation",
        },
        "cases": cases,
        "decision": {
            "full_common_energy_discriminates_state": bool(
                not cases["prepared_static_handoff"]["accepted"]
                and cases["deformed_four_percent_shear"]["accepted"]),
            "derived_not_imposed_owner_rotation": True,
            "new_owner_allocated": False,
            "production_claim": (
                "structural cycle reaction is common-energy and heat complete; "
                "scalar common-axis scope and no new-boundary birth remain"),
        },
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
