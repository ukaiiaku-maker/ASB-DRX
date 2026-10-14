#!/usr/bin/env python3
"""Diagnose V66 dynamic RX reversal on immutable production states.

The calculation reproduces the n16 coupled trajectory and records complete
directional prices before growth, at the first target maximum, after reversal,
and at the terminal state.  Each state is also priced with conserved
common-axis structural interface content.  The structural price is a bounded
diagnostic: it does not mutate the trajectory or claim general 3-D
Frank--Bilby compatibility.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from full_model.analysis.audit_v63_drx_event_energy import (
    energy_delta, grain_partition,
)
from full_model.analysis.run_v66_coupled_rx_asb import (
    _physics, _prepared_state,
)
from full_model.production.complete_front_energy import CompleteFrontEnergy
from full_model.production.complete_multigrain_energy import (
    MultiGrainDissipation, evaluate_joint_multigrain_transaction,
)
from full_model.production.coupled_multigrain_interval import (
    advance_complete_multigrain_interval, make_endpoint,
)
from full_model.production.multigrain_common_state import (
    derive_physical_transfer_proposal, joint_material_transaction,
)
from full_model.production.multigrain_production import _geometric_sweep_weight
from full_model.production.structural_boundary import StructuralBoundaryParameters
from full_model.production.structural_interface_network import (
    JunctionDisclinationParameters, initialize_structural_interfaces,
    interface_lengths_m,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "full_model" / "verification" / "v67_growth_reversal"
ENERGY_FIELDS = tuple(item.name for item in fields(CompleteFrontEnergy))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _components(value) -> dict[str, float]:
    return {name: float(getattr(value, name)) for name in ENERGY_FIELDS} | {
        "helmholtz_J": float(value.helmholtz_J),
        "internal_J": float(value.internal_J),
    }


def _target_volume(state, spacing: float, thickness: float) -> float:
    return float(np.sum(state.supports[1], dtype=np.longdouble)
                 * spacing**2 * thickness)


def _structural_state(state, runtime, spacing, burgers):
    edges = tuple(
        (item.grain_a_id, item.grain_b_id) for item in runtime.interfaces)
    lengths = interface_lengths_m(
        state.supports, runtime.interfaces, state.grain_ids, spacing)
    orientations = np.asarray([
        np.mean(owner.orientation_rad) for owner in state.owners], dtype=float)
    interfaces = initialize_structural_interfaces(
        state.grain_ids, orientations, edges, lengths, burgers)
    return interfaces, lengths, orientations


def _directional_audit(endpoint, context, *, fraction: float) -> dict:
    state = endpoint.state
    runtime = endpoint.runtime
    spacing = context.spacing_m
    thickness = context.represented_thickness_m
    wall = context.wall_parameters
    strain = np.array([[0.0, .5 * endpoint.load],
                       [.5 * endpoint.load, 0.0]])
    base_energy = dict(context.energy_kwargs, mean_strain=strain)
    structural_parameters = StructuralBoundaryParameters(
        burgers_m=wall.burgers_m)
    structural_interfaces, structural_lengths, orientations = _structural_state(
        state, runtime, spacing, wall.burgers_m)
    structural_energy = dict(
        base_energy,
        structural_interfaces=structural_interfaces,
        structural_burgers_m=wall.burgers_m,
        structural_parameters=structural_parameters,
        junction_disclination_parameters=JunctionDisclinationParameters(),
    )
    index = {grain_id: position
             for position, grain_id in enumerate(state.grain_ids)}
    cell_volume = spacing**2 * thickness
    rows = []
    for interface in runtime.interfaces:
        for donor, receiver, label in (
                (interface.grain_a_id, interface.grain_b_id, "a_to_b"),
                (interface.grain_b_id, interface.grain_a_id, "b_to_a")):
            weight = _geometric_sweep_weight(
                state, index[donor], index[receiver], spacing)
            proposal = derive_physical_transfer_proposal(
                state, interface_id=interface.component_id,
                donor_id=donor, receiver_id=receiver,
                requested_fraction=fraction * weight,
                law=context.front_kinetics.transfer_law,
                interval_s=context.maximum_front_substep_s,
                systems=context.systems)
            capacity = joint_material_transaction(state, (proposal,))
            extent = capacity.accepted_fraction_by_interface[
                interface.component_id]
            swept_volume = float(
                np.sum(extent, dtype=np.longdouble) * cell_volume)
            base = evaluate_joint_multigrain_transaction(
                state, capacity, spacing_m=spacing,
                represented_thickness_m=thickness, wall_parameters=wall,
                interval_s=context.maximum_front_substep_s,
                dissipation=MultiGrainDissipation(),
                energy_kwargs=base_energy, bind_state_identity=False)
            structural = evaluate_joint_multigrain_transaction(
                state, capacity, spacing_m=spacing,
                represented_thickness_m=thickness, wall_parameters=wall,
                interval_s=context.maximum_front_substep_s,
                dissipation=MultiGrainDissipation(),
                energy_kwargs=structural_energy, bind_state_identity=False)
            base_delta = energy_delta(
                base.decision.before, base.decision.candidate)
            structural_delta = energy_delta(
                structural.decision.before, structural.decision.candidate)
            rows.append({
                "interface": interface.component_id,
                "direction": label,
                "donor_id": donor,
                "receiver_id": receiver,
                "target_owner_grows": bool(receiver == 20),
                "target_owner_shrinks": bool(donor == 20),
                "swept_volume_m3": swept_volume,
                "base_cold_delta_helmholtz_J": (
                    base.decision.delta_helmholtz_J),
                "base_directional_energy_density_Pa": (
                    base.decision.delta_helmholtz_J
                    / max(swept_volume, 1e-300)),
                "base_energy_component_change_J": base_delta,
                "structural_cold_delta_helmholtz_J": (
                    structural.decision.delta_helmholtz_J),
                "structural_directional_energy_density_Pa": (
                    structural.decision.delta_helmholtz_J
                    / max(swept_volume, 1e-300)),
                "structural_energy_component_change_J": structural_delta,
                "physical_channel_cell_sums_m2": (
                    capacity.physical_channel_cell_sums_m2),
            })
    partition = grain_partition(
        state, spacing, thickness, wall, context.systems, strain)
    required_content = [
        float(length * (orientations[state.grain_ids.index(item.owner_b)]
                        - orientations[state.grain_ids.index(item.owner_a)])
              / wall.burgers_m)
        for item, length in zip(structural_interfaces, structural_lengths)]
    return {
        "load": endpoint.load,
        "physical_time_s": runtime.ledger.physical_time_s,
        "mean_temperature_K": float(np.mean(
            endpoint.observables["common_temperature_K"])),
        "mean_shear_stress_Pa": float(np.mean(
            endpoint.observables["common_stress_tensor_Pa"][..., 0, 1])),
        "target_volume_m3": _target_volume(state, spacing, thickness),
        "grain_partition": partition,
        "directional_prices": rows,
        "structural_diagnostic": {
            "scope": "common-axis conserved-content price; no trajectory mutation",
            "interface_lengths_m": structural_lengths.tolist(),
            "owner_orientations_rad": orientations.tolist(),
            "required_signed_integrated_content": required_content,
            "production_coupled_interval_carries_this_state": False,
        },
    }


def run(output: Path, *, steps: int, snapshot_steps: tuple[int, ...]) -> dict:
    args = argparse.Namespace(
        n=16, length=4.8e-6, interface_width=6e-7,
        temperature=1000.0, shear_rate=4e4, dt=2e-9, steps=steps,
        thermal_control="adiabatic", bath_rate=5e9,
        maximum_mechanical_subdivisions=10,
        source_mode="disabled", source_grain_size=2.4e-6,
        sink_mode="disabled", sink_activation_enthalpy=1.4,
        sink_stress_assisted=False)
    physics = _physics(args)
    spacing, thickness, _, _, _, _, context = physics
    state, runtime, preparation = _prepared_state(args, physics)
    endpoint = make_endpoint(state, runtime, 0.0, context)
    initial_target = _target_volume(state, spacing, thickness)
    requested = set(snapshot_steps) | {0, steps}
    snapshots = {"0": _directional_audit(endpoint, context, fraction=2e-4)}
    growth = []
    directions = []
    for step in range(1, steps + 1):
        interval = advance_complete_multigrain_interval(endpoint, args.dt, context)
        endpoint = interval.endpoint
        current = _target_volume(endpoint.state, spacing, thickness)
        growth.append((current - initial_target) / initial_target)
        directions.append(interval.audit["front_directions"])
        if step in requested:
            snapshots[str(step)] = _directional_audit(
                endpoint, context, fraction=2e-4)
    peak_index = int(np.argmax(growth)) + 1
    if str(peak_index) not in snapshots:
        raise RuntimeError(
            "peak step was not requested; rerun with the identified peak snapshot")
    peak = growth[peak_index - 1]
    terminal = growth[-1]
    result = {
        "schema": "v67-production-growth-reversal-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
        "configuration": {
            "grid": [args.n, args.n], "length_m": args.length,
            "interface_width_m": args.interface_width,
            "temperature_K": args.temperature,
            "shear_rate_s": args.shear_rate, "dt_s": args.dt,
            "steps": steps, "source_mode": args.source_mode,
            "sink_mode": args.sink_mode,
            "front_energy": "authoritative complete multigrain functional",
        },
        "preparation": preparation,
        "snapshots": snapshots,
        "trajectory": {
            "peak_step": peak_index,
            "peak_target_growth_fraction": peak,
            "terminal_target_growth_fraction": terminal,
            "first_target_recession_step": next(
                (i + 1 for i, value in enumerate(growth) if value < 0.0), None),
            "directions_at_peak": directions[peak_index - 1],
            "directions_terminal": directions[-1],
        },
        "operator_exercise": {
            "arrhenius_EXP_floor_Peierls_Taylor_mechanics": True,
            "complete_front_transaction": True,
            "stationary_structural_absorption": False,
            "structural_junction_reaction": False,
            "pair_source": False,
            "neutral_sink": False,
        },
        "interpretation_boundary": (
            "The audit identifies complete-energy direction and the effect of "
            "pricing conserved common-axis structural content. It does not "
            "establish new-boundary birth, arbitrary-network compatibility, "
            "or sustained DRX."),
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "result.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument(
        "--snapshot-steps", type=int, nargs="+",
        default=(1, 279, 280, 450, 900))
    args = parser.parse_args()
    result = run(args.out, steps=args.steps,
                 snapshot_steps=tuple(args.snapshot_steps))
    path = args.out / "result.json"
    print(json.dumps({
        "sha256": _sha(path),
        "trajectory": result["trajectory"],
        "snapshots": sorted(result["snapshots"], key=int),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
