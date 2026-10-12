#!/usr/bin/env python3
"""Price feasible directed interface transfers in the complete functional."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from full_model.analysis.audit_v63_drx_event_energy import (
    build_kinetics, digest, energy_delta, grain_partition,
)
from full_model.analysis.postprocess_v59_physical_asb import _wall_parameters
from full_model.analysis.run_v58_three_grain_production import _load_checkpoint
from full_model.production.complete_multigrain_energy import (
    MultiGrainDissipation, evaluate_joint_multigrain_transaction,
)
from full_model.production.multigrain_common_state import (
    derive_physical_transfer_proposal, joint_material_transaction,
)
from full_model.production.multigrain_production import (
    _geometric_sweep_weight, advance_energy_qualified_mechanics,
    advance_multigrain_front_interval,
)
from full_model.production.tensorial_nye import bcc_four_family_systems


def run(checkpoint: Path, output: Path,
        fractions: tuple[float, ...]) -> dict:
    state, runtime, step, gamma, _, configuration, provenance = _load_checkpoint(
        checkpoint)
    n = int(configuration["n"]); spacing = float(configuration["length_m"])/n
    thickness = 2.0*2.48e-10; dt = float(configuration["dt_s"])
    shear_rate = float(configuration["shear_rate_s"])
    systems = bcc_four_family_systems(); wall = _wall_parameters(configuration, spacing)
    kinetics = build_kinetics(configuration, wall)
    gamma_after = gamma+shear_rate*dt
    strain0 = np.array([[0.0, .5*gamma], [.5*gamma, 0.0]])
    strain1 = np.array([[0.0, .5*gamma_after], [.5*gamma_after, 0.0]])
    energy_options = {
        "mean_strain": strain1, "topologies": (), "systems": systems,
        "phase_barrier_J_m3": 5e6, "phase_gradient_J_m": 5e-7,
        "boundary_line_energy_J_m": wall.line_energy_J_m,
        "boundary_junction_energy_J_m": wall.junction_energy_J_m,
        "reference_temperature_K": float(configuration["temperature_K"]),
    }
    mechanics = advance_energy_qualified_mechanics(
        state, mean_strain_before=strain0, mean_strain_candidate=strain1,
        systems=systems, topologies=(), wall_parameters=wall, dt_s=dt,
        represented_thickness_m=thickness, energy_kwargs=energy_options,
        maximum_relative_first_law_residual=.05, maximum_subdivisions=10)
    prepared = mechanics.state
    index = {grain_id: i for i, grain_id in enumerate(prepared.grain_ids)}
    cell_volume = spacing**2*thickness
    directions = []
    for interface in runtime.interfaces:
        for donor, receiver, label in (
                (interface.grain_a_id, interface.grain_b_id, "a_to_b"),
                (interface.grain_b_id, interface.grain_a_id, "b_to_a")):
            weight = _geometric_sweep_weight(
                prepared, index[donor], index[receiver], spacing)
            rows = []
            for fraction in fractions:
                proposal = derive_physical_transfer_proposal(
                    prepared, interface_id=interface.component_id,
                    donor_id=donor, receiver_id=receiver,
                    requested_fraction=float(fraction)*weight,
                    law=kinetics.transfer_law, interval_s=dt, systems=systems)
                capacity = joint_material_transaction(prepared, (proposal,))
                evaluated = evaluate_joint_multigrain_transaction(
                    prepared, capacity, spacing_m=spacing,
                    represented_thickness_m=thickness,
                    wall_parameters=wall, interval_s=dt,
                    dissipation=MultiGrainDissipation(),
                    energy_kwargs=energy_options, absolute_tolerance_J=0.0)
                swept = sum(capacity.accepted_fraction_by_interface.values())
                swept_volume = float(np.sum(swept, dtype=np.longdouble)*cell_volume)
                delta = energy_delta(
                    evaluated.decision.before, evaluated.decision.candidate)
                rows.append({
                    "virtual_fraction": fraction,
                    "capacity_feasible": capacity.capacity_feasible,
                    "accepted_fraction_cell_sum": float(np.sum(
                        swept, dtype=np.longdouble)),
                    "swept_volume_m3": swept_volume,
                    "cold_delta_helmholtz_J": evaluated.decision.delta_helmholtz_J,
                    "directional_energy_density_Pa": (
                        evaluated.decision.delta_helmholtz_J
                        /max(swept_volume, 1e-300)),
                    "energy_component_change_J": delta,
                    "physical_channel_cell_sums_m2": (
                        capacity.physical_channel_cell_sums_m2),
                    "material_sink_export_J": evaluated.decision.material_sink_export_J,
                })
            reference = rows[-1]["directional_energy_density_Pa"]
            for row in rows:
                row["relative_slope_difference_from_smallest_extent"] = abs(
                    row["directional_energy_density_Pa"]-reference)/max(
                        abs(reference), 1e-300)
            smallest = rows[-1]
            receiver_partition_before = grain_partition(
                prepared, spacing, thickness, wall, systems, strain1)[str(receiver)]
            # The smallest virtual state is analysis-only and is never published.
            proposal = derive_physical_transfer_proposal(
                prepared, interface_id=interface.component_id,
                donor_id=donor, receiver_id=receiver,
                requested_fraction=float(fractions[-1])*weight,
                law=kinetics.transfer_law, interval_s=dt, systems=systems)
            smallest_capacity = joint_material_transaction(prepared, (proposal,))
            receiver_partition_after = grain_partition(
                smallest_capacity.candidate, spacing, thickness, wall, systems,
                strain1)[str(receiver)]
            directions.append({
                "interface": interface.component_id, "direction": label,
                "donor_id": donor, "receiver_id": receiver,
                "virtual_sequence": rows,
                "receiver_smallest_extent_change": {
                    name: receiver_partition_after[name]-receiver_partition_before[name]
                    for name in ("volume_m3", "defect_storage_J",
                                 "defect_storage_mean_J_m3",
                                 "recoverable_elastic_share_J", "total_line_m")},
                "selection_semantics": (
                    "independent feasible directional derivative only; values "
                    "are never summed into a competing joint event"),
            })
    evolved, evolved_runtime, realized = advance_multigrain_front_interval(
        prepared, replace(runtime, ledger=replace(
            runtime.ledger,
            cumulative_mechanical_external_work_J=(
                runtime.ledger.cumulative_mechanical_external_work_J
                +mechanics.external_work_J),
            cumulative_mechanical_internal_energy_change_J=(
                runtime.ledger.cumulative_mechanical_internal_energy_change_J
                +mechanics.internal_energy_change_J),
            cumulative_mechanical_first_law_residual_J=(
                runtime.ledger.cumulative_mechanical_first_law_residual_J
                +mechanics.first_law_residual_J))),
        kinetics=kinetics, dt_s=dt,
        maximum_substep_s=float(configuration["front_maximum_substep_s"]),
        spacing_m=spacing, represented_thickness_m=thickness,
        wall_parameters=wall, energy_kwargs=energy_options,
        applied_shear_rate_s=shear_rate, systems=systems,
        heat_deposition_mode=configuration["front_heat_deposition"])
    result = {
        "schema": "asb-drx-v64-directed-complete-energy-v1",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": digest(checkpoint),
        "source_commit": (provenance or {}).get("source_commit"),
        "prepared_state": "after exact mechanical stage and before front",
        "step": step+1, "virtual_fractions": list(fractions),
        "directed_interface_transfers": directions,
        "realized_joint_event": [{
            "classification": item.classification,
            "directions": item.direction_by_interface,
            "pressures_Pa": item.pressure_by_interface_Pa,
            "accepted_fraction_cell_sum": item.accepted_fraction_by_interface,
            "generated_heat_J": (None if item.energy_decision is None else
                                 item.energy_decision.generated_heat_J),
            "cold_delta_helmholtz_J": (None if item.energy_decision is None else
                                       item.energy_decision.delta_helmholtz_J),
        } for item in realized],
        "joint_donor_semantics": (
            "the production event is the sole jointly capacity-limited accepted "
            "transaction; independent directional probes do not spend material"),
        "interpretation_boundary": (
            "directional complete-functional evidence for prepared-boundary "
            "selection; no spontaneous grain birth or scalar-density ranking claim"),
    }
    smallest_slopes = {
        f"{row['interface']}:{row['direction']}":
            row["virtual_sequence"][-1]["directional_energy_density_Pa"]
        for row in directions
    }
    by_interface = {}
    for row in directions:
        by_interface.setdefault(row["interface"], {})[row["direction"]] = (
            row["virtual_sequence"][-1]["directional_energy_density_Pa"])
    result["summary"] = {
        "smallest_extent_directional_energy_density_Pa": smallest_slopes,
        "downhill_direction_count": sum(value < 0.0
                                          for value in smallest_slopes.values()),
        "uphill_direction_count": sum(value > 0.0
                                        for value in smallest_slopes.values()),
        "interfaces_downhill_in_both_directions": sorted(
            key for key, value in by_interface.items()
            if len(value) == 2 and all(item < 0.0 for item in value.values())),
        "maximum_relative_slope_change_over_virtual_sequence": max(
            row["virtual_sequence"][0][
                "relative_slope_difference_from_smallest_extent"]
            for row in directions),
        "interpretation": (
            "a downhill directional derivative establishes local energetic "
            "availability, not unique direction selection; the production "
            "joint event additionally applies mobility and donor competition"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fractions", type=float, nargs="+",
                        default=(2e-4, 1e-4, 5e-5))
    args = parser.parse_args()
    result = run(args.checkpoint, args.output, tuple(args.fractions))
    print(json.dumps({"sha256": digest(args.output),
                      "directions": len(result["directed_interface_transfers"])}))


if __name__ == "__main__":
    main()
