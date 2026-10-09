#!/usr/bin/env python3
"""Replay one accepted macro interval and ledger its current-state DRX event."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields, replace
import json
from pathlib import Path

import numpy as np

from full_model.analysis.audit_v63_phase_sensitive_fields import digest
from full_model.analysis.postprocess_v59_physical_asb import _wall_parameters
from full_model.analysis.run_v58_three_grain_production import _load_checkpoint
from full_model.production.arrhenius_kinetics import ActivatedProcess, EV_J
from full_model.production.common_tensorial_wall import (
    wall_total_free_energy_density_J_m3,
)
from full_model.production.complete_front_energy import CompleteFrontEnergy
from full_model.production.complete_multigrain_energy import (
    evaluate_complete_multigrain_energy,
)
from full_model.production.multigrain_common_state import (
    LINE_FIELDS, PhysicalTransferLaw, multigrain_state_digest,
    reconstruct_multigrain_common,
)
from full_model.production.multigrain_production import (
    MultiGrainFrontKinetics, advance_energy_qualified_mechanics,
    advance_multigrain_front_interval,
)
from full_model.production.nonlocal_elasticity import (
    elastic_energy_density, solve_periodic_eigenstrain,
)
from full_model.production.tensorial_nye import bcc_four_family_systems


ENERGY_FIELDS = tuple(item.name for item in fields(CompleteFrontEnergy))


def energy_components(energy: CompleteFrontEnergy) -> dict:
    return {name: float(getattr(energy, name)) for name in ENERGY_FIELDS} | {
        "helmholtz_J": float(energy.helmholtz_J),
        "internal_J": float(energy.internal_J),
    }


def energy_delta(before: CompleteFrontEnergy,
                 candidate: CompleteFrontEnergy) -> dict:
    a, b = energy_components(before), energy_components(candidate)
    return {name: b[name]-a[name] for name in a}


def grain_partition(state, spacing_m, thickness_m, wall, systems,
                    mean_strain) -> dict:
    common, _ = reconstruct_multigrain_common(state, spacing_m)
    beta = np.asarray(common.beta_p)[..., :2, :2]
    eigenstrain = .5*(beta+np.swapaxes(beta, -1, -2))
    stress, strain = solve_periodic_eigenstrain(
        eigenstrain, mean_strain, spacing_m, wall.c11_Pa, wall.c12_Pa,
        wall.c44_Pa, iterations=wall.elastic_iterations)
    elastic = elastic_energy_density(stress, strain, eigenstrain)
    cell_volume = spacing_m**2*thickness_m
    rows = {}
    for grain_id, support, owner in zip(
            state.grain_ids, state.supports, state.owners):
        support = np.asarray(support, dtype=float)
        defect = wall_total_free_energy_density_J_m3(
            owner, wall, (), systems)
        total_line = sum(np.sum(np.asarray(getattr(owner, name)), axis=2)
                         for name in LINE_FIELDS)
        signed_family = sum(
            np.asarray(getattr(owner, plus))-np.asarray(getattr(owner, minus))
            for plus, minus in (("mobile_plus_m2", "mobile_minus_m2"),
                                ("forest_plus_m2", "forest_minus_m2"),
                                ("wall_plus_m2", "wall_minus_m2")))
        volume = float(np.sum(support, dtype=np.longdouble)*cell_volume)
        rows[str(grain_id)] = {
            "volume_m3": volume,
            "defect_storage_J": float(np.sum(
                support*defect, dtype=np.longdouble)*cell_volume),
            "defect_storage_mean_J_m3": float(np.sum(
                support*defect, dtype=np.longdouble)*cell_volume/max(volume, 1e-300)),
            "recoverable_elastic_share_J": float(np.sum(
                support*elastic, dtype=np.longdouble)*cell_volume),
            "total_line_m": float(np.sum(
                support*total_line, dtype=np.longdouble)*cell_volume),
            "signed_family_line_m": (
                np.sum(support[..., None]*signed_family, axis=(0, 1),
                       dtype=np.longdouble)*cell_volume).astype(float).tolist(),
            "orientation_weighted_mean_rad": float(np.sum(
                support*owner.orientation_rad, dtype=np.longdouble)
                /max(float(np.sum(support, dtype=np.longdouble)), 1e-300)),
        }
    return rows


def build_kinetics(configuration: dict, wall) -> MultiGrainFrontKinetics:
    override = (float(configuration["temperature_K"])
                if configuration.get("front_temperature_mode") == "frozen"
                else None)
    return MultiGrainFrontKinetics(
        ActivatedProcess("V58 multi-grain HAGB",
                         float(configuration["front_attempt_frequency_s"]),
                         0.0, float(configuration["front_attempt_frequency_s"])),
        PhysicalTransferLaw(.55, .08, .02), .35*EV_J, 1e9,
        2.0, 1.5, .10, wall.burgers_m**3, wall.burgers_m,
        virtual_fraction=2e-4,
        maximum_fraction_per_step=float(
            configuration["front_maximum_fraction_per_step"]),
        closure_fraction=.05, maximum_backtracks=14,
        temperature_override_K=override,
        temperature_resolution=configuration.get(
            "front_temperature_resolution", "global_mean"))


def run(before_path: Path, target_path: Path, output: Path) -> dict:
    before, runtime, step, gamma, _, configuration, provenance = (
        _load_checkpoint(before_path))
    target, target_runtime, target_step, target_gamma, _, target_configuration, _ = (
        _load_checkpoint(target_path))
    if target_step != step+1 or configuration != target_configuration:
        raise ValueError("event replay requires consecutive identical-configuration states")
    n = int(configuration["n"])
    spacing = float(configuration["length_m"])/n
    thickness = 2.0*2.48e-10
    dt = float(configuration["dt_s"])
    shear_rate = float(configuration["shear_rate_s"])
    systems = bcc_four_family_systems()
    wall = _wall_parameters(configuration, spacing)
    kinetics = build_kinetics(configuration, wall)
    gamma_after = gamma+shear_rate*dt
    strain_before = np.array([[0.0, .5*gamma], [.5*gamma, 0.0]])
    strain_after = np.array([[0.0, .5*gamma_after], [.5*gamma_after, 0.0]])
    energy_options = dict(
        mean_strain=strain_after, topologies=(), systems=systems,
        phase_barrier_J_m3=5e6, phase_gradient_J_m=5e-7,
        boundary_line_energy_J_m=wall.line_energy_J_m,
        boundary_junction_energy_J_m=wall.junction_energy_J_m,
        reference_temperature_K=float(configuration["temperature_K"]))
    mechanical = advance_energy_qualified_mechanics(
        before, mean_strain_before=strain_before,
        mean_strain_candidate=strain_after, systems=systems, topologies=(),
        wall_parameters=wall, dt_s=dt, represented_thickness_m=thickness,
        energy_kwargs=energy_options, maximum_relative_first_law_residual=.05,
        maximum_subdivisions=10)
    after_mechanics = mechanical.state
    runtime = replace(runtime, ledger=replace(
        runtime.ledger,
        cumulative_mechanical_external_work_J=(
            runtime.ledger.cumulative_mechanical_external_work_J
            +mechanical.external_work_J),
        cumulative_mechanical_internal_energy_change_J=(
            runtime.ledger.cumulative_mechanical_internal_energy_change_J
            +mechanical.internal_energy_change_J),
        cumulative_mechanical_first_law_residual_J=(
            runtime.ledger.cumulative_mechanical_first_law_residual_J
            +mechanical.first_law_residual_J),
        maximum_mechanical_relative_first_law_residual=max(
            runtime.ledger.maximum_mechanical_relative_first_law_residual,
            mechanical.relative_first_law_residual)))
    evolved, evolved_runtime, decisions = advance_multigrain_front_interval(
        after_mechanics, runtime, kinetics=kinetics, dt_s=dt,
        maximum_substep_s=float(configuration["front_maximum_substep_s"]),
        spacing_m=spacing, represented_thickness_m=thickness,
        wall_parameters=wall, energy_kwargs=energy_options,
        applied_shear_rate_s=shear_rate, systems=systems,
        heat_deposition_mode=configuration["front_heat_deposition"])
    replay_digest = multigrain_state_digest(evolved)
    target_digest = multigrain_state_digest(target)
    if replay_digest != target_digest:
        raise RuntimeError("replayed production state does not match target checkpoint")
    if (not np.isclose(evolved_runtime.ledger.physical_time_s,
                       target_runtime.ledger.physical_time_s, rtol=0.0, atol=1e-18)
            or not np.isclose(gamma_after, target_gamma, rtol=0.0, atol=1e-15)):
        raise RuntimeError("replayed production clock does not match target")
    cell_volume = spacing**2*thickness
    before_partition = grain_partition(
        after_mechanics, spacing, thickness, wall, systems, strain_after)
    after_partition = grain_partition(
        evolved, spacing, thickness, wall, systems, strain_after)
    grain_changes = {}
    for key in before_partition:
        grain_changes[key] = {
            name: after_partition[key][name]-before_partition[key][name]
            for name in ("volume_m3", "defect_storage_J",
                         "recoverable_elastic_share_J", "total_line_m")}
    atomic = []
    for index, decision in enumerate(decisions):
        energy = decision.energy_decision
        atomic.append({
            "atomic_index": index, "accepted": decision.accepted,
            "classification": decision.classification,
            "directions": decision.direction_by_interface,
            "pressure_Pa": decision.pressure_by_interface_Pa,
            "independent_pressure_Pa": decision.independent_pressure_by_interface_Pa,
            "selected_velocity_m_s": decision.selected_velocity_by_interface_m_s,
            "accepted_fraction_cell_sum": decision.accepted_fraction_by_interface,
            "realized_volume_m3_by_interface": {
                key: value*cell_volume
                for key, value in decision.accepted_fraction_by_interface.items()},
            "energy_before_J": (None if energy is None else
                                energy_components(energy.before)),
            "energy_after_J": (None if energy is None else
                               energy_components(energy.candidate)),
            "energy_component_change_J": (None if energy is None else
                                          energy_delta(energy.before,
                                                       energy.candidate)),
            "cold_delta_helmholtz_J": (None if energy is None else
                                       energy.delta_helmholtz_J),
            "generated_heat_J": (None if energy is None else
                                 energy.generated_heat_J),
            "material_sink_export_J": (None if energy is None else
                                       energy.material_sink_export_J),
            "first_law_residual_J": (None if energy is None else
                                     energy.first_law_residual_J),
            "dissipation_residual_J": (None if energy is None else
                                       energy.dissipation_residual_J),
        })
    total_before = evaluate_complete_multigrain_energy(
        after_mechanics, spacing_m=spacing, represented_thickness_m=thickness,
        wall_parameters=wall, **energy_options)
    total_after = evaluate_complete_multigrain_energy(
        evolved, spacing_m=spacing, represented_thickness_m=thickness,
        wall_parameters=wall, **energy_options)
    ledger_before = after_mechanics.ledger
    ledger_after = evolved.ledger
    result = {
        "schema": "asb-drx-v63-current-state-drx-event-energy-v1",
        "source_scope": (
            "exact replay of unchanged 1c0cb4c physical operators from the "
            "checksum-bound before checkpoint"),
        "before_checkpoint": str(before_path.resolve()),
        "before_checkpoint_sha256": digest(before_path),
        "target_checkpoint": str(target_path.resolve()),
        "target_checkpoint_sha256": digest(target_path),
        "parent_source_commit": (provenance or {}).get("source_commit"),
        "step": target_step, "physical_time_s": target_runtime.ledger.physical_time_s,
        "state_digest_replay_exact": True,
        "replayed_state_digest": replay_digest,
        "mechanical_stage": {
            "external_work_J": mechanical.external_work_J,
            "internal_energy_change_J": mechanical.internal_energy_change_J,
            "first_law_residual_J": mechanical.first_law_residual_J,
            "relative_first_law_residual": mechanical.relative_first_law_residual,
            "accepted_operator_intervals": len(mechanical.operator_decisions),
            "internal_substeps": sum(
                value.maximum_substeps for value in mechanical.operator_decisions),
        },
        "front_atomic_events": atomic,
        "front_total_energy_before_J": energy_components(total_before),
        "front_total_energy_after_J": energy_components(total_after),
        "front_total_energy_component_change_J": energy_delta(
            total_before, total_after),
        "grain_partition_before_front": before_partition,
        "grain_partition_after_front": after_partition,
        "grain_partition_front_changes": grain_changes,
        "physical_transfer_ledger_change": {
            name: float(getattr(ledger_after, name)-getattr(ledger_before, name))
            for name in ("fresh_sweep_fraction", "revisit_sweep_fraction",
                         "annihilated_line_cell_sum_m2",
                         "external_sink_line_cell_sum_m2",
                         "accepted_material_fraction")},
        "interpretation_boundary": (
            "current-state energy and realized transfer are reported; this "
            "does not relabel the growing prepared grain as spontaneous birth"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.before, args.target, args.output)
    print(json.dumps({
        "sha256": digest(args.output),
        "step": result["step"],
        "state_digest_replay_exact": result["state_digest_replay_exact"],
        "grain_partition_front_changes": result[
            "grain_partition_front_changes"],
        "front_total_energy_component_change_J": result[
            "front_total_energy_component_change_J"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
