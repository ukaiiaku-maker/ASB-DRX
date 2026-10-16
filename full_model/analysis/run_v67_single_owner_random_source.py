#!/usr/bin/env python3
"""Production-derived one-owner test of random sources and wall rotation.

This is a prepared mechanism control, not an endogenous grain-birth claim.
One retained material owner is extended over the periodic section so the
calculation can resolve intragranular topology without the n16 polycrystal's
2--20-cell owner interiors.  No orientation, density, or plastic-history
field is reset.  The ordinary production mechanics, complete-energy random
Hall--Petch pair source, and ordered-wall orientation transaction share one
physical clock.
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
    _load_checkpoint, _save_checkpoint,
)
from full_model.analysis.run_v66_coupled_rx_asb import _physics
from full_model.analysis.run_v67_interface_structural_bridge import (
    _args_from_configuration,
)
from full_model.production.arrhenius_dislocation_sources import (
    ArrheniusSourceParameters,
)
from full_model.production.common_subgrain_recognition import (
    recognize_common_orientation_plateau,
)
from full_model.production.common_tensorial_wall import CommonWallState
from full_model.production.collective_wall_nucleation import (
    CollectiveWallNucleationParameters,
)
from full_model.production.coupled_multigrain_interval import (
    advance_complete_multigrain_interval, make_endpoint,
)
from full_model.production.multigrain_common_state import MultiGrainCommonState
from full_model.production.implicit_wall_order_relaxation import (
    ImplicitWallOrderRelaxationParameters,
)
from full_model.production.compatible_wall_reorganization import (
    CompatibleWallReorganizationParameters,
)
from full_model.production.conservative_wall_glide import (
    ConservativeWallGlideParameters,
)
from full_model.production.multigrain_production import MultiGrainProductionRuntime
from full_model.production.orientation_wall_compatibility import (
    evaluate_orientation_wall_compatibility, ordered_wall_nye_from_common,
)
from full_model.production.wall_orientation_relaxation import (
    WallOrientationRelaxationParameters,
)


ROOT = Path(__file__).resolve().parents[2]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _density(owner):
    return sum(np.asarray(getattr(owner, name)) for name in (
        "mobile_plus_m2", "mobile_minus_m2", "forest_plus_m2",
        "forest_minus_m2", "wall_plus_m2", "wall_minus_m2"
    )).sum(axis=2)+np.sum(owner.junction_m2, axis=2)


def _serial_recognition(value):
    return {key: item for key, item in value.items()
            if not isinstance(item, np.ndarray)}


def _recognitions(owner, systems, spacing):
    domain = np.ones(owner.orientation_rad.shape, dtype=bool)
    density = _density(owner)
    plastic = recognize_common_orientation_plateau(
        owner.orientation_rad, owner.family_nye_m1, owner.wall_order,
        density, spacing, domain_mask=domain)
    ordered = ordered_wall_nye_from_common(owner, systems)
    structural = recognize_common_orientation_plateau(
        owner.orientation_rad, ordered, owner.wall_order, density, spacing,
        domain_mask=domain)
    combined = recognize_common_orientation_plateau(
        owner.orientation_rad,
        np.sum(owner.family_nye_m1, axis=2)+ordered,
        owner.wall_order, density, spacing, domain_mask=domain)
    return {
        "plastic_incompatibility_nye": _serial_recognition(plastic),
        "ordered_structural_nye": _serial_recognition(structural),
        "combined_declared_nye": _serial_recognition(combined),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--owner-id", type=int, default=10)
    parser.add_argument("--intervals", type=int, default=10)
    parser.add_argument("--dt", type=float, default=8e-4)
    parser.add_argument("--shear-rate", type=float, default=10.0)
    parser.add_argument("--source-seed", type=int, default=67)
    parser.add_argument("--nucleation-seed", type=int)
    parser.add_argument(
        "--source-mode", choices=("random_bulk", "random_discrete_bulk"),
        default="random_bulk")
    parser.add_argument("--source-site-density-m2", type=float, default=1e12)
    parser.add_argument(
        "--transport-scheme", choices=("spectral", "groma_upwind"),
        default="spectral")
    parser.add_argument("--collective-wall-nucleation", action="store_true")
    parser.add_argument(
        "--initial-state", choices=(
            "retained_owner", "uniform_single_crystal",
            "continued_single_crystal"),
        default="retained_owner")
    args = parser.parse_args()
    nucleation_seed = (args.source_seed if args.nucleation_seed is None
                       else args.nucleation_seed)
    args.out.mkdir(parents=True, exist_ok=True)
    source_status = subprocess.check_output(
        ("git", "status", "--porcelain"), cwd=ROOT, text=True)
    state, loaded_runtime, step, load, initial_volume, configuration, provenance = (
        _load_checkpoint(args.checkpoint))
    try:
        owner_index = state.grain_ids.index(args.owner_id)
    except ValueError as error:
        raise ValueError("requested material owner is absent") from error
    retained_owner = state.owners[owner_index]
    grid = retained_owner.temperature_K.shape
    if args.initial_state in {"retained_owner", "continued_single_crystal"}:
        owner = retained_owner
        initial_load = load
        active_owner_id = args.owner_id
    else:
        family = grid+(len(retained_owner.mobile_plus_m2[0, 0]),)
        zfamily = np.zeros(family)
        temperature = float(configuration.get("temperature_K", 1000.0))
        owner = CommonWallState(
            np.full(family, 1.0e13), np.full(family, 1.0e13),
            np.full(family, 1.0e13), np.full(family, 1.0e13),
            np.full(family, 1.0e12), np.full(family, 1.0e12),
            np.zeros(grid+(0,)), np.zeros(grid), np.zeros(grid),
            zfamily.copy(), np.zeros(grid+(3, 3)),
            np.zeros(family+(3,)), np.zeros(family+(3, 3)),
            np.zeros(grid), np.full(grid, temperature))
        initial_load = 0.0
        active_owner_id = 1
    single = MultiGrainCommonState(
        (active_owner_id,), np.ones((1,)+grid), (owner,))
    single.validate()
    spacing, thickness, systems, wall, _, _, context = _physics(
        _args_from_configuration(configuration))
    # Gradient-elastic compatibility at the resolved wall width.  The former
    # 2e-5 J/m value corresponds to a sub-grid ~14 nm length and lets the
    # O(1e6 J/m3) ordering well overwhelm Frank--Bilby consistency.
    wall_updates = {
        "wall_nye_match_coefficient_J_m": wall.c44_Pa*spacing**2}
    if args.initial_state in {
            "uniform_single_crystal", "continued_single_crystal"}:
        poisson = wall.c12_Pa/(wall.c11_Pa+wall.c12_Pa)
        reduced_modulus = wall.c44_Pa/(2.0*np.pi*(1.0-poisson))
        gb2 = reduced_modulus*wall.burgers_m**2
        wall_updates.update(
            polarized_wall_absorption_enabled=True,
            polarized_mobile_wall_absorption_enabled=True,
            groma_family_correlation_energy_J_m=1.25*gb2,
            groma_gnd_correlation_energy_J_m=.5*gb2,
            groma_sign_resolved_transport_enabled=True,
            mobile_correlation_diffusivity_m2_s=0.0,
            transport_scheme=args.transport_scheme)
    wall = replace(wall, **wall_updates)
    source_parameters = ArrheniusSourceParameters(
        grain_size_m=float(configuration.get("length_m", spacing*grid[0])),
        pair_increment_m2=1.0e10)
    context = replace(
        context, wall_parameters=wall, shear_rate_s=args.shear_rate,
        maximum_mechanical_subdivisions=(
            14 if args.collective_wall_nucleation
            else context.maximum_mechanical_subdivisions),
        maximum_mechanical_internal_substeps=(
            512 if args.collective_wall_nucleation
            else context.maximum_mechanical_internal_substeps),
        front_enabled=False, source_parameters=source_parameters,
        source_mode=args.source_mode, source_random_seed=args.source_seed,
        source_bulk_site_density_m2=args.source_site_density_m2,
        sink_mode="disabled", interface_sink_mode="disabled",
        structural_absorption_mode="disabled",
        lattice_absorption_mode="disabled",
        wall_orientation_relaxation_mode="ordered_nye",
        wall_orientation_relaxation_parameters=(
            WallOrientationRelaxationParameters()),
        wall_order_relaxation_mode=(
            "implicit_arrhenius" if args.collective_wall_nucleation
            else "disabled"),
        wall_order_relaxation_parameters=(
            ImplicitWallOrderRelaxationParameters()
            if args.collective_wall_nucleation else None),
        compatible_wall_reorganization_mode=(
            "frank_bilby_projection" if args.collective_wall_nucleation
            else "disabled"),
        compatible_wall_reorganization_parameters=(
            CompatibleWallReorganizationParameters(
                conservation_mode="burgers_vector")
            if args.collective_wall_nucleation else None),
        conservative_wall_glide_mode=(
            "chemical_edge_glide" if args.collective_wall_nucleation
            else "disabled"),
        conservative_wall_glide_parameters=(
            ConservativeWallGlideParameters()
            if args.collective_wall_nucleation else None),
        collective_wall_nucleation_mode=(
            "activated_closed_wall" if args.collective_wall_nucleation
            else "disabled"),
        collective_wall_nucleation_parameters=(
            CollectiveWallNucleationParameters(
                source_site_density_m2=args.source_site_density_m2,
                source_seed=args.source_seed)
            if args.collective_wall_nucleation else None),
        collective_wall_nucleation_seed=nucleation_seed)
    runtime = (loaded_runtime
               if args.initial_state == "continued_single_crystal"
               else MultiGrainProductionRuntime(()))
    endpoint = make_endpoint(single, runtime, initial_load, context)
    initial = owner
    records = []
    source_commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()
    progress_configuration = dict(configuration)
    progress_configuration.update(
        schema="v67-single-owner-random-source-v1",
        owner_id=active_owner_id, initial_state=args.initial_state,
        intervals=args.intervals, dt_s=args.dt,
        shear_rate_s=args.shear_rate, source_mode=args.source_mode,
        source_seed=args.source_seed,
        nucleation_seed=nucleation_seed,
        source_bulk_site_density_m2=args.source_site_density_m2,
        collective_wall_nucleation=args.collective_wall_nucleation,
        wall_order_relaxation_mode=context.wall_order_relaxation_mode,
        conservative_wall_glide_mode=context.conservative_wall_glide_mode,
        transport_scheme=args.transport_scheme,
        wall_nye_match_coefficient_J_m=wall.wall_nye_match_coefficient_J_m)
    for index in range(1, args.intervals+1):
        interval = advance_complete_multigrain_interval(
            endpoint, args.dt, context)
        endpoint = interval.endpoint
        current = endpoint.state.owners[0]
        compatibility = evaluate_orientation_wall_compatibility(
            current, systems, np.ones(grid, dtype=bool), spacing)
        records.append({
            "interval": index,
            "physical_time_s": endpoint.runtime.ledger.physical_time_s,
            "applied_shear_strain": endpoint.load,
            "source_classification": interval.audit["source_classification"],
            "source_event_count": interval.audit["source_event_count"],
            "source_relative_first_law_residual": interval.audit[
                "source_relative_first_law_residual"],
            "source_line_balance_residual_m2": interval.audit[
                "source_line_balance_residual_m2"],
            "mechanical_relative_first_law_residual": interval.audit[
                "mechanical_relative_first_law_residual"],
            "wall_order_classification": interval.audit[
                "wall_order_relaxation_classification"],
            "wall_order_fraction": interval.audit[
                "wall_order_relaxation_fractions"].get(active_owner_id, 0.0),
            "wall_order_rate_s": interval.audit[
                "wall_order_relaxation_rates_s"].get(active_owner_id, 0.0),
            "wall_order_target_energy_change_J": interval.audit[
                "wall_order_relaxation_target_energy_changes_J"].get(
                    active_owner_id, 0.0),
            "wall_order_relative_first_law_residual": interval.audit[
                "wall_order_relaxation_relative_first_law_residual"],
            "wall_reorganization_classification": interval.audit[
                "compatible_wall_reorganization_classification"],
            "wall_reorganization_fraction": interval.audit[
                "compatible_wall_reorganization_fractions"].get(
                    active_owner_id, 0.0),
            "wall_reorganization_affinity_J_m": interval.audit[
                "compatible_wall_reorganization_affinities_J_m"].get(
                    active_owner_id, 0.0),
            "wall_reorganization_signed_residual_m2": interval.audit[
                "compatible_wall_reorganization_signed_residual_m2"],
            "wall_reorganization_line_residual_m2": interval.audit[
                "compatible_wall_reorganization_line_residual_m2"],
            "wall_reorganization_relative_first_law_residual": interval.audit[
                "compatible_wall_reorganization_relative_first_law_residual"],
            "conservative_wall_glide_classification": interval.audit[
                "conservative_wall_glide_classification"],
            "conservative_wall_glide_moved_line_cell_sum_m2": interval.audit[
                "conservative_wall_glide_moved_line_cell_sum_m2"],
            "conservative_wall_glide_inventory_residual_m2": interval.audit[
                "conservative_wall_glide_inventory_residual_m2"],
            "conservative_wall_glide_maximum_rate_s": interval.audit[
                "conservative_wall_glide_maximum_rate_s"],
            "conservative_wall_glide_internal_substeps": interval.audit[
                "conservative_wall_glide_internal_substeps"],
            "conservative_wall_glide_relative_first_law_residual": interval.audit[
                "conservative_wall_glide_relative_first_law_residual"],
            "wall_orientation_classification": interval.audit[
                "wall_orientation_relaxation_classification"],
            "wall_orientation_fraction": interval.audit[
                "wall_orientation_relaxation_fractions"].get(
                    active_owner_id, 0.0),
            "wall_orientation_affinity_J_m": interval.audit[
                "wall_orientation_relaxation_affinities_J_m"].get(
                    active_owner_id, 0.0),
            "orientation_span_deg": float(np.degrees(np.ptp(
                current.orientation_rad))),
            "compatible_target_span_deg": float(np.degrees(
                compatibility.reconstructed_orientation_span_rad)),
            "orientation_wall_before_relative_mismatch": float(
                compatibility.before_relative_mismatch),
            "orientation_wall_reconstructed_relative_mismatch": float(
                compatibility.reconstructed_relative_mismatch),
            "wall_order_mean": float(np.mean(current.wall_order)),
            "wall_order_range": [float(np.min(current.wall_order)),
                                 float(np.max(current.wall_order))],
            "collective_wall_classification": (
                interval.audit["collective_wall_nucleation_classification"]),
            "collective_wall_downhill_candidates": (
                interval.audit[
                    "collective_wall_nucleation_downhill_candidates"]),
            "collective_wall_event_probability": (
                interval.audit["collective_wall_nucleation_event_probability"]),
            "collective_wall_selected_radius_m": (
                interval.audit[
                    "collective_wall_nucleation_selected_radius_m"]),
            "collective_wall_delta_helmholtz_J": (
                interval.audit[
                    "collective_wall_nucleation_delta_helmholtz_J"]),
            "collective_wall_first_law_relative_residual": (
                interval.audit[
                    "collective_wall_nucleation_relative_first_law_residual"]),
        })
        print(json.dumps(records[-1]), flush=True)
        # Publish each accepted endpoint atomically.  A disconnect can lose at
        # most the currently executing interval, never the complete prefix.
        progress = args.out/"checkpoint_latest.npz"
        temporary = args.out/"checkpoint_latest.tmp.npz"
        _save_checkpoint(
            temporary, endpoint.state, endpoint.runtime, step+index,
            endpoint.load, initial_volume, progress_configuration,
            {"source_commit": source_commit,
             "parent_provenance": provenance},
            diagnostic_arrays={
                "diagnostic_v67_records_json": np.asarray(json.dumps(records))})
        os.replace(temporary, progress)
    final = endpoint.state.owners[0]
    result_configuration = progress_configuration
    checkpoint = args.out/"checkpoint_latest.npz"
    temporary = args.out/"checkpoint_latest.tmp.npz"
    _save_checkpoint(
        temporary, endpoint.state, endpoint.runtime, step+args.intervals,
        endpoint.load, initial_volume, result_configuration,
        {"source_commit": source_commit,
         "parent_provenance": provenance})
    os.replace(temporary, checkpoint)
    recognitions = _recognitions(final, systems, spacing)
    payload = {
        "schema": "v67-single-owner-random-source-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
        "source_tree_clean": not bool(source_status),
        "parent_checkpoint": str(args.checkpoint.resolve()),
        "parent_checkpoint_sha256": _sha(args.checkpoint),
        "scope": "prepared retained-material one-owner mechanism control",
        "configuration": {
            "owner_id": active_owner_id, "grid": list(grid),
            "initial_state": args.initial_state,
            "spacing_m": spacing, "represented_thickness_m": thickness,
            "intervals": args.intervals, "dt_s": args.dt,
            "added_physical_time_s": args.intervals*args.dt,
            "shear_rate_s": args.shear_rate,
            "added_engineering_shear": (
                args.intervals*args.dt*args.shear_rate),
            "source_mode": args.source_mode, "source_seed": args.source_seed,
            "nucleation_seed": nucleation_seed,
            "source_bulk_site_density_m2": args.source_site_density_m2,
            "collective_wall_nucleation": args.collective_wall_nucleation,
            "wall_order_relaxation_mode": context.wall_order_relaxation_mode,
            "conservative_wall_glide_mode": (
                context.conservative_wall_glide_mode),
            "source_parameters": asdict(source_parameters),
            "wall_nye_match_coefficient_J_m": (
                wall.wall_nye_match_coefficient_J_m),
            "polarized_wall_absorption_enabled": (
                wall.polarized_wall_absorption_enabled),
            "polarized_mobile_wall_absorption_enabled": (
                wall.polarized_mobile_wall_absorption_enabled),
            "groma_sign_resolved_transport_enabled": (
                wall.groma_sign_resolved_transport_enabled),
            "transport_scheme": wall.transport_scheme,
        },
        "initial": {
            "orientation_span_deg": float(np.degrees(np.ptp(
                initial.orientation_rad))),
            "wall_order_mean": float(np.mean(initial.wall_order)),
        },
        "records": records,
        "final_recognition": recognitions,
        "ledger": asdict(endpoint.runtime.ledger),
        "decision": {
            "all_source_events_accepted": all(
                row["source_classification"].startswith("ACCEPTED")
                for row in records),
            "all_hard_balances_passed": all(
                row["source_relative_first_law_residual"] < .05
                and row["mechanical_relative_first_law_residual"] < .05
                and row["wall_order_relative_first_law_residual"] < .05
                and row[
                    "conservative_wall_glide_relative_first_law_residual"] < .05
                and row["wall_reorganization_relative_first_law_residual"] < .05
                and row["collective_wall_first_law_relative_residual"] < .05
                for row in records),
            "accepted_collective_wall_events": sum(
                row["collective_wall_classification"].startswith("ACCEPTED")
                for row in records),
            "strict_plastic_nye_subgrain": bool(
                recognitions["plastic_incompatibility_nye"].get(
                    "qualified", False)),
            "strict_ordered_structural_subgrain": bool(
                recognitions["ordered_structural_nye"].get(
                    "qualified", False)),
            "strict_combined_nye_subgrain": bool(
                recognitions["combined_declared_nye"].get(
                    "qualified", False)),
            "grain_count_change": 0,
        },
        "claim_boundary": (
            "Prepared one-owner topology test; extending one retained owner "
            "over the section is not endogenous nucleation or DRX."),
        "artifacts": {"checkpoint_latest.npz": _sha(checkpoint)},
    }
    (args.out/"result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
