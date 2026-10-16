#!/usr/bin/env python3
"""Retained-state production discriminator for lattice->GB->structure flow."""

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
from full_model.analysis.run_v67_interface_structural_bridge import (
    _args_from_configuration,
)
from full_model.production.coupled_multigrain_interval import (
    advance_complete_multigrain_interval, make_endpoint,
)
from full_model.production.interface_structural_absorption import (
    InterfaceStructuralAbsorptionParameters,
)
from full_model.production.structural_interface_network import (
    JunctionReactionParameters,
)
from full_model.production.lattice_interface_absorption import (
    LatticeInterfaceAbsorptionParameters,
)
from full_model.production.wall_orientation_relaxation import (
    WallOrientationRelaxationParameters,
)
from full_model.production.implicit_wall_order_relaxation import (
    ImplicitWallOrderRelaxationParameters,
)
from full_model.production.conservative_wall_glide import (
    ConservativeWallGlideParameters,
)
from full_model.production.conservative_dipole_wall_organization import (
    ConservativeDipoleWallParameters,
)
from full_model.production.collective_wall_nucleation import (
    CollectiveWallNucleationParameters,
)
from full_model.production.arrhenius_dislocation_sources import (
    ArrheniusSourceParameters,
)
from full_model.production.multigrain_source_transaction import (
    NeutralBoundarySinkParameters,
)


ROOT = Path(__file__).resolve().parents[2]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--intervals", type=int, default=50)
    parser.add_argument("--capture-distance", type=float, default=5e-7)
    parser.add_argument(
        "--lattice-maximum-fraction", type=float, default=.05,
        help=("small lattice/interface affinity-probe extent; only the legacy "
              "clipped-clock control also uses it as a kinetic cap"))
    parser.add_argument(
        "--clipped-lattice-clock", action="store_true",
        help=("negative numerical control that reproduces the legacy loss of "
              "unadvanced Arrhenius physical time"))
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--front-enabled", action="store_true")
    parser.add_argument("--shear-rate", type=float)
    parser.add_argument("--dt", type=float)
    parser.add_argument(
        "--front-substep", type=float,
        help=("maximum front transaction substep in seconds; defaults to the "
              "continuation interval rather than an inherited parent cadence"))
    parser.add_argument("--wall-orientation-relaxation", action="store_true")
    parser.add_argument(
        "--junction-cycle-relaxation", action="store_true",
        help=("advance the conservative Arrhenius structural-cycle reaction "
              "before mechanics and front motion"))
    parser.add_argument(
        "--wall-order-relaxation", action="store_true",
        help=("advance the implicit Arrhenius wall-order variable without "
              "requiring conservative wall-line glide"))
    parser.add_argument("--wall-glide", action="store_true")
    parser.add_argument(
        "--wall-glide-maximum-edge-fraction", type=float, default=.02,
        help=("bounded checkerboard fraction used by conservative wall glide; "
              "changes numerical cadence, not its Arrhenius clock"))
    parser.add_argument("--dipole-wall-organization", action="store_true")
    parser.add_argument(
        "--collective-wall-nucleation", action="store_true",
        help=("enable the separately labeled finite-amplitude closed-wall "
              "ablation with a complete-energy reaction-path audit"))
    parser.add_argument("--collective-wall-order-increment", type=float,
                        default=.01)
    parser.add_argument("--collective-wall-source-site-density", type=float,
                        default=1.0e12)
    parser.add_argument("--collective-wall-seed", type=int, default=67)
    parser.add_argument(
        "--wall-nye-match-length", type=float,
        help="physical wall/orientation match length in metres")
    parser.add_argument("--wall-order-support-threshold", type=float,
                        default=1.0e-8)
    parser.add_argument("--wall-order-max-iterations", type=int, default=300)
    parser.add_argument(
        "--source-mode", choices=("disabled", "gb_tj_only", "matched_random"),
        default="disabled")
    parser.add_argument("--source-seed", type=int, default=67)
    parser.add_argument(
        "--interface-neutral-sink", action="store_true",
        help=("enable the separately ledgered thermal annihilation of neutral "
              "positive/negative line pairs already stored at interfaces"))
    parser.add_argument(
        "--interface-vector-junction-sink", action="store_true",
        help=("ablation: react only scalar interface line above the exact "
              "Burgers-capacity lower bound while preserving the Burgers "
              "tensor bitwise"))
    args = parser.parse_args()
    if args.interface_neutral_sink and args.interface_vector_junction_sink:
        raise ValueError("select at most one interface sink hypothesis")
    source_status = subprocess.check_output(
        ("git", "status", "--porcelain", "--untracked-files=no"),
        cwd=ROOT, text=True)
    args.out.mkdir(parents=True, exist_ok=True)
    state, runtime, step, load, initial_volume, configuration, provenance = (
        _load_checkpoint(args.checkpoint))
    physics_args = _args_from_configuration(configuration)
    spacing, _, _, _, _, _, context = _physics(physics_args)
    imposed_rate = (float(args.shear_rate) if args.shear_rate is not None
                    else (float(configuration.get("shear_rate_s", 0.0))
                          if args.dynamic else 0.0))
    fronts_enabled = bool(args.dynamic or args.front_enabled
                          or args.shear_rate is not None)
    wall_parameters = context.wall_parameters
    if args.wall_orientation_relaxation or args.wall_glide:
        match_length = (spacing if args.wall_nye_match_length is None
                        else float(args.wall_nye_match_length))
        if not np.isfinite(match_length) or match_length <= 0.0:
            raise ValueError("wall Nye match length must be positive finite")
        wall_parameters = replace(
            wall_parameters,
            wall_nye_match_coefficient_J_m=(
                wall_parameters.c44_Pa*match_length**2))
    else:
        match_length = None
    context = replace(
        context, wall_parameters=wall_parameters,
        shear_rate_s=imposed_rate, front_enabled=fronts_enabled,
        source_mode=args.source_mode,
        source_parameters=(
            None if args.source_mode == "disabled"
            else ArrheniusSourceParameters(grain_size_m=2.4e-6)),
        source_random_seed=args.source_seed, sink_mode="disabled",
        interface_sink_mode=(
            "vector_neutral_junction"
            if args.interface_vector_junction_sink else
            ("neutral_pair" if args.interface_neutral_sink else "disabled")),
        interface_sink_parameters=(
            NeutralBoundarySinkParameters(complete_physical_clock=True)
            if (args.interface_neutral_sink
                or args.interface_vector_junction_sink) else None),
        structural_absorption_mode="common_axis",
        structural_absorption_parameters=InterfaceStructuralAbsorptionParameters(
            complete_physical_clock=True),
        structural_junction_mode=(
            "arrhenius_cycle" if args.junction_cycle_relaxation else "disabled"),
        structural_junction_parameters=(
            JunctionReactionParameters(complete_physical_clock=True)
            if args.junction_cycle_relaxation else None),
        lattice_absorption_mode="one_flux_mura",
        lattice_absorption_parameters=LatticeInterfaceAbsorptionParameters(
            maximum_fraction_per_interval=args.lattice_maximum_fraction,
            capture_distance_m=args.capture_distance,
            complete_physical_clock=not args.clipped_lattice_clock),
        wall_orientation_relaxation_mode=(
            "ordered_nye" if (args.wall_orientation_relaxation
                               or args.wall_glide) else "disabled"),
        wall_orientation_relaxation_parameters=(
            WallOrientationRelaxationParameters()
            if (args.wall_orientation_relaxation or args.wall_glide) else None),
        wall_order_relaxation_mode=(
            "implicit_arrhenius" if (
                args.wall_order_relaxation or args.wall_glide) else "disabled"),
        wall_order_relaxation_parameters=(
            ImplicitWallOrderRelaxationParameters(
                owner_support_threshold=args.wall_order_support_threshold,
                optimizer_max_iterations=args.wall_order_max_iterations)
            if (args.wall_order_relaxation or args.wall_glide) else None),
        conservative_wall_glide_mode=(
            "chemical_edge_glide" if args.wall_glide else "disabled"),
        conservative_wall_glide_parameters=(
            ConservativeWallGlideParameters(
                maximum_edge_fraction=args.wall_glide_maximum_edge_fraction)
            if args.wall_glide else None),
        conservative_dipole_wall_mode=(
            "neutral_pair_expansion" if args.dipole_wall_organization
            else "disabled"),
        conservative_dipole_wall_parameters=(
            ConservativeDipoleWallParameters()
            if args.dipole_wall_organization else None),
        collective_wall_nucleation_mode=(
            "activated_closed_wall" if args.collective_wall_nucleation
            else "disabled"),
        collective_wall_nucleation_parameters=(
            CollectiveWallNucleationParameters(
                source_site_density_m2=(
                    args.collective_wall_source_site_density),
                order_increment=args.collective_wall_order_increment)
            if args.collective_wall_nucleation else None),
        collective_wall_nucleation_seed=args.collective_wall_seed)
    dt = (float(args.dt) if args.dt is not None
          else float(configuration["dt_s"]))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("continuation dt must be positive finite")
    front_substep = (dt if args.front_substep is None
                     else float(args.front_substep))
    if not np.isfinite(front_substep) or front_substep <= 0.0:
        raise ValueError("front substep must be positive finite")
    if front_substep > dt:
        raise ValueError("front substep cannot exceed the continuation interval")
    context = replace(context, maximum_front_substep_s=front_substep)
    endpoint = make_endpoint(state, runtime, load, context)
    before = _directional_audit(endpoint, context, fraction=2e-4)
    records = []
    progress_configuration = dict(configuration)
    progress_configuration.update(
        schema="v67-one-flux-lattice-structural-continuation-v1",
        parent_checkpoint=str(args.checkpoint.resolve()),
        target_steps=args.intervals, shear_rate_s=imposed_rate, dt_s=dt,
        front_maximum_substep_s=front_substep,
        front_enabled=fronts_enabled,
        lattice_absorption_mode="one_flux_mura",
        lattice_capture_distance_m=args.capture_distance,
        lattice_maximum_fraction=args.lattice_maximum_fraction,
        lattice_complete_physical_clock=not args.clipped_lattice_clock,
        structural_absorption_mode="common_axis",
        structural_junction_mode=context.structural_junction_mode,
        wall_orientation_relaxation_mode=(
            "ordered_nye" if (args.wall_orientation_relaxation
                              or args.wall_glide) else "disabled"),
        wall_nye_match_coefficient_J_m=(
            wall_parameters.wall_nye_match_coefficient_J_m),
        wall_nye_match_length_m=match_length,
        wall_order_support_threshold=args.wall_order_support_threshold,
        wall_order_max_iterations=args.wall_order_max_iterations,
        wall_order_relaxation_mode=context.wall_order_relaxation_mode,
        conservative_wall_glide_mode=context.conservative_wall_glide_mode,
        wall_glide_maximum_edge_fraction=(
            args.wall_glide_maximum_edge_fraction),
        conservative_dipole_wall_mode=context.conservative_dipole_wall_mode,
        collective_wall_nucleation_mode=(
            context.collective_wall_nucleation_mode),
        collective_wall_order_increment=(
            args.collective_wall_order_increment),
        collective_wall_source_site_density_m2=(
            args.collective_wall_source_site_density),
        collective_wall_seed=args.collective_wall_seed,
        source_mode=args.source_mode, source_seed=args.source_seed)
    progress_configuration["interface_neutral_sink"] = bool(
        args.interface_neutral_sink)
    progress_configuration["interface_vector_junction_sink"] = bool(
        args.interface_vector_junction_sink)
    source_commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()
    for index in range(1, args.intervals+1):
        interval = advance_complete_multigrain_interval(endpoint, dt, context)
        endpoint = interval.endpoint
        records.append({
            "interval": index,
            "interface_sink_classification": interval.audit[
                "interface_sink_classification"],
            "interface_sink_annihilated_pair_cell_sum_m2": interval.audit[
                "interface_sink_annihilated_pair_cell_sum_m2"],
            "interface_sink_first_law_relative_residual": interval.audit[
                "interface_sink_relative_first_law_residual"],
            "lattice_classification": interval.audit[
                "lattice_absorption_classification"],
            "lattice_fraction": interval.audit["lattice_absorption_fraction"],
            "lattice_event_rate_s": interval.audit[
                "lattice_absorption_event_rate_s"],
            "lattice_maximum_uncapped_fraction": interval.audit[
                "lattice_absorption_maximum_uncapped_fraction"],
            "lattice_selected_hazard_scale": interval.audit[
                "lattice_absorption_selected_hazard_scale"],
            "lattice_energy_search_evaluations": interval.audit[
                "lattice_absorption_energy_search_evaluations"],
            "lattice_kinetics_mode": interval.audit[
                "lattice_absorption_kinetics_mode"],
            "lattice_edge_fractions": interval.audit[
                "lattice_absorption_edge_fractions"],
            "lattice_edge_affinities_J_m": interval.audit[
                "lattice_absorption_edge_affinities_J_m"],
            "lattice_removed_line_m": interval.audit[
                "lattice_absorption_removed_line_m"],
            "lattice_mura_relative_residual": interval.audit[
                "lattice_absorption_mura_relative_residual"],
            "lattice_first_law_relative_residual": interval.audit[
                "lattice_absorption_relative_first_law_residual"],
            "structural_classification": interval.audit[
                "structural_absorption_classification"],
            "structural_content_change": interval.audit[
                "structural_absorption_content_change"],
            "structural_post_classification": interval.audit[
                "structural_absorption_post_classification"],
            "structural_post_fraction": interval.audit[
                "structural_absorption_post_fraction"],
            "structural_post_content_change": interval.audit[
                "structural_absorption_post_content_change"],
            "structural_post_first_law_relative_residual": interval.audit[
                "structural_absorption_post_relative_first_law_residual"],
            "structural_junction_classification": interval.audit[
                "structural_junction_classification"],
            "structural_junction_reacted_fraction": interval.audit[
                "structural_junction_reacted_fraction"],
            "structural_junction_backstress_Pa": interval.audit[
                "structural_junction_backstress_Pa"],
            "structural_junction_absolute_reacted_content": interval.audit[
                "structural_junction_absolute_reacted_content"],
            "structural_junction_relative_first_law_residual": interval.audit[
                "structural_junction_relative_first_law_residual"],
            "structural_junction_post_classification": interval.audit[
                "structural_junction_post_classification"],
            "structural_junction_post_reacted_fraction": interval.audit[
                "structural_junction_post_reacted_fraction"],
            "structural_junction_post_absolute_reacted_content": interval.audit[
                "structural_junction_post_absolute_reacted_content"],
            "structural_junction_post_relative_first_law_residual": interval.audit[
                "structural_junction_post_relative_first_law_residual"],
            "wall_orientation_classification": interval.audit[
                "wall_orientation_relaxation_classification"],
            "wall_orientation_fractions": interval.audit[
                "wall_orientation_relaxation_fractions"],
            "wall_orientation_affinities_J_m": interval.audit[
                "wall_orientation_relaxation_affinities_J_m"],
            "wall_orientation_first_law_relative_residual": interval.audit[
                "wall_orientation_relaxation_relative_first_law_residual"],
            "wall_glide_classification": interval.audit[
                "conservative_wall_glide_classification"],
            "wall_glide_moved_line_cell_sum_m2": interval.audit[
                "conservative_wall_glide_moved_line_cell_sum_m2"],
            "wall_glide_inventory_residual_m2": interval.audit[
                "conservative_wall_glide_inventory_residual_m2"],
            "wall_glide_internal_substeps": interval.audit[
                "conservative_wall_glide_internal_substeps"],
            "wall_glide_backtracks": interval.audit[
                "conservative_wall_glide_backtracks"],
            "wall_glide_maximum_rate_s": interval.audit[
                "conservative_wall_glide_maximum_rate_s"],
            "wall_glide_delta_helmholtz_J": interval.audit[
                "conservative_wall_glide_delta_helmholtz_J"],
            "wall_glide_first_law_relative_residual": interval.audit[
                "conservative_wall_glide_relative_first_law_residual"],
            "dipole_wall_classification": interval.audit[
                "conservative_dipole_wall_classification"],
            "dipole_wall_separated_pair_cell_sum_m2": interval.audit[
                "conservative_dipole_wall_separated_pair_cell_sum_m2"],
            "dipole_wall_line_residual_m2": interval.audit[
                "conservative_dipole_wall_line_residual_m2"],
            "dipole_wall_backtracks": interval.audit[
                "conservative_dipole_wall_backtracks"],
            "dipole_wall_maximum_rate_s": interval.audit[
                "conservative_dipole_wall_maximum_rate_s"],
            "dipole_wall_delta_helmholtz_J": interval.audit[
                "conservative_dipole_wall_delta_helmholtz_J"],
            "dipole_wall_first_law_relative_residual": interval.audit[
                "conservative_dipole_wall_relative_first_law_residual"],
            "source_classification": interval.audit["source_classification"],
            "source_event_count": interval.audit["source_event_count"],
            "source_first_law_relative_residual": interval.audit[
                "source_relative_first_law_residual"],
            "collective_wall_classification": interval.audit[
                "collective_wall_nucleation_classification"],
            "collective_wall_downhill_candidates": interval.audit[
                "collective_wall_nucleation_downhill_candidates"],
            "collective_wall_event_probability": interval.audit[
                "collective_wall_nucleation_event_probability"],
            "collective_wall_delta_helmholtz_J": interval.audit[
                "collective_wall_nucleation_delta_helmholtz_J"],
            "collective_wall_resolved_forward_barrier_J": interval.audit[
                "collective_wall_nucleation_resolved_forward_barrier_J"],
            "collective_wall_forward_barrier_eV": interval.audit[
                "collective_wall_nucleation_forward_barrier_eV"],
            "collective_wall_reverse_barrier_eV": interval.audit[
                "collective_wall_nucleation_reverse_barrier_eV"],
            "collective_wall_total_forward_event_rate_Hz": interval.audit[
                "collective_wall_nucleation_total_forward_event_rate_Hz"],
            "collective_wall_selected_channel_probability": interval.audit[
                "collective_wall_nucleation_selected_channel_probability"],
            "collective_wall_detailed_balance_log_residual": interval.audit[
                "collective_wall_nucleation_detailed_balance_log_residual"],
            "owner_orientation_span_deg": {
                str(grain): float(np.degrees(np.ptp(
                    owner.orientation_rad[np.asarray(support) > .75])))
                for grain, support, owner in zip(
                    endpoint.state.grain_ids, endpoint.state.supports,
                    endpoint.state.owners)},
            "owner_support_weighted_wall_order": {
                str(grain): float(np.sum(support*owner.wall_order)
                                  /np.sum(support))
                for grain, support, owner in zip(
                    endpoint.state.grain_ids, endpoint.state.supports,
                    endpoint.state.owners)},
        })
        print(json.dumps(records[-1], sort_keys=True), flush=True)
        temporary = args.out/"checkpoint_latest.tmp.npz"
        _save_checkpoint(
            temporary, endpoint.state, endpoint.runtime, step+index,
            endpoint.load, initial_volume, progress_configuration,
            {"source_commit": source_commit,
             "parent_provenance": provenance},
            diagnostic_arrays={
                "diagnostic_v67_records_json": np.asarray(json.dumps(records))})
        os.replace(temporary, args.out/"checkpoint_latest.npz")
    after = _directional_audit(endpoint, context, fraction=2e-4)
    commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()
    accepted_lattice = sum(row["lattice_classification"].startswith("ACCEPTED")
                           for row in records)
    accepted_structural = sum(
        row["structural_classification"].startswith("ACCEPTED")
        for row in records)
    continuation_configuration = progress_configuration
    checkpoint = args.out/"checkpoint_latest.npz"
    temporary = args.out/"checkpoint_latest.tmp.npz"
    _save_checkpoint(
        temporary, endpoint.state, endpoint.runtime, step+args.intervals,
        endpoint.load, initial_volume, continuation_configuration,
        {"source_commit": commit, "parent_provenance": provenance})
    os.replace(temporary, checkpoint)
    initial_owner_areas = np.sum(state.supports, axis=(1, 2))*spacing**2
    final_owner_areas = np.sum(endpoint.state.supports, axis=(1, 2))*spacing**2
    payload = {
        "schema": "v67-one-flux-lattice-structural-continuation-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": commit,
        "source_tree_clean": not bool(source_status),
        "parent_checkpoint": str(args.checkpoint.resolve()),
        "parent_checkpoint_sha256": _sha(args.checkpoint),
        "parent_provenance": provenance,
        "configuration": {
            "intervals": args.intervals, "dt_s": dt,
            "front_maximum_substep_s": front_substep,
            "capture_distance_m": args.capture_distance,
            "lattice_maximum_fraction": args.lattice_maximum_fraction,
            "lattice_complete_physical_clock": not args.clipped_lattice_clock,
            "shear_rate_s": imposed_rate, "front_enabled": fronts_enabled,
            "lattice_absorption": "one_flux_mura",
            "structural_absorption": "common_axis",
            "wall_orientation_relaxation": bool(
                args.wall_orientation_relaxation or args.wall_glide),
            "wall_order_relaxation": bool(
                args.wall_order_relaxation or args.wall_glide),
            "wall_glide": bool(args.wall_glide),
            "wall_glide_maximum_edge_fraction": (
                args.wall_glide_maximum_edge_fraction),
            "dipole_wall_organization": bool(args.dipole_wall_organization),
            "collective_wall_nucleation": bool(
                args.collective_wall_nucleation),
            "collective_wall_order_increment": (
                args.collective_wall_order_increment),
            "collective_wall_source_site_density_m2": (
                args.collective_wall_source_site_density),
            "collective_wall_seed": args.collective_wall_seed,
            "source_mode": args.source_mode,
            "source_seed": args.source_seed,
            "interface_neutral_sink": bool(args.interface_neutral_sink),
            "interface_vector_junction_sink": bool(
                args.interface_vector_junction_sink),
            "wall_nye_match_coefficient_J_m": (
                wall_parameters.wall_nye_match_coefficient_J_m),
            "wall_nye_match_length_m": match_length,
            "wall_order_support_threshold": args.wall_order_support_threshold,
            "wall_order_max_iterations": args.wall_order_max_iterations,
        },
        "before_directional_audit": before,
        "after_directional_audit": after,
        "interval_records": records,
        "ledger": asdict(endpoint.runtime.ledger),
        "owner_area_m2": {
            str(grain): {"before": float(initial_owner_areas[index]),
                         "after": float(final_owner_areas[index]),
                         "change": float(final_owner_areas[index]
                                         -initial_owner_areas[index])}
            for index, grain in enumerate(state.grain_ids)},
        "decision": {
            "accepted_lattice_intervals": accepted_lattice,
            "accepted_structural_intervals": accepted_structural,
            "accepted_wall_orientation_intervals": sum(
                row["wall_orientation_classification"].startswith("ACCEPTED")
                for row in records),
            "accepted_wall_glide_intervals": sum(
                row["wall_glide_classification"].startswith("ACCEPTED")
                for row in records),
            "accepted_dipole_wall_intervals": sum(
                row["dipole_wall_classification"].startswith("ACCEPTED")
                for row in records),
            "accepted_source_intervals": sum(
                row["source_classification"].startswith("ACCEPTED")
                for row in records),
            "accepted_interface_sink_intervals": sum(
                row["interface_sink_classification"].startswith("ACCEPTED")
                for row in records),
            "mura_hard_balance_passed": bool(all(
                row["lattice_mura_relative_residual"] < 2e-11
                for row in records)),
            "first_law_hard_balance_passed": bool(all(
                row["lattice_first_law_relative_residual"] < .05
                and row["wall_glide_first_law_relative_residual"] < .05
                and row["dipole_wall_first_law_relative_residual"] < .05
                and row["source_first_law_relative_residual"] < .05
                for row in records
                if row["lattice_classification"].startswith("ACCEPTED"))),
            "supports_unchanged": bool(np.array_equal(
                endpoint.state.supports, state.supports)),
            "grain_ids_unchanged": endpoint.state.grain_ids == state.grain_ids,
            "maximum_support_change": float(np.max(np.abs(
                endpoint.state.supports-state.supports))),
        },
        "claim_boundary": (
            "Retained-state stationary production discriminator. It qualifies "
            "the coupled absorption path but not sustained dynamic DRX."),
        "artifacts": {"checkpoint_latest.npz": _sha(checkpoint)},
    }
    (args.out/"result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
