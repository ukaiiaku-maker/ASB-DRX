#!/usr/bin/env python3
"""Feedback-controlled release toward zero mean shear stress.

The production map is strain controlled.  This driver therefore changes the
applied strain through ordinary complete, work-ledgered intervals.  A local
elastic secant predicts the strain needed for zero mean stress and a bounded
increment approaches it.  It is a near-zero-mean-stress control, not an
instantaneous stress-boundary-condition solver.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np

from full_model.analysis.run_v58_three_grain_production import (
    _load_checkpoint, _save_checkpoint,
)
from full_model.analysis.run_v66_coupled_rx_asb import (
    _atomic_json, _physics, _record, _source_provenance,
)
from full_model.analysis.run_v67_interface_structural_bridge import (
    _args_from_configuration,
)
from full_model.production.arrhenius_dislocation_sources import (
    ArrheniusSourceParameters,
)
from full_model.production.collective_wall_nucleation import (
    CollectiveWallNucleationParameters,
)
from full_model.production.compatible_wall_reorganization import (
    CompatibleWallReorganizationParameters,
)
from full_model.production.conservative_wall_glide import (
    ConservativeWallGlideParameters,
)
from full_model.production.conservative_dipole_wall_organization import (
    ConservativeDipoleWallParameters,
)
from full_model.production.coupled_multigrain_interval import (
    advance_complete_multigrain_interval, make_endpoint,
)
from full_model.production.implicit_wall_order_relaxation import (
    ImplicitWallOrderRelaxationParameters,
)
from full_model.production.interface_structural_absorption import (
    InterfaceStructuralAbsorptionParameters,
)
from full_model.production.lattice_interface_absorption import (
    LatticeInterfaceAbsorptionParameters,
)
from full_model.production.orientation_wall_compatibility import (
    evaluate_orientation_wall_compatibility,
)
from full_model.production.wall_orientation_relaxation import (
    WallOrientationRelaxationParameters,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/"full_model"/"verification"/"v66_unload_static"/"checkpoint_latest.npz"
DEFAULT_OUT = ROOT/"full_model"/"verification"/"v66_zero_mean_stress"


def _mean_shear(endpoint):
    return float(np.mean(
        np.asarray(endpoint.observables["common_stress_tensor_Pa"])[..., 0, 1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--dt", type=float, default=2e-9)
    parser.add_argument("--probe-strain", type=float, default=1e-4)
    parser.add_argument("--maximum-strain-change", type=float, default=5e-4)
    parser.add_argument("--minimum-rx-growth-fraction", type=float, default=.01)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--v67-wall-physics", action="store_true")
    parser.add_argument("--v67-polycrystal-physics", action="store_true")
    parser.add_argument("--source-seed", type=int, default=67)
    parser.add_argument("--nucleation-seed", type=int, default=24)
    args = parser.parse_args()
    args.source = args.source.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    state, runtime, _, load, _, source_configuration, _ = _load_checkpoint(
        args.source)
    physical_args = _args_from_configuration(source_configuration)
    physical_args.shear_rate = 0.0
    physical_args.dt = args.dt
    spacing, _, systems, wall, _, _, base_context = _physics(physical_args)
    if args.v67_wall_physics and args.v67_polycrystal_physics:
        parser.error("select only one V67 physics continuation")
    if args.v67_wall_physics:
        poisson = wall.c12_Pa/(wall.c11_Pa+wall.c12_Pa)
        reduced_modulus = wall.c44_Pa/(2.0*np.pi*(1.0-poisson))
        gb2 = reduced_modulus*wall.burgers_m**2
        wall = replace(
            wall,
            wall_nye_match_coefficient_J_m=wall.c44_Pa*spacing**2,
            polarized_wall_absorption_enabled=True,
            polarized_mobile_wall_absorption_enabled=True,
            groma_family_correlation_energy_J_m=1.25*gb2,
            groma_gnd_correlation_energy_J_m=.5*gb2,
            groma_sign_resolved_transport_enabled=True,
            mobile_correlation_diffusivity_m2_s=0.0,
            transport_scheme="groma_upwind")
        base_context = replace(
            base_context, wall_parameters=wall, front_enabled=False,
            maximum_mechanical_subdivisions=14,
            maximum_mechanical_internal_substeps=512,
            source_parameters=ArrheniusSourceParameters(
                grain_size_m=physical_args.length, pair_increment_m2=1.0e10),
            source_mode="random_bulk", source_random_seed=args.source_seed,
            source_bulk_site_density_m2=1.0e12,
            sink_mode="disabled", interface_sink_mode="disabled",
            structural_absorption_mode="disabled",
            lattice_absorption_mode="disabled",
            wall_orientation_relaxation_mode="ordered_nye",
            wall_orientation_relaxation_parameters=(
                WallOrientationRelaxationParameters()),
            wall_order_relaxation_mode="implicit_arrhenius",
            wall_order_relaxation_parameters=(
                ImplicitWallOrderRelaxationParameters()),
            compatible_wall_reorganization_mode="frank_bilby_projection",
            compatible_wall_reorganization_parameters=(
                CompatibleWallReorganizationParameters(
                    conservation_mode="burgers_vector")),
            conservative_wall_glide_mode="chemical_edge_glide",
            conservative_wall_glide_parameters=ConservativeWallGlideParameters(),
            collective_wall_nucleation_mode="activated_closed_wall",
            collective_wall_nucleation_parameters=(
                CollectiveWallNucleationParameters(
                    source_site_density_m2=1.0e12,
                    source_seed=args.source_seed)),
            collective_wall_nucleation_seed=args.nucleation_seed)
    if args.v67_polycrystal_physics:
        wall = replace(
            wall,
            wall_nye_match_coefficient_J_m=wall.c44_Pa*spacing**2)
        base_context = replace(
            base_context, wall_parameters=wall, front_enabled=True,
            maximum_mechanical_subdivisions=14,
            maximum_mechanical_internal_substeps=512,
            source_parameters=ArrheniusSourceParameters(grain_size_m=2.4e-6),
            source_mode="gb_tj_only", source_random_seed=args.source_seed,
            sink_mode="disabled", interface_sink_mode="disabled",
            structural_absorption_mode="common_axis",
            structural_absorption_parameters=(
                InterfaceStructuralAbsorptionParameters()),
            lattice_absorption_mode="one_flux_mura",
            lattice_absorption_parameters=LatticeInterfaceAbsorptionParameters(
                capture_distance_m=5e-7),
            wall_orientation_relaxation_mode="ordered_nye",
            wall_orientation_relaxation_parameters=(
                WallOrientationRelaxationParameters()),
            wall_order_relaxation_mode="implicit_arrhenius",
            wall_order_relaxation_parameters=(
                ImplicitWallOrderRelaxationParameters(
                    owner_support_threshold=float(source_configuration.get(
                        "wall_order_support_threshold", 1.0e-8)),
                    optimizer_max_iterations=int(source_configuration.get(
                        "wall_order_max_iterations", 300)))),
            conservative_wall_glide_mode="chemical_edge_glide",
            conservative_wall_glide_parameters=ConservativeWallGlideParameters(),
            conservative_dipole_wall_mode="neutral_pair_expansion",
            conservative_dipole_wall_parameters=(
                ConservativeDipoleWallParameters()))
    endpoint = make_endpoint(state, runtime, load, base_context)
    initial_volume = (np.sum(state.supports, axis=(1, 2))
                      *base_context.spacing_m**2
                      *base_context.represented_thickness_m)
    configuration = dict(source_configuration)
    configuration.update({
        "schema": "v66-zero-mean-stress-feedback-v1",
        "source_checkpoint": str(args.source.relative_to(ROOT)),
        "steps": args.steps, "dt_s": args.dt,
        "probe_strain": args.probe_strain,
        "maximum_strain_change_per_interval": args.maximum_strain_change,
        "minimum_rx_growth_fraction": args.minimum_rx_growth_fraction,
        "controller": "bounded local elastic secant; complete work-ledgered interval",
        "v67_wall_physics": args.v67_wall_physics,
        "v67_polycrystal_physics": args.v67_polycrystal_physics,
        "source_seed": args.source_seed,
        "nucleation_seed": args.nucleation_seed,
    })
    provenance = dict(
        _source_provenance(),
        scientific_scope="near-zero-mean-stress discriminator")
    history = []
    for step in range(1, args.steps+1):
        stress0 = _mean_shear(endpoint)
        probe = make_endpoint(
            endpoint.state, endpoint.runtime, endpoint.load+args.probe_strain,
            base_context, numerical_hints=endpoint.numerical_hints)
        tangent = (_mean_shear(probe)-stress0)/args.probe_strain
        if not np.isfinite(tangent) or tangent <= 0.0:
            raise RuntimeError("mean shear-stress tangent is not positive finite")
        predicted = endpoint.load-stress0/tangent
        increment = float(np.clip(
            predicted-endpoint.load, -args.maximum_strain_change,
            args.maximum_strain_change))
        context = replace(base_context, shear_rate_s=increment/args.dt)
        started = time.perf_counter()
        interval = advance_complete_multigrain_interval(endpoint, args.dt, context)
        endpoint = interval.endpoint
        row = _record(
            step, endpoint, interval, initial_volume, context,
            time.perf_counter()-started)
        row.update({
            "controller_stress_before_Pa": stress0,
            "controller_elastic_tangent_Pa": tangent,
            "controller_predicted_zero_stress_strain": predicted,
            "controller_strain_increment": increment,
        })
        if args.v67_wall_physics and len(endpoint.state.owners) == 1:
            owner = endpoint.state.owners[0]
            compatibility = evaluate_orientation_wall_compatibility(
                owner, systems, np.ones(owner.orientation_rad.shape, dtype=bool),
                spacing)
            row.update({
                "orientation_span_deg": float(np.degrees(np.ptp(
                    owner.orientation_rad))),
                "compatible_target_span_deg": float(np.degrees(
                    compatibility.reconstructed_orientation_span_rad)),
                "orientation_wall_relative_mismatch": float(
                    compatibility.reconstructed_relative_mismatch),
                "wall_order_mean": float(np.mean(owner.wall_order)),
                "conservative_wall_glide_classification": interval.audit[
                    "conservative_wall_glide_classification"],
                "conservative_wall_glide_relative_first_law_residual": (
                    interval.audit[
                        "conservative_wall_glide_relative_first_law_residual"]),
            })
        if args.v67_polycrystal_physics:
            row.update({
                "owner_orientation_span_deg": {
                    str(grain): float(np.degrees(np.ptp(
                        owner.orientation_rad[np.asarray(support) > .75])))
                    for grain, support, owner in zip(
                        endpoint.state.grain_ids, endpoint.state.supports,
                        endpoint.state.owners)},
                "lattice_absorption_classification": interval.audit[
                    "lattice_absorption_classification"],
                "lattice_mura_relative_residual": interval.audit[
                    "lattice_absorption_mura_relative_residual"],
                "structural_absorption_classification": interval.audit[
                    "structural_absorption_classification"],
                "dipole_wall_classification": interval.audit[
                    "conservative_dipole_wall_classification"],
                "wall_glide_classification": interval.audit[
                    "conservative_wall_glide_classification"],
                "wall_orientation_classification": interval.audit[
                    "wall_orientation_relaxation_classification"],
            })
        history.append(row)
        if step == 1 or step % args.checkpoint_every == 0 or step == args.steps:
            temporary = args.out/"checkpoint_latest.tmp.npz"
            _save_checkpoint(
                temporary, endpoint.state, endpoint.runtime, step,
                endpoint.load, initial_volume, configuration, provenance)
            os.replace(temporary, args.out/"checkpoint_latest.npz")
            _atomic_json(args.out/"history.json", history)
            stress = np.asarray([item["mean_shear_stress_Pa"] for item in history])
            target = np.asarray([item["target_growth_fraction"] for item in history])
            settled = np.abs(stress) <= 10e6
            first_settled = int(np.argmax(settled)) if np.any(settled) else None
            payload = {
                "schema": "v66-zero-mean-stress-result-v1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "configuration": configuration,
                "result": {
                    "steps_completed": step,
                    "final_applied_shear_strain": endpoint.load,
                    "final_mean_shear_stress_Pa": float(stress[-1]),
                    "minimum_absolute_mean_shear_stress_Pa": float(
                        np.min(np.abs(stress))),
                    "first_step_below_10MPa": (
                        None if first_settled is None else first_settled+1),
                    "final_target_change_fraction": float(target[-1]),
                    "target_change_after_first_below_10MPa": (
                        None if first_settled is None else
                        float(target[-1]-target[first_settled])),
                    "cumulative_external_work_J": float(sum(
                        item["mechanical_external_work_J"] for item in history)),
                    "maximum_mechanical_relative_first_law_residual": (
                        endpoint.runtime.ledger.maximum_mechanical_relative_first_law_residual),
                    "maximum_front_relative_energy_closure": (
                        endpoint.runtime.ledger.maximum_relative_energy_closure),
                    "rejected_front_events_total": endpoint.runtime.ledger.rejected_events,
                    "final_owner_orientation_span_deg": (
                        history[-1].get("owner_orientation_span_deg")),
                },
                "decision": {
                    "run_complete": step == args.steps,
                    "near_zero_mean_stress_reached": first_settled is not None,
                    "stress_free_claim": False,
                    "static_rx_at_near_zero_mean_stress_supported": bool(
                        first_settled is not None
                        and target[-1]-target[first_settled]
                        >= args.minimum_rx_growth_fraction),
                    "minimum_rx_growth_fraction": args.minimum_rx_growth_fraction,
                    "reason": (
                        "Mean stress is feedback-controlled by work-ledgered strain "
                        "increments; local residual stresses and finite controller error remain."),
                },
            }
            _atomic_json(args.out/"result.json", payload)
            print(json.dumps({
                "step": step, "target": args.steps,
                "load": endpoint.load, "mean_stress_MPa": stress[-1]/1e6,
                "target_change_fraction": target[-1],
            }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
