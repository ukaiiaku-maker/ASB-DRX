#!/usr/bin/env python3
"""Run the recurrent V58 three-grain common-owner production trajectory."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from full_model.production.arrhenius_kinetics import ActivatedProcess, EV_J
from full_model.production.common_tensorial_wall import (
    CommonWallParameters, CommonWallState,
)
from full_model.production.complete_multigrain_energy import (
    _mean_mechanical_stress, evaluate_complete_multigrain_energy,
)
from full_model.production.multigrain_common_state import (
    MultiGrainCommonState, PhysicalTransferLaw, audit_multigrain_nye,
    multigrain_checkpoint_arrays, multigrain_checkpoint_metadata,
    multigrain_from_checkpoint,
)
from full_model.production.multigrain_production import (
    MultiGrainFrontKinetics, MultiGrainInterface,
    MultiGrainProductionLedger, MultiGrainProductionRuntime,
    advance_energy_qualified_mechanics, advance_multigrain_front_interval,
)
from full_model.production.tensorial_nye import bcc_four_family_systems
from full_model.analysis.spatial_localization import spatial_localization_metrics


def _owner(n, density, orientation, temperature):
    grid = (n, n); family = grid+(4,)
    # The argument is total line density. Reservoir/family weights below sum
    # to unity, matching the retained V55 physical density scale.
    value = np.full(family, density/(4.0*(2*.08+2*.18+2*.12)))
    return CommonWallState(
        .08*value, .08*value, .18*value, .18*value,
        .12*value, .12*value, np.zeros(grid+(0,)),
        np.full(grid, .25), np.zeros(grid), np.zeros(family),
        np.zeros(grid+(3, 3)), np.zeros(family+(3,)),
        np.zeros(family+(3, 3)), np.full(grid, orientation),
        np.full(grid, temperature))


def _periodic_distance(coordinate, center, n):
    delta = np.abs(coordinate-center)
    return np.minimum(delta, n-delta)


def _smooth_voronoi_supports(n, centers, width_cells):
    """Return a partition with a declared *distance* transition scale.

    A Gaussian softmax of squared center distance does not have the requested
    interface width: near a bisector its width is proportional to
    ``width_cells**2 / center_separation``.  That made the former nominally
    physical interface subcell and grid dependent.  A distance softmax has a
    bisector transition proportional to ``width_cells`` itself.
    """
    if width_cells <= 0.0 or not np.isfinite(width_cells):
        raise ValueError("interface width in cells must be positive and finite")
    x, y = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    radial_distance = np.stack([
        np.sqrt(_periodic_distance(x, cx, n)**2
                + _periodic_distance(y, cy, n)**2)
        for cx, cy in centers])
    logits = -radial_distance/float(width_cells)
    logits -= np.max(logits, axis=0, keepdims=True)
    supports = np.exp(logits)
    supports /= np.sum(supports, axis=0, keepdims=True)
    return supports


def initialize_state(n, temperature, equal_density=False, *,
                     spacing_m=1.5625e-7,
                     interface_width_m=3.125e-7):
    centers = ((.22*n, .50*n), (.72*n, .27*n), (.72*n, .73*n))
    # Smooth Voronoi ownership produces resolved pure cores, three distinct
    # orientations, and two incident arms meeting at a real triple junction.
    width_cells = float(interface_width_m)/float(spacing_m)
    supports = _smooth_voronoi_supports(n, centers, width_cells)
    densities = ((4.0e17, 4.0e17, 4.0e17) if equal_density
                 else (4.0e17, 1.0e17, 3.0e17))
    owners = tuple(_owner(n, rho, angle, temperature) for rho, angle in zip(
        densities, (0.0, np.deg2rad(18.0), np.deg2rad(-14.0))))
    state = MultiGrainCommonState((10, 20, 30), supports, owners)
    state.validate()
    return state


def initialize_network_state(n, temperature, grain_count, *,
                             spacing_m=1.5625e-7,
                             interface_width_m=3.125e-7,
                             equal_density=False,
                             initial_temperature_band_K=0.0,
                             initial_temperature_band_width_m=3.125e-7,
                             single_crystal_band_normal="x"):
    """Initialize a resolved deterministic prepared-grain periodic network."""
    count = int(grain_count)
    if count == 1:
        support = np.ones((1, n, n))
        owner = _owner(n, 4.0e17, 0.0, temperature)
        amplitude = float(initial_temperature_band_K)
        width_cells = float(initial_temperature_band_width_m)/float(spacing_m)
        if amplitude < 0.0 or not np.isfinite(amplitude):
            raise ValueError("initial temperature-band amplitude must be nonnegative")
        if width_cells <= 0.0 or not np.isfinite(width_cells):
            raise ValueError("initial temperature-band width must be positive")
        if amplitude > 0.0:
            x, y = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
            if single_crystal_band_normal == "x":
                coordinate = x
            elif single_crystal_band_normal == "y":
                coordinate = y
            elif single_crystal_band_normal == "diagonal":
                coordinate = np.mod(x-y, n)
            else:
                raise ValueError("unknown single-crystal band normal")
            distance = _periodic_distance(coordinate, 0.5*n, n)
            profile = np.exp(-0.5*(distance/width_cells)**2)
            profile -= profile.mean()
            profile /= max(float(profile.max()), 1e-300)
            owner = replace(
                owner, temperature_K=np.asarray(owner.temperature_K)
                +amplitude*profile)
        state = MultiGrainCommonState((10,), support, (owner,))
        state.validate()
        return state
    if count == 3:
        if initial_temperature_band_K != 0.0:
            raise ValueError("temperature-band seed is a single-crystal control")
        return initialize_state(
            n, temperature, equal_density, spacing_m=spacing_m,
            interface_width_m=interface_width_m)
    layouts = {
        4: ((.20, .22), (.72, .18), (.28, .72), (.76, .68)),
        6: ((.16, .22), (.50, .17), (.82, .25),
            (.20, .72), (.54, .78), (.85, .68)),
        8: ((.12, .22), (.38, .17), (.64, .24), (.88, .19),
            (.16, .72), (.42, .79), (.68, .69), (.91, .76)),
    }
    if count not in layouts:
        raise ValueError("V59 grain count must be 1, 3, 4, 6, or 8")
    if initial_temperature_band_K != 0.0:
        raise ValueError("temperature-band seed is a single-crystal control")
    centers = tuple((x*n, y*n) for x, y in layouts[count])
    width_cells = float(interface_width_m)/float(spacing_m)
    supports = _smooth_voronoi_supports(n, centers, width_cells)
    density_scale = (4.0, 1.0, 3.0, 2.2, 3.6, 1.7, 2.8, 1.3)
    angles_deg = (0.0, 18.0, -14.0, 31.0, -27.0, 43.0, -39.0, 9.0)
    densities = ((4.0e17,)*count if equal_density else
                 tuple(value*1e17 for value in density_scale[:count]))
    owners = tuple(_owner(n, density, np.deg2rad(angle), temperature)
                   for density, angle in zip(densities, angles_deg[:count]))
    state = MultiGrainCommonState(
        tuple(10*(index+1) for index in range(count)), supports, owners)
    state.validate()
    return state


def network_interfaces(state):
    """Declare actual dominant-owner adjacencies on the periodic grid."""
    labels = np.argmax(state.supports, axis=0)
    pairs = set()
    for axis in (0, 1):
        neighbor = np.roll(labels, -1, axis=axis)
        for left, right in zip(labels[labels != neighbor], neighbor[labels != neighbor]):
            pairs.add(tuple(sorted((int(left), int(right)))))
    return tuple(MultiGrainInterface(
        f"edge-{state.grain_ids[left]}-{state.grain_ids[right]}",
        state.grain_ids[left], state.grain_ids[right])
        for left, right in sorted(pairs))


def _save_checkpoint(path, state, runtime, step, gamma, initial_volume,
                     configuration, provenance):
    arrays = multigrain_checkpoint_arrays(state)
    arrays["metadata_json"] = np.asarray(multigrain_checkpoint_metadata(state))
    arrays["runtime_json"] = np.asarray(json.dumps({
        "step": int(step), "gamma": float(gamma),
        "ledger": asdict(runtime.ledger),
        "interfaces": [asdict(item) for item in runtime.interfaces],
        "configuration": configuration,
        "provenance": provenance,
    }, sort_keys=True))
    arrays["initial_grain_volume_m3"] = np.asarray(initial_volume)
    np.savez_compressed(path, **arrays)


def _load_checkpoint(path):
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files
                  if key not in ("metadata_json", "runtime_json",
                                 "initial_grain_volume_m3")}
        state = multigrain_from_checkpoint(str(data["metadata_json"]), arrays)
        meta = json.loads(str(data["runtime_json"]))
        initial_volume = np.asarray(data["initial_grain_volume_m3"]).copy()
    runtime = MultiGrainProductionRuntime(
        tuple(MultiGrainInterface(**item) for item in meta["interfaces"]),
        MultiGrainProductionLedger(**meta["ledger"]))
    return (state, runtime, int(meta["step"]), float(meta["gamma"]),
            initial_volume, meta.get("configuration"),
            meta.get("provenance"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--grain-count", type=int, choices=(1, 3, 4, 6, 8), default=3)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--dt", type=float, default=2e-10)
    parser.add_argument("--shear-rate", type=float, default=2e4)
    parser.add_argument("--temperature", type=float, default=900.0)
    parser.add_argument("--case", choices=("baseline", "equal_density", "no_front"),
                        default="baseline")
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--length", type=float, default=5e-6)
    parser.add_argument("--interface-width", type=float, default=3.125e-7)
    parser.add_argument("--initial-temperature-band-K", type=float, default=0.0)
    parser.add_argument("--initial-temperature-band-width", type=float,
                        default=3.125e-7)
    parser.add_argument("--single-crystal-band-normal",
                        choices=("x", "y", "diagonal"), default="x")
    parser.add_argument("--front-attempt-frequency", type=float, default=1e8)
    parser.add_argument("--front-maximum-fraction", type=float, default=.015,
                        help="numerical contour-CFL bound per front substep")
    parser.add_argument(
        "--front-maximum-substep", type=float, default=1e-8,
        help=("maximum physical front integration substep in seconds; "
              "affinity and dissipation are recomputed after each substep"))
    parser.add_argument("--maximum-mechanical-subdivisions", type=int,
                        default=10)
    parser.add_argument(
        "--mechanical-maximum-fraction", type=float, default=.10,
        help="maximum reservoir/bound change per accepted Euler microstep")
    parser.add_argument("--thermal-diffusivity", type=float,
                        default=0.15/3.8e6)
    parser.add_argument("--flow-temperature-mode",
                        choices=("physical", "frozen"), default="physical")
    parser.add_argument(
        "--recovery-temperature-mode", choices=("physical", "frozen"),
        default="physical",
        help=("independent causal routing for locking, annihilation, wall, "
              "junction, and ordering kinetics; the physical temperature "
              "state and heat equation always continue to evolve"))
    parser.add_argument(
        "--front-temperature-mode", choices=("physical", "frozen"),
        default="physical",
        help=("independent causal routing for the EXP-floor moving-front "
              "kinetics; the physical temperature state and heat equation "
              "always continue to evolve"))
    parser.add_argument(
        "--front-heat-deposition",
        choices=("local_realized_event", "uniform_ablation"),
        default="local_realized_event",
        help=("deposit front dissipation on its realized cellwise event "
              "measure; uniform_ablation is a labeled matched diagnostic"))
    parser.add_argument(
        "--mechanics-mode", choices=("physical", "frozen_hold"),
        default="physical",
        help=("frozen_hold isolates stored-energy front migration at fixed "
              "strain and requires zero applied shear rate"))
    parser.add_argument("--resume")
    parser.add_argument("--source-commit")
    parser.add_argument(
        "--resume-transition", choices=(
            "none", "common_temperature_once_from_v2",
            "adaptive_bisection_controller_v5",
            "v58_smaller_macro_step_from_97b91df",
            "v59_conjugate_front_from_ce3d101",
            "v59_all_temperature_routing_from_42a5432",
            "v60_local_front_heat_from_9b40708"),
        default="none")
    parser.add_argument("--expected-resume-sha256")
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    spacing = args.length/args.n; thickness = 2.0*2.48e-10
    head_commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), text=True).strip()
    if args.source_commit:
        declared_commit = subprocess.check_output(
            ("git", "rev-parse", args.source_commit), text=True).strip()
        if declared_commit != head_commit:
            raise ValueError(
                "declared source commit does not match the executing worktree HEAD")
    source_commit = head_commit
    configuration = {
        "schema": ("v59-single-crystal-production-v1"
                   if args.grain_count == 1 else
                   "v58-three-grain-production-v4-geometric-front"
                   if args.grain_count == 3 else
                   "v59-prepared-network-production-v1"),
        "n": args.n, "length_m": args.length,
        "grain_count": args.grain_count,
        "interface_width_m": args.interface_width, "dt_s": args.dt,
        "initial_temperature_band_K": args.initial_temperature_band_K,
        "initial_temperature_band_width_m": args.initial_temperature_band_width,
        "single_crystal_band_normal": args.single_crystal_band_normal,
        "shear_rate_s": args.shear_rate, "temperature_K": args.temperature,
        "case": args.case,
        "front_attempt_frequency_s": args.front_attempt_frequency,
        "front_maximum_fraction_per_step": args.front_maximum_fraction,
        "front_maximum_substep_s": args.front_maximum_substep,
        "thermal_diffusivity_m2_s": args.thermal_diffusivity,
        "transport_scheme": "upwind",
        "maximum_fraction_per_step": args.mechanical_maximum_fraction,
        "flow_temperature_mode": args.flow_temperature_mode,
        "recovery_temperature_mode": args.recovery_temperature_mode,
        "front_temperature_mode": args.front_temperature_mode,
        "front_heat_deposition": args.front_heat_deposition,
        "mechanics_mode": args.mechanics_mode,
        "mechanical_stress_integrator": "synchronized_common_stress_substeps",
        "front_sweep_measure": "level_set_gradient_pair_partition",
        "front_closure_fraction": .05,
    }
    provenance = {"source_commit": source_commit}
    if args.mechanics_mode == "frozen_hold" and args.shear_rate != 0.0:
        raise ValueError("frozen_hold mechanics requires zero shear rate")
    systems = bcc_four_family_systems()
    wall = CommonWallParameters(
        spacing_m=spacing, elastic_iterations=2,
        mobile_correlation_diffusivity_m2_s=0.0,
        transport_scheme="upwind",
        maximum_fraction_per_step=args.mechanical_maximum_fraction,
        flow_temperature_override_K=(
            args.temperature if args.flow_temperature_mode == "frozen"
            else None),
        recovery_temperature_override_K=(
            args.temperature if args.recovery_temperature_mode == "frozen"
            else None),
        volumetric_heat_capacity_J_m3_K=3.8e6,
        thermal_diffusivity_m2_s=args.thermal_diffusivity, bath_rate_s=0.0)
    kinetics = MultiGrainFrontKinetics(
        ActivatedProcess(
            "V58 multi-grain HAGB", args.front_attempt_frequency, 0.0,
            args.front_attempt_frequency),
        PhysicalTransferLaw(.55, .08, .02), .35*EV_J, 1e9,
        2.0, 1.5, .10, wall.burgers_m**3, wall.burgers_m,
        # This is a numerical contour-CFL control, recorded in the bound
        # configuration and varied only for refinement diagnostics.
        virtual_fraction=2e-4,
        maximum_fraction_per_step=args.front_maximum_fraction,
        closure_fraction=.05, maximum_backtracks=14,
        temperature_override_K=(
            args.temperature if args.front_temperature_mode == "frozen"
            else None))
    if args.resume:
        (state, runtime, start, gamma, initial_volume,
         checkpoint_configuration, checkpoint_provenance) = _load_checkpoint(
             args.resume)
        resume_sha = hashlib.sha256(Path(args.resume).read_bytes()).hexdigest()
        if args.expected_resume_sha256 and resume_sha != args.expected_resume_sha256:
            raise ValueError("resume checkpoint SHA-256 does not match expectation")
        if args.resume_transition == "common_temperature_once_from_v2":
            if not args.expected_resume_sha256:
                raise ValueError("repair transition requires expected checkpoint SHA-256")
            legacy = dict(checkpoint_configuration or {})
            if legacy.pop("schema", None) != (
                    "v58-three-grain-production-v2-common-stress-upwind"):
                raise ValueError("temperature repair transition requires v2 checkpoint")
            current = dict(configuration); current.pop("schema")
            current.pop("flow_temperature_mode")
            if legacy != current or args.flow_temperature_mode != "physical":
                raise ValueError("v2 checkpoint physical configuration mismatch")
            provenance["restart_transition"] = args.resume_transition
            provenance["parent_checkpoint_sha256"] = resume_sha
            provenance["parent_source_commit"] = "f3d7d7f"
        elif args.resume_transition == "adaptive_bisection_controller_v5":
            if not args.expected_resume_sha256:
                raise ValueError("controller transition requires expected checkpoint SHA-256")
            if (checkpoint_configuration != configuration
                    or not checkpoint_provenance
                    or checkpoint_provenance.get("source_commit") != "ed1cb78"):
                raise ValueError(
                    "adaptive controller transition requires the ed1cb78 "
                    "exact-elasticity configuration")
            provenance["restart_transition"] = args.resume_transition
            provenance["parent_checkpoint_sha256"] = resume_sha
            provenance["parent_source_commit"] = "ed1cb78"
        elif args.resume_transition == "v58_smaller_macro_step_from_97b91df":
            if not args.expected_resume_sha256:
                raise ValueError(
                    "time-refinement transition requires expected checkpoint SHA-256")
            legacy = dict(checkpoint_configuration or {})
            current = dict(configuration)
            legacy_dt = legacy.pop("dt_s", None)
            current_dt = current.pop("dt_s", None)
            if (legacy != current or not checkpoint_provenance
                    or checkpoint_provenance.get("source_commit") != "97b91df"
                    or current_dt is None or legacy_dt is None
                    or not float(current_dt) < float(legacy_dt)):
                raise ValueError(
                    "V58 time refinement requires the exact 97b91df state "
                    "and an otherwise identical smaller-step configuration")
            provenance["restart_transition"] = args.resume_transition
            provenance["parent_checkpoint_sha256"] = resume_sha
            provenance["parent_source_commit"] = "97b91df"
            provenance["parent_dt_s"] = float(legacy_dt)
        elif args.resume_transition == "v59_conjugate_front_from_ce3d101":
            if not args.expected_resume_sha256:
                raise ValueError(
                    "V59 force-rate transition requires expected checkpoint SHA-256")
            legacy_configuration = dict(checkpoint_configuration or {})
            legacy_configuration.setdefault("grain_count", 3)
            if (legacy_configuration != configuration
                    or not checkpoint_provenance
                    or checkpoint_provenance.get("source_commit") != "ce3d101"):
                raise ValueError(
                    "V59 force-rate transition requires an exact ce3d101 "
                    "checkpoint with unchanged physical configuration")
            provenance["restart_transition"] = args.resume_transition
            provenance["parent_checkpoint_sha256"] = resume_sha
            provenance["parent_source_commit"] = "ce3d101"
            provenance["governing_change"] = (
                "fixed-point conjugacy between constrained complete front "
                "force, EXP-floor rate, and mobility dissipation")
        elif args.resume_transition == "v59_all_temperature_routing_from_42a5432":
            if not args.expected_resume_sha256:
                raise ValueError(
                    "V59 temperature-routing transition requires an exact "
                    "parent checkpoint checksum")
            legacy = dict(checkpoint_configuration or {})
            current = dict(configuration)
            if current.pop("front_temperature_mode", None) != "physical":
                raise ValueError(
                    "only the unchanged physical front-temperature path may "
                    "cross the routing transition")
            if (current.pop("initial_temperature_band_K", None) != 0.0
                    or current.pop("initial_temperature_band_width_m", None)
                    != 3.125e-7
                    or current.pop("single_crystal_band_normal", None) != "x"):
                raise ValueError(
                    "temperature-routing transition requires the unchanged "
                    "three-grain analytic initialization")
            if (legacy != current or not checkpoint_provenance
                    or checkpoint_provenance.get("source_commit")
                    != "42a5432173cc77ebc70faa0d20fa922189e33c66"):
                raise ValueError(
                    "V59 temperature-routing transition requires the exact "
                    "42a5432 physical-temperature baseline configuration")
            provenance["restart_transition"] = args.resume_transition
            provenance["parent_checkpoint_sha256"] = resume_sha
            provenance["parent_source_commit"] = checkpoint_provenance[
                "source_commit"]
            provenance["governing_change"] = (
                "added an independently frozen moving-front temperature "
                "route and single-crystal controls; all temperature routes "
                "remain physical and production evolution is unchanged in "
                "this baseline continuation")
        elif args.resume_transition == "v60_local_front_heat_from_9b40708":
            if not args.expected_resume_sha256:
                raise ValueError(
                    "V60 heat-deposition transition requires an exact parent "
                    "checkpoint checksum")
            legacy = dict(checkpoint_configuration or {})
            current = dict(configuration)
            deposition = current.pop("front_heat_deposition", None)
            if (legacy != current or not checkpoint_provenance
                    or checkpoint_provenance.get("source_commit")
                    != "9b40708a775699d00b3504c97cee879f85a54ff0"
                    or deposition not in {
                        "local_realized_event", "uniform_ablation"}):
                raise ValueError(
                    "V60 heat transition requires the exact 9b40708 state "
                    "with unchanged physical configuration")
            provenance["restart_transition"] = args.resume_transition
            provenance["parent_checkpoint_sha256"] = resume_sha
            provenance["parent_source_commit"] = checkpoint_provenance[
                "source_commit"]
            provenance["governing_change"] = (
                "front mobility dissipation is retained cellwise as pressure "
                "times realized accepted extent; past heat is not relocated")
            provenance["front_heat_deposition"] = deposition
        elif (checkpoint_configuration != configuration
              or not checkpoint_provenance
              or checkpoint_provenance.get("source_commit") != source_commit):
            raise ValueError(
                "checkpoint physical configuration or source provenance differs")
        else:
            # Exact same-source restarts were already validated above, but
            # previously failed to carry an explicit parent edge into the new
            # checkpoint provenance.  Preserve that edge so restart evidence
            # is machine-auditable without reconstructing the launch command.
            provenance["restart_transition"] = "exact_same_source_restart"
            provenance["parent_checkpoint_sha256"] = resume_sha
            provenance["parent_source_commit"] = checkpoint_provenance[
                "source_commit"]
    else:
        state = initialize_network_state(
            args.n, args.temperature, args.grain_count,
            spacing_m=spacing, interface_width_m=args.interface_width,
            equal_density=args.case == "equal_density",
            initial_temperature_band_K=args.initial_temperature_band_K,
            initial_temperature_band_width_m=(
                args.initial_temperature_band_width),
            single_crystal_band_normal=args.single_crystal_band_normal)
        runtime = MultiGrainProductionRuntime(network_interfaces(state))
        start = 0; gamma = .012
        initial_volume = np.sum(
            state.supports, axis=(1, 2))*spacing**2*thickness
    history_path = out/"history.json"
    history = (json.loads(history_path.read_text())
               if args.resume and history_path.exists() else [])
    energy_options = dict(
        mean_strain=np.array([[0.0, .5*gamma], [.5*gamma, 0.0]]),
        topologies=(), systems=systems, phase_barrier_J_m3=5e6,
        phase_gradient_J_m=5e-7,
        boundary_line_energy_J_m=wall.line_energy_J_m,
        boundary_junction_energy_J_m=wall.junction_energy_J_m,
        reference_temperature_K=args.temperature)
    for step in range(start, args.steps):
        gamma_before = gamma
        gamma += args.shear_rate*args.dt
        strain_before = np.array(
            [[0.0, .5*gamma_before], [.5*gamma_before, 0.0]])
        strain_after = np.array([[0.0, .5*gamma], [.5*gamma, 0.0]])
        strain_midpoint = .5*(strain_before+strain_after)
        if args.mechanics_mode == "physical":
            mechanical = advance_energy_qualified_mechanics(
                state, mean_strain_before=strain_before,
                mean_strain_candidate=strain_after,
                systems=systems, topologies=(), wall_parameters=wall,
                dt_s=args.dt, represented_thickness_m=thickness,
                energy_kwargs=energy_options,
                maximum_relative_first_law_residual=.05,
                maximum_subdivisions=args.maximum_mechanical_subdivisions)
            state = mechanical.state
            shear_stress = float(
                mechanical.energy_balances[-1].mean_stress_candidate_Pa[0, 1])
            mechanical_record = {
                "accepted": True,
                "mode": "physical",
                "subintervals": len(mechanical.operator_decisions),
                "subdivision_depth": mechanical.subdivisions,
                "minimum_step_scale": min(
                    value.minimum_step_scale
                    for value in mechanical.operator_decisions),
                "maximum_substeps": max(
                    value.maximum_substeps
                    for value in mechanical.operator_decisions),
                "external_plastic_work_J": sum(
                    value.external_plastic_work_J
                    for value in mechanical.operator_decisions),
                "irreversible_heat_J": sum(
                    value.irreversible_heat_J
                    for value in mechanical.operator_decisions),
                "consumed_interval_s": sum(
                    value.consumed_interval_s
                    for value in mechanical.operator_decisions),
            }
            localization_record = _localization_diagnostics(
                sum((value.plastic_work_J_m3_cells
                     for value in mechanical.operator_decisions),
                    np.zeros((args.n, args.n)))/args.dt,
                sum((value.irreversible_heat_J_m3_cells
                     for value in mechanical.operator_decisions),
                    np.zeros((args.n, args.n)))/args.dt)
            mechanical_energy_record = {
                "external_work_J": mechanical.external_work_J,
                "internal_energy_change_J": mechanical.internal_energy_change_J,
                "first_law_residual_J": mechanical.first_law_residual_J,
                "relative_first_law_residual": (
                    mechanical.relative_first_law_residual),
                "maximum_leaf_relative_first_law_residual": max(
                    value.relative_first_law_residual
                    for value in mechanical.energy_balances),
            }
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
        else:
            mechanical = None
            shear_stress = float(_mean_mechanical_stress(
                state, spacing, wall, strain_after)[0, 1])
            mechanical_record = {
                "accepted": True, "mode": "frozen_hold",
                "subintervals": 0, "subdivision_depth": 0,
                "minimum_step_scale": 1.0, "maximum_substeps": 0,
                "external_plastic_work_J": 0.0,
                "irreversible_heat_J": 0.0,
                "consumed_interval_s": args.dt,
            }
            zeros = np.zeros((args.n, args.n))
            localization_record = _localization_diagnostics(zeros, zeros)
            mechanical_energy_record = {
                "external_work_J": 0.0, "internal_energy_change_J": 0.0,
                "first_law_residual_J": 0.0,
                "relative_first_law_residual": 0.0,
                "maximum_leaf_relative_first_law_residual": 0.0,
            }
        energy_options["mean_strain"] = strain_after
        if args.case == "no_front":
            fronts = ()
            runtime = MultiGrainProductionRuntime(
                runtime.interfaces, replace_ledger(runtime.ledger, args.dt,
                                                   args.shear_rate))
        else:
            state, runtime, fronts = advance_multigrain_front_interval(
                state, runtime, kinetics=kinetics, dt_s=args.dt,
                maximum_substep_s=args.front_maximum_substep,
                spacing_m=spacing, represented_thickness_m=thickness,
                wall_parameters=wall, energy_kwargs=energy_options,
                applied_shear_rate_s=args.shear_rate, systems=systems,
                heat_deposition_mode=args.front_heat_deposition)
        if (step == start
                or (step+1) % max(args.checkpoint_every, 1) == 0
                or step+1 == args.steps):
            energy = evaluate_complete_multigrain_energy(
                state, spacing_m=spacing,
                represented_thickness_m=thickness,
                wall_parameters=wall, **energy_options)
            audit = audit_multigrain_nye(state, spacing)
            volumes = np.sum(state.supports, axis=(1, 2))*spacing**2*thickness
            temperatures = sum(state.supports[index]*owner.temperature_K
                               for index, owner in enumerate(state.owners))
            history.append({
                "step": step+1, "time_s": runtime.ledger.physical_time_s,
                "applied_shear_strain": gamma,
                "shear_stress_Pa": float(shear_stress),
                "grain_volume_m3": volumes.tolist(),
                "grain_volume_change_m3": (volumes-initial_volume).tolist(),
                "temperature_mean_K": float(np.mean(temperatures)),
                "temperature_contrast_K": float(np.max(temperatures)
                                                -np.min(temperatures)),
                "helmholtz_J": energy.helmholtz_J,
                "thermal_internal_J": energy.thermal_internal_J,
                "support_gradient_nye_norm_m1": float(np.linalg.norm(
                    audit.support_gradient_m1)),
                "owner_nye_mismatch_norm_m1": float(np.linalg.norm(
                    audit.owner_reservoir_mismatch_m1)),
                "mechanical": mechanical_record,
                "localization": localization_record,
                "mechanical_energy": mechanical_energy_record,
                "front": None if not fronts else {
                    "accepted": all(item.accepted for item in fronts),
                    "classification": (
                        fronts[-1].classification if len(fronts) == 1 else
                        "SUBCYCLED_" + (
                            "ACCEPTED" if all(item.accepted for item in fronts)
                            else "WITH_REJECTION")),
                    "directions": fronts[-1].direction_by_interface,
                    "pressures_Pa": fronts[-1].pressure_by_interface_Pa,
                    "independent_pressures_Pa": (
                        fronts[-1].independent_pressure_by_interface_Pa),
                    "selected_velocities_m_s": (
                        fronts[-1].selected_velocity_by_interface_m_s),
                    "joint_pressure_factor": fronts[-1].joint_pressure_factor,
                    "selected_rate_conjugate_to_recorded_force": (
                        fronts[-1].selected_rate_conjugate_to_recorded_force),
                    "backtracks": max(item.backtracks for item in fronts),
                    "subintervals": len(fronts),
                    "rejected_subintervals": sum(
                        not item.accepted for item in fronts),
                    "classifications": [
                        item.classification for item in fronts],
                },
                "runtime": asdict(runtime.ledger),
            })
            checkpoint = out/f"checkpoint_{step+1:06d}.npz"
            _save_checkpoint(
                checkpoint, state, runtime, step+1, gamma, initial_volume,
                configuration, provenance)
            (out/"history.json").write_text(json.dumps(history, indent=2))
            print(json.dumps(history[-1], sort_keys=True), flush=True)
    latest = sorted(out.glob("checkpoint_*.npz"))[-1]
    result = {
        "schema": ("asb-drx-v59-single-crystal-production-v1"
                   if args.grain_count == 1 else
                   "asb-drx-v58-three-grain-production-v1"
                   if args.grain_count == 3 else
                   "asb-drx-v59-prepared-network-production-v1"),
        "case": args.case, "n": args.n, "steps": args.steps,
        "dt_s": args.dt, "temperature_K": args.temperature,
        "shear_rate_s": args.shear_rate,
        "length_m": args.length, "spacing_m": spacing,
        "represented_thickness_m": thickness,
        "interface_width_m": args.interface_width,
        "front_attempt_frequency_s": args.front_attempt_frequency,
        "front_maximum_substep_s": args.front_maximum_substep,
        "front_heat_deposition": args.front_heat_deposition,
        "thermal_diffusivity_m2_s": args.thermal_diffusivity,
        "source_commit": source_commit,
        "initial_grain_volume_m3": initial_volume.tolist(),
        "final": history[-1], "runtime": asdict(runtime.ledger),
        "checkpoint": str(latest),
        "checkpoint_sha256": hashlib.sha256(latest.read_bytes()).hexdigest(),
        "configuration": configuration,
        "provenance": provenance,
    }
    (out/"result.json").write_text(json.dumps(result, indent=2))


def replace_ledger(ledger, dt, shear_rate):
    from dataclasses import replace
    return replace(
        ledger, intervals=ledger.intervals+1,
        stationary_intervals=ledger.stationary_intervals+1,
        physical_time_s=ledger.physical_time_s+dt,
        applied_shear_strain=ledger.applied_shear_strain+shear_rate*dt)


def _localization_diagnostics(plastic_power, heat_rate):
    """Return outcome-neutral instantaneous spatial concentration metrics."""
    return {
        "plastic_power": spatial_localization_metrics(plastic_power),
        "irreversible_heat_rate": spatial_localization_metrics(heat_rate),
    }


if __name__ == "__main__":
    main()
