#!/usr/bin/env python3
"""Run a restartable complete-interval, error-controlled production suffix."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from full_model.analysis.postprocess_v59_physical_asb import _wall_parameters
from full_model.analysis.run_v58_three_grain_production import (
    _load_checkpoint, _save_checkpoint,
)
from full_model.production.arrhenius_kinetics import ActivatedProcess, EV_J
from full_model.production.complete_multigrain_energy import (
    evaluate_complete_multigrain_energy,
)
from full_model.production.coupled_interval_control import (
    AdaptiveControllerState, ErrorMetric, advance_error_controlled_interval,
)
from full_model.production.coupled_multigrain_interval import (
    CoupledMultigrainContext, advance_complete_multigrain_interval, make_endpoint,
)
from full_model.production.multigrain_common_state import PhysicalTransferLaw
from full_model.production.multigrain_production import MultiGrainFrontKinetics
from full_model.production.tensorial_nye import bcc_four_family_systems


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def target_pending(current_time_s: float, target_time_s: float) -> bool:
    tolerance = 128*np.finfo(float).eps*max(
        abs(float(target_time_s)), np.finfo(float).tiny)
    return float(current_time_s) < float(target_time_s)-tolerance


def _physical_configuration(configuration: dict) -> dict:
    if configuration.get("schema") == "asb-drx-v64-adaptive-production-v1":
        return dict(configuration["physical_configuration"])
    return dict(configuration)


def _controller_from_checkpoint(path: Path):
    with np.load(path, allow_pickle=False) as data:
        if "diagnostic_adaptive_controller_json" not in data.files:
            return None
        return AdaptiveControllerState.from_dict(json.loads(
            str(data["diagnostic_adaptive_controller_json"])))


def _front_factor_from_checkpoint(path: Path):
    with np.load(path, allow_pickle=False) as data:
        if "diagnostic_front_joint_pressure_factor" not in data.files:
            return None
        value = float(data["diagnostic_front_joint_pressure_factor"])
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("saved front pressure factor is not positive finite")
        return value


def restore_horizon_proposal(
        controller: AdaptiveControllerState, proposal_s: float
        ) -> AdaptiveControllerState:
    """Undo only a smaller exact-target clip in a persisted controller."""
    proposal = float(proposal_s)
    if not (controller.minimum_step_s <= proposal
            <= controller.maximum_step_s):
        raise ValueError("restored proposal lies outside controller bounds")
    if proposal < controller.proposed_step_s:
        raise ValueError("horizon restoration may not reduce the proposal")
    return replace(
        controller, proposed_step_s=proposal,
        consecutive_low_error_accepts=0)


def _context(configuration: dict, *, warm_start_front_factor=False
             ) -> CoupledMultigrainContext:
    n = int(configuration["n"])
    spacing = float(configuration["length_m"])/n
    thickness = 2.0*2.48e-10
    systems = bcc_four_family_systems()
    wall = _wall_parameters(configuration, spacing)
    attempt = float(configuration["front_attempt_frequency_s"])
    kinetics = MultiGrainFrontKinetics(
        ActivatedProcess("V64 multi-grain HAGB", attempt, 0.0, attempt),
        PhysicalTransferLaw(.55, .08, .02), .35*EV_J, 1e9,
        2.0, 1.5, .10, wall.burgers_m**3, wall.burgers_m,
        virtual_fraction=2e-4,
        maximum_fraction_per_step=float(
            configuration["front_maximum_fraction_per_step"]),
        closure_fraction=.05, maximum_backtracks=14,
        temperature_override_K=(
            float(configuration["temperature_K"])
            if configuration.get("front_temperature_mode") == "frozen"
            else None),
        temperature_resolution=configuration.get(
            "front_temperature_resolution", "global_mean"))
    energy = {
        "topologies": (), "systems": systems,
        "phase_barrier_J_m3": 5e6, "phase_gradient_J_m": 5e-7,
        "boundary_line_energy_J_m": wall.line_energy_J_m,
        "boundary_junction_energy_J_m": wall.junction_energy_J_m,
        "reference_temperature_K": float(configuration["temperature_K"]),
    }
    return CoupledMultigrainContext(
        systems, (), wall, kinetics, spacing, thickness,
        float(configuration["shear_rate_s"]),
        float(configuration["front_maximum_substep_s"]), 10, energy,
        heat_deposition_mode=configuration.get(
            "front_heat_deposition", "local_realized_event"),
        front_enabled=configuration.get("case") != "no_front",
        warm_start_front_factor=warm_start_front_factor)


def _metrics(parent, state_tolerance, rate_tolerance,
             integral_tolerance) -> tuple[ErrorMetric, ...]:
    obs = parent.observables
    size = lambda name: np.sqrt(np.asarray(obs[name]).size)
    return (
        ErrorMetric("supports", "endpoint_increment", state_tolerance,
                    1e-9*size("supports")),
        ErrorMetric("common_beta_p", "endpoint_increment", state_tolerance,
                    1e-12*size("common_beta_p")),
        ErrorMetric("signed_total_density_m2", "endpoint_increment",
                    state_tolerance, 1e8*size("signed_total_density_m2")),
        ErrorMetric("exact_reconstructed_nye_m1", "endpoint_increment",
                    state_tolerance, 1e2*size("exact_reconstructed_nye_m1")),
        ErrorMetric("common_temperature_K", "endpoint_increment",
                    state_tolerance, 1e-3*size("common_temperature_K")),
        ErrorMetric("common_stress_tensor_Pa", "endpoint_signal",
                    state_tolerance, 1e6*size("common_stress_tensor_Pa")),
        ErrorMetric("instantaneous_plastic_power_W_m3", "endpoint_signal",
                    rate_tolerance, 1e8*size(
                        "instantaneous_plastic_power_W_m3")),
        ErrorMetric("instantaneous_plastic_power_change_W_m3",
                    "endpoint_increment", rate_tolerance,
                    1e8*size("instantaneous_plastic_power_W_m3"),
                    "instantaneous_plastic_power_W_m3"),
        ErrorMetric("instantaneous_signed_slip_rate_s", "endpoint_signal",
                    rate_tolerance, 1e-7*size(
                        "instantaneous_signed_slip_rate_s")),
        ErrorMetric("instantaneous_signed_slip_rate_change_s",
                    "endpoint_increment", rate_tolerance,
                    1e-7*size("instantaneous_signed_slip_rate_s"),
                    "instantaneous_signed_slip_rate_s"),
        ErrorMetric("owner_weighted_speed_m_s", "endpoint_signal",
                    rate_tolerance, 1e-12*size("owner_weighted_speed_m_s")),
        ErrorMetric("owner_weighted_speed_change_m_s", "endpoint_increment",
                    rate_tolerance, 1e-12*size("owner_weighted_speed_m_s"),
                    "owner_weighted_speed_m_s"),
        ErrorMetric("owner_weighted_signed_slip_rate_s", "endpoint_signal",
                    rate_tolerance, 1e-7*size(
                        "owner_weighted_signed_slip_rate_s")),
        ErrorMetric("owner_weighted_signed_slip_rate_change_s",
                    "endpoint_increment", rate_tolerance,
                    1e-7*size("owner_weighted_signed_slip_rate_s"),
                    "owner_weighted_signed_slip_rate_s"),
        ErrorMetric("mechanical_external_work_J", "interval_integral",
                    integral_tolerance, 1e-20),
        ErrorMetric("mechanical_irreversible_heat_J", "interval_integral",
                    integral_tolerance, 1e-20),
        ErrorMetric("front_heat_J", "interval_integral",
                    integral_tolerance, 1e-20),
        ErrorMetric("mechanical_work_J_m3", "interval_integral",
                    integral_tolerance, 1.0*np.sqrt(
                        np.asarray(obs["common_temperature_K"]).size)),
        ErrorMetric("mechanical_heat_J_m3", "interval_integral",
                    integral_tolerance, 1.0*np.sqrt(
                        np.asarray(obs["common_temperature_K"]).size)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--expected-resume-sha256", required=True)
    parser.add_argument("--target-time", type=float, required=True)
    parser.add_argument("--initial-step", type=float, default=6.25e-9)
    parser.add_argument("--minimum-step", type=float, default=9.765625e-11)
    parser.add_argument("--maximum-step", type=float, default=6.25e-9)
    parser.add_argument("--state-tolerance", type=float, default=.02)
    parser.add_argument("--rate-tolerance", type=float, default=.02)
    parser.add_argument("--integral-tolerance", type=float, default=.02)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument(
        "--restart-proposed-step", type=float,
        help=("replace only the restored controller proposal after an exact "
              "target-horizon clip; physical state and tolerances are unchanged"))
    parser.add_argument("--source-commit")
    parser.add_argument("--warm-start-front-factor", action="store_true")
    parser.add_argument(
        "--numerical-transition",
        choices=("none", "v64_bracketed_front_root",
                 "v64_adaptive_growth", "v64_exact_front_cache",
                 "v64_set_valued_active_cone",
                 "v65_terminal_horizon_restore",
                 "v65_warm_started_front"),
        default="none")
    args = parser.parse_args()
    if digest(args.resume) != args.expected_resume_sha256:
        raise ValueError("resume checkpoint SHA-256 does not match expectation")
    head = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), text=True).strip()
    if args.source_commit and subprocess.check_output(
            ("git", "rev-parse", args.source_commit), text=True).strip() != head:
        raise ValueError("declared source commit does not match executing HEAD")
    (state, runtime, parent_step, gamma, initial_volume,
     checkpoint_configuration, checkpoint_provenance) = _load_checkpoint(
        args.resume)
    analytic_origin_parent = bool(
        runtime.ledger.physical_time_s == 0.0
        and parent_step == 0
        and (checkpoint_provenance or {}).get("origin")
        == "deterministic_analytic_prepared_network")
    physical = _physical_configuration(checkpoint_configuration or {})
    context = _context(
        physical, warm_start_front_factor=args.warm_start_front_factor)
    if float(args.target_time) <= runtime.ledger.physical_time_s:
        raise ValueError("target physical time must exceed the checkpoint time")
    restored = _controller_from_checkpoint(args.resume)
    controller = restored or AdaptiveControllerState(
        args.initial_step, args.minimum_step, args.maximum_step)
    if restored is not None and not (
            np.isclose(restored.minimum_step_s, args.minimum_step)
            and np.isclose(restored.maximum_step_s, args.maximum_step)):
        raise ValueError("restart controller bounds differ from checkpoint")
    if restored is not None:
        prior_accuracy = (checkpoint_configuration or {}).get("accuracy", {})
        requested_accuracy = {
            "state_relative_tolerance": args.state_tolerance,
            "rate_relative_tolerance": args.rate_tolerance,
            "integral_relative_tolerance": args.integral_tolerance,
        }
        if any(not np.isclose(float(prior_accuracy.get(name, np.nan)), value,
                              rtol=0.0, atol=0.0)
               for name, value in requested_accuracy.items()):
            raise ValueError("restart accuracy tolerances differ from checkpoint")
    if args.restart_proposed_step is not None:
        if restored is None:
            raise ValueError("proposal restoration requires a saved controller")
        controller = restore_horizon_proposal(
            controller, args.restart_proposed_step)
    parent_source = (checkpoint_provenance or {}).get("source_commit")
    source_changed = parent_source not in {None, head}
    if restored is not None and source_changed and args.numerical_transition == "none":
        raise ValueError("adaptive source change requires a declared transition")
    if args.numerical_transition != "none" and not (
            restored is not None and source_changed):
        raise ValueError(
            "declared numerical transition requires a changed adaptive source")
    restores_terminal_horizon = args.numerical_transition in {
        "v65_terminal_horizon_restore", "v65_warm_started_front"}
    if (restores_terminal_horizon
            != (args.restart_proposed_step is not None)):
        raise ValueError(
            "V65 terminal-horizon transition and proposal restoration must "
            "be declared together")
    if args.numerical_transition == "v65_warm_started_front":
        if not (args.warm_start_front_factor
                and args.restart_proposed_step is not None):
            raise ValueError(
                "V65 warm-start transition requires warm start and explicit "
                "terminal-horizon proposal restoration")
    elif args.warm_start_front_factor:
        raise ValueError("front warm start requires its declared transition")
    provenance = {
        "source_commit": head,
        "restart_transition": (
            args.numerical_transition if args.numerical_transition != "none"
            else "v64_analytic_origin_accuracy_control" if analytic_origin_parent
            else "v64_complete_interval_accuracy_control"),
        "parent_checkpoint_sha256": args.expected_resume_sha256,
        "parent_source_commit": (checkpoint_provenance or {}).get(
            "source_commit"),
        "physical_kernels_changed": False,
        "numerical_change": (
            "complete coupled step doubling with private rollback and fine-path "
            "publication; established mechanics/front constitutive laws unchanged"),
    }
    if args.numerical_transition == "v64_bracketed_front_root":
        provenance["numerical_change"] = (
            "safeguarded bracketed scalar solution of the existing joint "
            "front force-rate conjugacy residual after fixed-point stagnation; "
            "unchanged functional, root tolerance, mobility, and physical laws")
    elif args.numerical_transition == "v64_adaptive_growth":
        provenance["numerical_change"] = (
            "bounded 1.25x adaptive proposal after two consecutive accepts "
            "below half tolerance; every proposal retains the complete coarse/"
            "two-half test, rollback, and unchanged physical tolerances")
    elif args.numerical_transition == "v64_exact_front_cache":
        provenance["numerical_change"] = (
            "per-map reuse of request-invariant directed product owners, line "
            "exports, and physical channels; joint capacity, complete energy, "
            "force/rate closure, and final hash-bound publication are recomputed")
    elif args.numerical_transition == "v64_set_valued_active_cone":
        provenance["numerical_change"] = (
            "explicit complementarity closure when a sign-changing front "
            "active-cone bracket collapses to adjacent floating-point values; "
            "the interval inclusion is independently reconstructed, while the "
            "unchanged pointwise force/rate tolerance remains mandatory away "
            "from that non-single-valued limit")
    elif args.numerical_transition == "v65_terminal_horizon_restore":
        provenance["numerical_change"] = (
            "restore the pre-terminal ordinary adaptive proposal after an "
            "exact-target horizon clip; physical state, controller bounds, "
            "complete coarse/two-half trial, and all 2% tolerances unchanged")
        provenance["restored_proposed_step_s"] = controller.proposed_step_s
    elif args.numerical_transition == "v65_warm_started_front":
        provenance["numerical_change"] = (
            "restore the pre-terminal ordinary proposal and warm-start each "
            "unchanged joint front conjugacy root from the last accepted "
            "factor; every root is independently converged and verified, and "
            "physical laws, complete-map trials, and 2% tolerances are unchanged")
        provenance["restored_proposed_step_s"] = controller.proposed_step_s
    args.out.mkdir(parents=True, exist_ok=True)
    history_path = args.out/"history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    saved_front_factor = _front_factor_from_checkpoint(args.resume)
    endpoint = make_endpoint(
        state, runtime, gamma, context,
        numerical_hints=({"front_joint_pressure_factor": saved_front_factor}
                         if (args.warm_start_front_factor
                             and saved_front_factor is not None) else {}))
    metrics = _metrics(endpoint, args.state_tolerance, args.rate_tolerance,
                       args.integral_tolerance)
    metric_definitions = [asdict(metric) for metric in metrics]
    if restored is not None and prior_accuracy.get(
            "metric_definitions") != metric_definitions:
        raise ValueError(
            "restart accuracy metric definitions differ from checkpoint")
    output_configuration = {
        "schema": "asb-drx-v64-adaptive-production-v1",
        "physical_configuration": physical,
        "accuracy": {
            "state_relative_tolerance": args.state_tolerance,
            "rate_relative_tolerance": args.rate_tolerance,
            "integral_relative_tolerance": args.integral_tolerance,
            "minimum_step_s": args.minimum_step,
            "maximum_step_s": args.maximum_step,
            "accepted_solution": "two_complete_half_intervals",
            "metric_definitions": metric_definitions,
        },
    }
    accepted_at_start = controller.accepted_intervals
    run_started = time.perf_counter()
    while target_pending(endpoint.runtime.ledger.physical_time_s,
                         args.target_time):
        remaining = args.target_time-endpoint.runtime.ledger.physical_time_s
        interval_started = time.perf_counter()
        accepted = advance_error_controlled_interval(
            endpoint, controller,
            lambda parent, h: advance_complete_multigrain_interval(
                parent, h, context), metrics, requested_maximum_s=remaining)
        endpoint = accepted.interval.endpoint
        controller = accepted.controller
        errors = accepted.decision["attempts"][-1]["metric_errors"]
        maximum_ratio = max(
            row["normalized_error"]/row["relative_tolerance"]
            for row in errors.values())
        temperature = endpoint.observables["common_temperature_K"]
        stress = endpoint.observables["common_stress_tensor_Pa"]
        volumes = (np.sum(endpoint.state.supports, axis=(1, 2))
                   *context.spacing_m**2*context.represented_thickness_m)
        record = {
            "schema": "asb-drx-v64-adaptive-production-record-v1",
            "adaptive_interval": controller.accepted_intervals,
            "physical_time_s": endpoint.runtime.ledger.physical_time_s,
            "applied_shear_strain": endpoint.load,
            "accepted_duration_s": accepted.decision["accepted_duration_s"],
            "next_proposed_step_s": controller.proposed_step_s,
            "shear_stress_Pa": float(np.mean(stress[..., 0, 1])),
            "temperature_mean_K": float(np.mean(temperature)),
            "temperature_contrast_K": float(np.max(temperature)-np.min(temperature)),
            "grain_volume_m3": volumes.tolist(),
            "maximum_tolerance_ratio": maximum_ratio,
            "accuracy_decision": accepted.decision,
            "published_interval_audit": accepted.interval.audit,
            "runtime": asdict(endpoint.runtime.ledger),
            "wall_time_s": time.perf_counter()-interval_started,
        }
        history.append(record)
        history_path.write_text(json.dumps(history, indent=2)+"\n")
        relative_index = controller.accepted_intervals-accepted_at_start
        if (relative_index == 1 or relative_index % max(args.checkpoint_every, 1) == 0
                or not target_pending(endpoint.runtime.ledger.physical_time_s,
                                      args.target_time)):
            checkpoint = args.out/f"checkpoint_adaptive_{controller.accepted_intervals:06d}.npz"
            _save_checkpoint(
                checkpoint, endpoint.state, endpoint.runtime,
                controller.accepted_intervals, endpoint.load,
                initial_volume, output_configuration, provenance,
                diagnostic_arrays={
                    "diagnostic_adaptive_controller_json": np.asarray(
                        json.dumps(controller.to_dict(), sort_keys=True)),
                    "diagnostic_accuracy_decision_json": np.asarray(
                        json.dumps(accepted.decision, sort_keys=True)),
                    "diagnostic_front_joint_pressure_factor": np.asarray(
                        endpoint.numerical_hints.get(
                            "front_joint_pressure_factor", 1.0)),
                    "diagnostic_front_heat_source_J_by_cell": np.asarray(
                        accepted.interval.integrals["front_heat_J_by_cell"]),
                    "diagnostic_mechanical_heat_J_m3_cells": np.asarray(
                        accepted.interval.integrals["mechanical_heat_J_m3"]),
                    "diagnostic_mechanical_work_J_m3_cells": np.asarray(
                        accepted.interval.integrals["mechanical_work_J_m3"]),
                    "diagnostic_thermal_conduction_J_m3_cells": np.asarray(
                        accepted.interval.integrals["thermal_conduction_J_m3"]),
                    "diagnostic_thermal_bath_exchange_J_m3_cells": np.asarray(
                        accepted.interval.integrals["thermal_bath_exchange_J_m3"]),
                })
            record["checkpoint"] = str(checkpoint)
            record["checkpoint_sha256"] = digest(checkpoint)
            history_path.write_text(json.dumps(history, indent=2)+"\n")
        print(json.dumps(record, sort_keys=True), flush=True)
    energy = evaluate_complete_multigrain_energy(
        endpoint.state, spacing_m=context.spacing_m,
        represented_thickness_m=context.represented_thickness_m,
        wall_parameters=context.wall_parameters,
        **dict(context.energy_kwargs, mean_strain=np.array(
            [[0.0, .5*endpoint.load], [.5*endpoint.load, 0.0]])))
    checkpoint = Path(history[-1]["checkpoint"])
    result = {
        "schema": "asb-drx-v64-adaptive-production-result-v1",
        "execution_status": "TARGET_REACHED",
        "delivery_status": "COMPLETE_AT_ATTAINED_SCOPE",
        "scoped_experiment_complete": True,
        "full_scientific_objective_complete": False,
        "source_commit": head,
        "parent_checkpoint": str(args.resume.resolve()),
        "parent_checkpoint_sha256": args.expected_resume_sha256,
        "inherited_prefix_uncontrolled_by_v64": not analytic_origin_parent,
        "target_time_s": args.target_time,
        "final_time_s": endpoint.runtime.ledger.physical_time_s,
        "final_load": endpoint.load,
        "controller": controller.to_dict(),
        "wall_time_s": time.perf_counter()-run_started,
        "helmholtz_J": energy.helmholtz_J,
        "runtime": asdict(endpoint.runtime.ledger),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest(checkpoint),
        "physical_configuration": physical,
        "accuracy_configuration": output_configuration["accuracy"],
        "provenance": provenance,
    }
    (args.out/"result.json").write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
