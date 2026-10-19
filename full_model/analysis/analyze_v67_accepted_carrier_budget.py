#!/usr/bin/env python3
"""Reconcile raw flow, accepted flow, and carrier budgets on V67 states.

The audit deliberately evaluates spatial products on immutable checkpoints.
It does not infer Orowan flow from products of marginal mean density and mean
speed.  It also compares the historical reaction-limited interface capture to
the candidate serial arrival/reaction clock without changing either state.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from full_model.analysis.run_v58_three_grain_production import _load_checkpoint
from full_model.analysis.run_v66_coupled_rx_asb import _physics
from full_model.analysis.run_v67_interface_structural_bridge import (
    _args_from_configuration,
)
from full_model.production.common_tensorial_wall import (
    CommonWallDriving,
    resolved_driving_components,
    wall_residual,
)
from full_model.production.interface_structural_absorption import (
    InterfaceStructuralAbsorptionParameters,
)
from full_model.production.lattice_interface_absorption import (
    LatticeInterfaceAbsorptionParameters,
    evaluate_lattice_interface_absorption,
)
from full_model.production.multigrain_production import (
    _owner_drivings_from_common_stress,
    advance_multigrain_mechanics,
)
from full_model.production.structural_interface_network import (
    JunctionDisclinationParameters,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _physics_for_checkpoint(configuration):
    args = _args_from_configuration(configuration)
    args.shear_rate = float(configuration["shear_rate_s"])
    args.dt = float(configuration["dt_s"])
    args.thermal_control = configuration["thermal_control"]
    args.bath_rate = float(configuration["thermal_bath_rate_s"])
    spacing, thickness, systems, wall, _, energy, context = _physics(args)
    wall = replace(
        wall,
        wall_nye_match_coefficient_J_m=wall.c44_Pa*(1.5e-7)**2)
    if "multiplication_law" in configuration:
        wall = replace(
            wall,
            multiplication_law=configuration["multiplication_law"],
            multiplication_coefficient_m1=float(
                configuration["multiplication_coefficient_m1"]),
            multiplication_storage_work_fraction=float(configuration[
                "multiplication_storage_work_fraction"]))
    if configuration.get("temperature_dependent_exp_floor", False):
        wall = replace(
            wall,
            kinetic_reference_temperature_K=float(configuration[
                "kinetic_reference_temperature_K"]),
            glide_barrier_temperature_coefficient=float(configuration[
                "glide_barrier_temperature_coefficient"]),
            critical_stress_temperature_coefficient=float(configuration[
                "critical_stress_temperature_coefficient"]))
    structural = InterfaceStructuralAbsorptionParameters(
        complete_physical_clock=True).boundary
    return spacing, thickness, systems, wall, dict(
        energy,
        structural_burgers_m=structural.burgers_m,
        structural_parameters=structural,
        junction_disclination_parameters=JunctionDisclinationParameters()), context


def _line_inventories_m(state, spacing, thickness):
    area_thickness = float(spacing)**2*float(thickness)
    bulk = {}
    for prefix in ("mobile", "forest", "wall"):
        plus = sum(float(np.sum(
            np.asarray(support)[..., None]
            *np.asarray(getattr(owner, f"{prefix}_plus_m2")),
            dtype=np.longdouble))*area_thickness
            for support, owner in zip(state.supports, state.owners))
        minus = sum(float(np.sum(
            np.asarray(support)[..., None]
            *np.asarray(getattr(owner, f"{prefix}_minus_m2")),
            dtype=np.longdouble))*area_thickness
            for support, owner in zip(state.supports, state.owners))
        bulk[prefix] = {"plus_m": plus, "minus_m": minus,
                        "total_m": plus+minus}
    interface_plus = sum(float(np.sum(
        np.asarray(component.boundary_plus_m2), dtype=np.longdouble))
        *area_thickness for component in state.interfaces)
    interface_minus = sum(float(np.sum(
        np.asarray(component.boundary_minus_m2), dtype=np.longdouble))
        *area_thickness for component in state.interfaces)
    bulk["interface"] = {
        "plus_m": interface_plus, "minus_m": interface_minus,
        "total_m": interface_plus+interface_minus}
    return bulk


def _weighted_sum(value, support, spacing):
    weight = np.asarray(support)[..., None]
    return np.asarray(np.sum(
        weight*np.asarray(value), axis=(0, 1), dtype=np.longdouble)
        *float(spacing)**2, dtype=float)


def _capture_probe(state, runtime, load, configuration, *, arrival_limited):
    spacing, thickness, systems, wall, energy, _ = _physics_for_checkpoint(
        configuration)
    energy["structural_interfaces"] = tuple(runtime.structural_interfaces)
    mean_strain = np.array([[0.0, .5*load], [.5*load, 0.0]])
    decision = evaluate_lattice_interface_absorption(
        state, mean_strain=mean_strain,
        interval_s=float(configuration["dt_s"]), spacing_m=spacing,
        represented_thickness_m=thickness, systems=systems,
        wall_parameters=wall, energy_kwargs=energy,
        parameters=LatticeInterfaceAbsorptionParameters(
            maximum_fraction_per_interval=float(
                configuration["lattice_maximum_fraction"]),
            capture_distance_m=float(
                configuration["lattice_capture_distance_m"]),
            complete_physical_clock=True,
            arrival_limited=arrival_limited))
    removed = thickness*(
        decision.projection.removed_plus_line_per_thickness
        +decision.projection.removed_minus_line_per_thickness)
    return {
        "arrival_limited": arrival_limited,
        "accepted": decision.accepted,
        "classification": decision.classification,
        "removed_line_m": removed,
        "kinetic_fraction": decision.kinetic_fraction,
        "maximum_incident_capacity_fraction": (
            decision.projection.maximum_incident_capacity),
        "maximum_arrival_exposure": decision.maximum_arrival_exposure,
        "reaction_event_rate_s": decision.event_rate_s,
        "selected_hazard_scale": decision.selected_hazard_scale,
        "delta_helmholtz_J": decision.delta_helmholtz_J,
        "relative_first_law_residual": decision.relative_first_law_residual,
        "maximum_mura_relative_residual": (
            decision.projection.maximum_mura_relative_residual),
    }


def audit_checkpoint(path: Path) -> dict:
    state, runtime, step, load, _, configuration, provenance = _load_checkpoint(
        path)
    spacing, thickness, systems, wall, _, context = _physics_for_checkpoint(
        configuration)
    mean_strain = np.array([[0.0, .5*load], [.5*load, 0.0]])
    drivings = _owner_drivings_from_common_stress(
        state, CommonWallDriving(mean_strain=mean_strain), systems, wall)
    owner_rows = []
    raw_power_integral_W = 0.0
    for grain, support, owner, driving in zip(
            state.grain_ids, state.supports, state.owners, drivings):
        resolved = resolved_driving_components(
            owner, driving, systems, context.topologies, wall)
        residual = wall_residual(
            owner, driving, systems, context.topologies, wall)
        rho_plus = np.asarray(owner.mobile_plus_m2)
        rho_minus = np.asarray(owner.mobile_minus_m2)
        plus_slip = wall.burgers_m*resolved["speed_plus_m_s"]*rho_plus
        minus_slip = wall.burgers_m*resolved["speed_minus_m_s"]*rho_minus
        family_power = resolved["raw_stress_Pa"]*(plus_slip+minus_slip)
        raw_power_integral_W += float(np.sum(
            np.asarray(support)[..., None]*family_power,
            dtype=np.longdouble))*spacing**2*thickness
        owner_rows.append({
            "grain_id": int(grain),
            "mobile_plus_line_per_thickness_by_family": _weighted_sum(
                rho_plus, support, spacing).tolist(),
            "mobile_minus_line_per_thickness_by_family": _weighted_sum(
                rho_minus, support, spacing).tolist(),
            "absolute_speed_weighted_line_m2_s_by_family": _weighted_sum(
                np.abs(resolved["speed_m_s"])*(rho_plus+rho_minus),
                support, spacing).tolist(),
            "raw_signed_slip_area_rate_m2_s_by_family": _weighted_sum(
                plus_slip+minus_slip, support, spacing).tolist(),
            "raw_absolute_slip_area_rate_m2_s_by_family": _weighted_sum(
                np.abs(plus_slip)+np.abs(minus_slip), support, spacing).tolist(),
            "raw_power_W_by_family_per_thickness": (
                _weighted_sum(family_power, support, spacing)*thickness
            ).tolist(),
            "mean_raw_stress_Pa_by_family": np.mean(
                resolved["raw_stress_Pa"], axis=(0, 1)).tolist(),
            "mean_effective_stress_Pa_by_family": np.mean(
                resolved["effective_stress_Pa"], axis=(0, 1)).tolist(),
            "mean_taylor_resistance_Pa_by_family": np.mean(
                resolved["taylor_resistance_Pa"], axis=(0, 1)).tolist(),
            "mean_absolute_speed_m_s_by_family": np.mean(
                np.abs(resolved["speed_m_s"]), axis=(0, 1)).tolist(),
            "instantaneous_constitutive_scale": 1.0,
            "transport_mobile_plus_global_rate_m_per_s": float(np.sum(
                np.asarray(support)[..., None]
                *residual.channel_rates_m2_s[
                    "transport_mobile_plus_rate_m2_s"],
                dtype=np.longdouble)*spacing**2*thickness),
            "transport_mobile_minus_global_rate_m_per_s": float(np.sum(
                np.asarray(support)[..., None]
                *residual.channel_rates_m2_s[
                    "transport_mobile_minus_rate_m2_s"],
                dtype=np.longdouble)*spacing**2*thickness),
        })

    candidate, accepted = advance_multigrain_mechanics(
        state, driving=CommonWallDriving(mean_strain=mean_strain),
        systems=systems, topologies=context.topologies,
        wall_parameters=wall, dt_s=float(configuration["dt_s"]),
        represented_thickness_m=thickness,
        maximum_internal_substeps=4096)
    accepted_rows = []
    for grain, support, before, after, driving in zip(
            state.grain_ids, state.supports, state.owners, candidate.owners,
            drivings):
        slip_rate = (np.asarray(after.slip)-np.asarray(before.slip)) / float(
            configuration["dt_s"])
        resolved = resolved_driving_components(
            before, driving, systems, context.topologies, wall)
        accepted_rows.append({
            "grain_id": int(grain),
            "accepted_signed_slip_area_rate_m2_s_by_family": _weighted_sum(
                slip_rate, support, spacing).tolist(),
            "accepted_absolute_slip_area_rate_m2_s_by_family": _weighted_sum(
                np.abs(slip_rate), support, spacing).tolist(),
            "initial_stress_conjugate_power_W_by_family": (
                _weighted_sum(resolved["raw_stress_Pa"]*slip_rate,
                              support, spacing)*thickness).tolist(),
            "accepted_beta_p_increment_l2": float(np.linalg.norm(
                np.asarray(after.beta_p)-np.asarray(before.beta_p))),
        })
    domain_volume = float(configuration["length_m"])**2*thickness
    raw_mean_power = raw_power_integral_W/domain_volume
    accepted_mean_power = (
        accepted.external_plastic_work_J
        /(float(configuration["dt_s"])*domain_volume))
    capture_legacy = _capture_probe(
        state, runtime, load, configuration, arrival_limited=False)
    capture_arrival = _capture_probe(
        state, runtime, load, configuration, arrival_limited=True)
    return {
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": _sha(path),
        "source_provenance": provenance,
        "step": int(step), "load": float(load),
        "physical_time_s": float(runtime.ledger.physical_time_s),
        "configuration": {
            key: configuration[key] for key in (
                "dt_s", "shear_rate_s", "source_mode",
                "lattice_capture_distance_m", "lattice_maximum_fraction")},
        "line_inventory": _line_inventories_m(state, spacing, thickness),
        "owners_raw": owner_rows,
        "owners_accepted": accepted_rows,
        "flow_reconciliation": {
            "raw_instantaneous_mean_plastic_power_W_m3": raw_mean_power,
            "accepted_interval_mean_plastic_power_W_m3": accepted_mean_power,
            "accepted_to_raw_power_ratio": (
                accepted_mean_power/max(abs(raw_mean_power), 1e-300)),
            "minimum_internal_step_scale": accepted.minimum_step_scale,
            "mechanical_internal_substeps": accepted.maximum_substeps,
            "consumed_interval_s": accepted.consumed_interval_s,
            "maximum_line_balance_residual_m2_s": (
                accepted.maximum_line_balance_residual_m2_s),
        },
        "capture_probes": {
            "legacy_reaction_clock": capture_legacy,
            "arrival_limited_serial_clock": capture_arrival,
            "arrival_to_legacy_removed_line_ratio": (
                capture_arrival["removed_line_m"]
                /max(capture_legacy["removed_line_m"], 1e-300)),
        },
        "ledger": asdict(runtime.ledger),
    }


def _segment_budget(early, late):
    delta_mobile = (late["line_inventory"]["mobile"]["total_m"]
                    -early["line_inventory"]["mobile"]["total_m"])
    absorbed = (late["ledger"]["lattice_absorbed_line_length_m"]
                -early["ledger"]["lattice_absorbed_line_length_m"])
    # The source ledger is a cell-sum density. Infer its conversion from the
    # checkpoint inventory geometry already represented in the source state.
    early_path = Path(early["checkpoint"])
    _, _, _, _, _, config, _ = _load_checkpoint(early_path)
    spacing, thickness, *_ = _physics_for_checkpoint(config)
    source = ((late["ledger"]["source_created_line_cell_sum_m2"]
               -early["ledger"]["source_created_line_cell_sum_m2"])
              *spacing**2*thickness)
    other = delta_mobile-source+absorbed
    return {
        "from_load": early["load"], "to_load": late["load"],
        "delta_mobile_line_m": delta_mobile,
        "source_created_line_m": source,
        "lattice_interface_absorbed_line_m": absorbed,
        "residual_mechanical_reaction_transport_line_m": other,
        "closure_identity_m": delta_mobile-(source-absorbed+other),
        "source_replacement_fraction_of_mobile_loss": (
            source/max(-delta_mobile, 1e-300)),
        "absorption_fraction_of_mobile_loss": (
            absorbed/max(-delta_mobile, 1e-300)),
    }


def classify(cases, segments):
    early, middle, late = cases
    limiter_ratios = [
        row["flow_reconciliation"]["accepted_to_raw_power_ratio"]
        for row in cases]
    power_retention = (
        late["flow_reconciliation"][
            "raw_instantaneous_mean_plastic_power_W_m3"]
        /early["flow_reconciliation"][
            "raw_instantaneous_mean_plastic_power_W_m3"])
    mobile_retention = (
        late["line_inventory"]["mobile"]["total_m"]
        /early["line_inventory"]["mobile"]["total_m"])
    arrival_ratio = late["capture_probes"][
        "arrival_to_legacy_removed_line_ratio"]
    if min(limiter_ratios) < .5:
        return "NUMERICAL_ACCEPTANCE_LIMITER_SUPPRESSES_VALID_FLOW"
    if power_retention < .01 and mobile_retention > .05 and arrival_ratio < .9:
        return (
            "ACTIVE_FAMILY_CORRELATION_COLLAPSE_WITH_OVERCONSUMING_"
            "NON_ARRIVAL_LIMITED_CAPTURE")
    if power_retention < .01 and mobile_retention > .05:
        return "ACTIVE_FAMILY_STRESS_CARRIER_CORRELATION_COLLAPSE"
    return "NO_DECISIVE_ACCEPTED_FLOW_SEPARATION"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("checkpoints", nargs=3, type=Path)
    args = parser.parse_args()
    cases = [audit_checkpoint(path) for path in args.checkpoints]
    segments = [_segment_budget(a, b) for a, b in zip(cases[:-1], cases[1:])]
    payload = {
        "schema": "v67-accepted-carrier-budget-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "immutable_mechanics": (
            "Arrhenius EXP-floor Peierls-Taylor-Hall-Petch flow and source "
            "event measure unchanged"),
        "cases": cases,
        "segments": segments,
        "decision": {
            "classification": classify(cases, segments),
            "accepted_mechanical_clock_consumes_full_interval": all(
                abs(row["flow_reconciliation"]["consumed_interval_s"]
                    -row["configuration"]["dt_s"])
                <= 1e-12*row["configuration"]["dt_s"] for row in cases),
            "arrival_limited_candidate_changes_only_boundary_capture_clock": True,
        },
        "claim_boundary": (
            "The one-interval accepted mechanics probes use immutable copied "
            "states. Per-family accepted slip is the exact published state "
            "increment; per-family stress-conjugate power uses the initial "
            "common stress, while total accepted work is the authoritative "
            "substep-integrated value. Segment residuals group transport, "
            "lock/recovery, multiplication and annihilation not separately "
            "accumulated in the historical ledger."),
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
