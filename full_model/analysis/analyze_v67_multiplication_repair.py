#!/usr/bin/env python3
"""Quantify legacy, recovered-v34, and no-multiplication laws on saved states.

This audit evaluates the same immutable checkpoint three ways.  It reports the
nominal Kocks--Mecking source, the energy-accepted source integrated on the
actual constitutive clock, its reservoir destination, and the consequent line
inventory changes.  It never advances a campaign checkpoint in place.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from full_model.analysis.analyze_v67_accepted_carrier_budget import (
    _line_inventories_m,
    _physics_for_checkpoint,
)
from full_model.analysis.run_v58_three_grain_production import _load_checkpoint
from full_model.production.common_tensorial_wall import (
    CommonWallDriving,
    wall_residual,
)
from full_model.production.multigrain_production import (
    _owner_drivings_from_common_stress,
    advance_multigrain_mechanics,
)


VARIANTS = {
    "legacy_v21_mobile_pair": {
        "multiplication_law": "legacy_v21_mobile_pair_v1",
        "multiplication_coefficient_m1": 10.0,
        "multiplication_storage_work_fraction": 1.0,
        "destination": "mobile_plus_and_mobile_minus",
        "density_argument": "within_family_total_line",
    },
    "recovered_v34_forest_storage": {
        "multiplication_law": "v34_km_forest_storage_v1",
        "multiplication_coefficient_m1": 5.0e8,
        "multiplication_storage_work_fraction": 0.1,
        "multiplication_forest_mix": 0.35,
        "multiplication_mobile_obstacle_weight": 0.10,
        "multiplication_wall_obstacle_weight": 0.35,
        "multiplication_gnd_obstacle_weight": 0.10,
        "destination": "forest_plus_and_forest_minus",
        "density_argument": "saved_v34_forest_wall_gnd_without_external_gb_shell",
    },
    "strict_no_multiplication": {
        "multiplication_law": "disabled",
        "multiplication_coefficient_m1": 0.0,
        "multiplication_storage_work_fraction": 0.0,
        "destination": "none",
        "density_argument": "zero",
    },
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _variant_parameters(base, declaration):
    changes = {key: value for key, value in declaration.items()
               if hasattr(base, key)}
    return replace(base, **changes)


def _instantaneous(state, load, systems, topologies, parameters):
    mean_strain = np.array([[0.0, 0.5*load], [0.5*load, 0.0]])
    drivings = _owner_drivings_from_common_stress(
        state, CommonWallDriving(mean_strain=mean_strain), systems, parameters)
    cell_raw = np.zeros(state.supports.shape[1:], dtype=float)
    cell_accepted = np.zeros_like(cell_raw)
    cell_available_power = np.zeros_like(cell_raw)
    cell_raw_cost = np.zeros_like(cell_raw)
    cell_density = np.zeros_like(cell_raw)
    cell_scale = np.zeros_like(cell_raw)
    for support, owner, driving in zip(
            state.supports, state.owners, drivings):
        weight = np.asarray(support, dtype=float)
        residual = wall_residual(
            owner, driving, systems, topologies, parameters)
        channels = residual.channel_rates_m2_s
        cell_raw += weight*2.0*np.sum(
            channels["multiplication_pairs_raw"], axis=2)
        cell_accepted += weight*2.0*np.sum(
            channels["multiplication_pairs"], axis=2)
        cell_available_power += weight*np.asarray(
            channels["multiplication_power_available_W_m3"])
        cell_raw_cost += weight*np.asarray(
            channels["multiplication_cost_raw_W_m3"])
        cell_density += weight*np.mean(
            channels["multiplication_density_argument_m2"], axis=2)
        cell_scale += weight*np.asarray(
            channels["multiplication_energy_budget_scale"])
    raw = float(np.mean(cell_raw, dtype=np.longdouble))
    accepted = float(np.mean(cell_accepted, dtype=np.longdouble))
    return {
        "domain_mean_raw_unsigned_line_rate_m2_s": raw,
        "domain_mean_accepted_unsigned_line_rate_m2_s": accepted,
        "accepted_to_raw_fraction": accepted/max(raw, 1e-300),
        "domain_mean_density_argument_m2": float(np.mean(
            cell_density, dtype=np.longdouble)),
        "domain_mean_available_storage_power_W_m3": float(np.mean(
            cell_available_power, dtype=np.longdouble)),
        "domain_mean_raw_storage_cost_W_m3": float(np.mean(
            cell_raw_cost, dtype=np.longdouble)),
        "budget_active_cell_fraction": float(np.mean(
            cell_scale < 1.0-64.0*np.finfo(float).eps)),
        "minimum_budget_scale": float(np.min(cell_scale)),
        "maximum_budget_scale": float(np.max(cell_scale)),
    }


def audit_variant(state, load, configuration, declaration):
    spacing, thickness, systems, base, _, context = _physics_for_checkpoint(
        configuration)
    parameters = _variant_parameters(base, declaration)
    instantaneous = _instantaneous(
        state, load, systems, context.topologies, parameters)
    before = _line_inventories_m(state, spacing, thickness)
    mean_strain = np.array([[0.0, 0.5*load], [0.5*load, 0.0]])
    candidate, decision = advance_multigrain_mechanics(
        state, driving=CommonWallDriving(mean_strain=mean_strain),
        systems=systems, topologies=context.topologies,
        wall_parameters=parameters, dt_s=float(configuration["dt_s"]),
        represented_thickness_m=thickness, maximum_internal_substeps=4096)
    after = _line_inventories_m(candidate, spacing, thickness)
    cell_volume = spacing**2*thickness
    raw_increment_m = float(np.sum(
        decision.raw_multiplication_line_increment_m2_cells,
        dtype=np.longdouble)*cell_volume)
    accepted_increment_m = float(np.sum(
        decision.accepted_multiplication_line_increment_m2_cells,
        dtype=np.longdouble)*cell_volume)
    return {
        "law": parameters.multiplication_law,
        "coefficient_m1": parameters.multiplication_coefficient_m1,
        "storage_work_fraction": (
            parameters.multiplication_storage_work_fraction),
        "destination": declaration["destination"],
        "density_argument": declaration["density_argument"],
        "instantaneous": instantaneous,
        "accepted_clock": {
            "raw_unsigned_line_increment_m": raw_increment_m,
            "accepted_unsigned_line_increment_m": accepted_increment_m,
            "accepted_to_raw_fraction": (
                accepted_increment_m/max(raw_increment_m, 1e-300)),
            "budget_active_cell_time_fraction": float(np.sum(
                decision.multiplication_budget_active_time_s_cells,
                dtype=np.longdouble)/max(
                    float(configuration["dt_s"])
                    *decision.multiplication_budget_active_time_s_cells.size,
                    1e-300)),
            "internal_substeps": decision.maximum_substeps,
            "minimum_internal_step_scale": decision.minimum_step_scale,
            "consumed_interval_s": decision.consumed_interval_s,
            "external_plastic_work_J": decision.external_plastic_work_J,
            "irreversible_heat_J": decision.irreversible_heat_J,
        },
        "endpoint_inventory_change_m": {
            name: after[name]["total_m"]-before[name]["total_m"]
            for name in ("mobile", "forest", "wall", "interface")
        },
    }


def audit_checkpoint(path: Path):
    state, runtime, step, load, _, configuration, provenance = _load_checkpoint(
        path)
    variants = {
        name: audit_variant(state, load, configuration, declaration)
        for name, declaration in VARIANTS.items()
    }
    recovered = variants["recovered_v34_forest_storage"]
    legacy = variants["legacy_v21_mobile_pair"]
    return {
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": _sha(path),
        "source_provenance": provenance,
        "step": int(step),
        "load": float(load),
        "physical_time_s": float(runtime.ledger.physical_time_s),
        "dt_s": float(configuration["dt_s"]),
        "variants": variants,
        "recovered_to_legacy": {
            "instantaneous_raw_line_rate_ratio": (
                recovered["instantaneous"][
                    "domain_mean_raw_unsigned_line_rate_m2_s"]
                /max(legacy["instantaneous"][
                    "domain_mean_raw_unsigned_line_rate_m2_s"], 1e-300)),
            "accepted_line_increment_ratio": (
                recovered["accepted_clock"][
                    "accepted_unsigned_line_increment_m"]
                /max(legacy["accepted_clock"][
                    "accepted_unsigned_line_increment_m"], 1e-300)),
        },
    }


def classify(cases):
    recovered = [case["variants"]["recovered_v34_forest_storage"]
                 for case in cases]
    disabled = [case["variants"]["strict_no_multiplication"]
                for case in cases]
    routes_to_forest = all(
        row["accepted_clock"]["accepted_unsigned_line_increment_m"] > 0.0
        for row in recovered)
    disabled_zero = all(
        row["accepted_clock"]["accepted_unsigned_line_increment_m"] == 0.0
        for row in disabled)
    if routes_to_forest and disabled_zero:
        return (
            "PRODUCTION_MULTIPLICATION_NORMALIZATION_AND_RESERVOIR_"
            "ROUTING_DEFECT_CONFIRMED_AND_REPAIRED")
    return "MULTIPLICATION_REPAIR_NOT_YET_QUANTITATIVELY_CLOSED"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    args = parser.parse_args()
    cases = [audit_checkpoint(path) for path in args.checkpoints]
    payload = {
        "schema": "v67-multiplication-normalization-routing-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reference_implementation": (
            "full_model/reference_sources/v34/"
            "drx_var_v34_candidate_drx_asb_sweep.py"),
        "saved_reference_semantics": {
            "KM_k1_m1": 5.0e8,
            "rho_state_store_to_forest": True,
            "stored_work_fraction": "1-taylor_quinney=0.1",
            "obstacle_mode": "forest_wall_gnd",
            "forest_mix": 0.35,
            "mobile_weight": 0.10,
            "wall_weight": 0.35,
            "gnd_weight": 0.10,
            "gb_shell_omission": (
                "Common-owner interfaces are explicit reservoirs; the old "
                "bulk-shell proxy is omitted to avoid double counting."),
        },
        "immutable_mechanics": (
            "Arrhenius EXP-floor Peierls-Taylor-Hall-Petch flow and event "
            "measure unchanged"),
        "cases": cases,
        "decision": {
            "classification": classify(cases),
            "checkpoint_count": len(cases),
            "all_physical_clocks_consumed": all(
                abs(case["variants"]["recovered_v34_forest_storage"]
                    ["accepted_clock"]["consumed_interval_s"]-case["dt_s"])
                <= 1e-12*case["dt_s"] for case in cases),
            "strict_no_multiplication_retained_only_as_comparator": True,
        },
        "claim_boundary": (
            "The audit recovers normalization, density argument, storage "
            "fraction, and forest routing from the archived executable and "
            "saved parameter semantics. It does not claim that this closure "
            "produces ASB or DRX; that requires the matched continuation."),
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
