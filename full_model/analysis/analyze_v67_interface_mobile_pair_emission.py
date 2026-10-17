#!/usr/bin/env python3
"""Price default-off interface carrier emission on retained V67 states."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from full_model.analysis.run_v58_three_grain_production import _load_checkpoint
from full_model.analysis.run_v66_coupled_rx_asb import _physics
from full_model.production.arrhenius_dislocation_sources import (
    ArrheniusSourceParameters,
)
from full_model.production.interface_mobile_pair_emission import (
    evaluate_interface_mobile_pair_emission,
)
from full_model.production.interface_structural_absorption import (
    InterfaceStructuralAbsorptionParameters,
)
from full_model.production.structural_interface_network import (
    JunctionDisclinationParameters,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def price(root: Path) -> dict:
    checkpoint = root/"checkpoint_latest.npz"
    state, runtime, step, load, _, config, provenance = _load_checkpoint(
        checkpoint)
    args = SimpleNamespace(
        n=config["n"], length=config["length_m"],
        temperature=config["temperature_K"],
        shear_rate=config["shear_rate_s"], dt=config["dt_s"],
        thermal_control=config["thermal_control"],
        bath_rate=config["thermal_bath_rate_s"])
    _, thickness, systems, wall, _, energy, _ = _physics(args)
    wall = replace(
        wall, wall_nye_match_coefficient_J_m=wall.c44_Pa*(1.5e-7)**2)
    structural = InterfaceStructuralAbsorptionParameters(
        complete_physical_clock=True).boundary
    energy = dict(
        energy, structural_interfaces=tuple(runtime.structural_interfaces),
        structural_burgers_m=structural.burgers_m,
        structural_parameters=structural,
        junction_disclination_parameters=JunctionDisclinationParameters())
    decision = evaluate_interface_mobile_pair_emission(
        state, mean_strain=np.array([[0.0, .5*load], [.5*load, 0.0]]),
        interval_s=config["dt_s"], systems=systems, wall_parameters=wall,
        represented_thickness_m=thickness,
        source_parameters=ArrheniusSourceParameters(grain_size_m=2.4e-6),
        energy_kwargs=energy)
    projection = decision.projection
    neutral_capacity = float(sum(np.sum(np.minimum(
        np.asarray(component.boundary_plus_m2)[..., 0, :],
        np.asarray(component.boundary_minus_m2)[..., 0, :]),
        dtype=np.longdouble) for component in state.interfaces))
    return {
        "directory": str(root.resolve()),
        "checkpoint_sha256": _sha(checkpoint),
        "parent_provenance": provenance,
        "step": step, "load": load,
        "accepted": decision.accepted,
        "classification": decision.classification,
        "maximum_kinetic_fraction": decision.maximum_kinetic_fraction,
        "neutral_interface_pair_capacity_cell_sum_m2": neutral_capacity,
        "aggregate_emitted_reservoir_pair_fraction": (
            projection.removed_interface_pair_cell_sum_m2
            /max(neutral_capacity, np.finfo(float).tiny)),
        "removed_interface_pair_cell_sum_m2": (
            projection.removed_interface_pair_cell_sum_m2),
        "created_lattice_pair_cell_sum_m2": (
            projection.created_lattice_pair_cell_sum_m2),
        "line_balance_residual_cell_sum_m2": (
            projection.line_balance_residual_cell_sum_m2),
        "signed_content_change_cell_sum_m2": (
            projection.signed_content_change_cell_sum_m2),
        "maximum_mura_relative_residual": (
            projection.maximum_mura_relative_residual),
        "resolved_work_J": decision.resolved_work_J,
        "defect_boundary_storage_change_J": (
            decision.defect_boundary_storage_change_J),
        "delta_helmholtz_before_heat_J": (
            decision.delta_helmholtz_before_heat_J),
        "generated_heat_J": decision.generated_heat_J,
        "first_law_residual_J": decision.first_law_residual_J,
        "relative_first_law_residual": (
            decision.relative_first_law_residual),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("cases", nargs="+", type=Path)
    args = parser.parse_args()
    cases = {case.name: price(case) for case in args.cases}
    accepted = bool(cases and all(row["accepted"] for row in cases.values()))
    payload = {
        "schema": "v67-interface-mobile-pair-emission-price-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": (
            "The explicit neutral mobile-channel GB reservoir can re-emit "
            "sign-balanced carriers under the unchanged Hall--Petch hazard."),
        "cases": cases,
        "decision": {
            "all_retained_states_energy_accessible": accepted,
            "classification": (
                "ENERGY_ACCESSIBLE_INTERFACE_CARRIER_EMISSION"
                if accepted else "INTERFACE_CARRIER_EMISSION_NOT_QUALIFIED"),
            "production_promoted": False,
        },
        "claim_boundary": (
            "Atomic retained-state energy accessibility is not finite-time "
            "flow recovery. Production remains disabled until the exact-parent "
            "source-geography discriminator completes and a coupled fork passes."),
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
