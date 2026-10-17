#!/usr/bin/env python3
"""Price residence-time release of only a newly captured neutral cohort."""

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
from full_model.production.lattice_interface_absorption import (
    LatticeInterfaceAbsorptionParameters, evaluate_lattice_interface_absorption,
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
    strain = np.array([[0.0, .5*load], [.5*load, 0.0]])
    capture = evaluate_lattice_interface_absorption(
        state, mean_strain=strain, interval_s=config["dt_s"],
        spacing_m=config["length_m"]/config["n"],
        represented_thickness_m=thickness, systems=systems,
        wall_parameters=wall, energy_kwargs=energy,
        parameters=LatticeInterfaceAbsorptionParameters(
            maximum_fraction_per_interval=config["lattice_maximum_fraction"],
            capture_distance_m=config["lattice_capture_distance_m"],
            complete_physical_clock=True))
    captured_state = capture.committed_state
    eligible = []
    for before, after in zip(state.interfaces, captured_state.interfaces):
        plus = np.maximum(
            np.asarray(after.boundary_plus_m2)[..., 0, :]
            -np.asarray(before.boundary_plus_m2)[..., 0, :], 0.0)
        minus = np.maximum(
            np.asarray(after.boundary_minus_m2)[..., 0, :]
            -np.asarray(before.boundary_minus_m2)[..., 0, :], 0.0)
        eligible.append(np.minimum(plus, minus))
    eligible = np.asarray(eligible)
    eligible_sum = float(np.sum(eligible, dtype=np.longdouble))
    cell_area = (config["length_m"]/config["n"])**2
    captured_line_per_thickness = (
        capture.projection.removed_plus_line_per_thickness
        +capture.projection.removed_minus_line_per_thickness)
    maximum_neutral_recycle_fraction = (
        2.0*eligible_sum*cell_area
        /max(captured_line_per_thickness, np.finfo(float).tiny))
    trials = []
    accepted = None
    for maximum_fraction in (1.0, .5, .25, .125, .0625, .03125,
                             .015625, .0078125, .00390625):
        decision = evaluate_interface_mobile_pair_emission(
            captured_state, mean_strain=strain, interval_s=config["dt_s"],
            systems=systems, wall_parameters=wall,
            represented_thickness_m=thickness,
            source_parameters=ArrheniusSourceParameters(grain_size_m=2.4e-6),
            energy_kwargs=energy, kinetics_mode="stored_pair_release",
            eligible_pair_density_m2_by_interface=eligible,
            maximum_fraction_per_interval=maximum_fraction)
        trials.append({
            "maximum_reaction_coordinate": maximum_fraction,
            "accepted": decision.accepted,
            "classification": decision.classification,
            "emitted_pair_cell_sum_m2": (
                decision.projection.created_lattice_pair_cell_sum_m2),
            "eligible_pair_fraction_emitted": (
                decision.projection.created_lattice_pair_cell_sum_m2
                /max(eligible_sum, np.finfo(float).tiny)),
            "delta_helmholtz_before_heat_J": (
                decision.delta_helmholtz_before_heat_J),
            "relative_first_law_residual": (
                decision.relative_first_law_residual),
            "line_balance_residual_cell_sum_m2": (
                decision.projection.line_balance_residual_cell_sum_m2),
            "maximum_mura_relative_residual": (
                decision.projection.maximum_mura_relative_residual),
        })
        if decision.accepted and accepted is None:
            accepted = trials[-1]
    if not capture.accepted:
        classification = "PARENT_CAPTURE_NOT_ENERGY_QUALIFIED"
    elif eligible_sum <= 0.0:
        classification = "NO_NEW_NEUTRAL_CAPTURE_COHORT"
    elif maximum_neutral_recycle_fraction < .01:
        classification = "NEW_CAPTURE_COHORT_NEUTRAL_CAPACITY_INSUFFICIENT"
    elif trials[0]["accepted"]:
        classification = "NEW_CAPTURE_COHORT_RESIDENCE_RELEASE_ENERGY_ACCESSIBLE"
    elif accepted is not None:
        classification = "NEW_CAPTURE_COHORT_RELEASE_REQUIRES_PATH_RESOLUTION"
    else:
        classification = "NEW_CAPTURE_COHORT_RESIDENCE_RELEASE_REJECTED"
    return {
        "directory": str(root.resolve()),
        "checkpoint_sha256": _sha(checkpoint),
        "parent_provenance": provenance,
        "step": step,
        "load": load,
        "capture": {
            "accepted": capture.accepted,
            "classification": capture.classification,
            "removed_plus_line_per_thickness": (
                capture.projection.removed_plus_line_per_thickness),
            "removed_minus_line_per_thickness": (
                capture.projection.removed_minus_line_per_thickness),
            "relative_first_law_residual": capture.relative_first_law_residual,
        },
        "new_neutral_capture_pair_cell_sum_m2": eligible_sum,
        "maximum_neutral_recycled_fraction_of_captured_line": (
            maximum_neutral_recycle_fraction),
        "release_trials": trials,
        "largest_accepted_trial": accepted,
        "classification": classification,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    case = price(args.case)
    payload = {
        "schema": "v67-captured-pair-residence-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "case": case,
        "decision": {
            "classification": case["classification"],
            "production_promoted": False,
        },
        "claim_boundary": (
            "Only neutral pairs delivered by the immediately preceding "
            "accepted capture probe are eligible. Historical reservoir content "
            "cannot release. The factor-free rate is a residence-time hypothesis "
            "for an already encountered carrier, not a change to the immutable "
            "Hall--Petch nucleation law. Static energy accessibility is not a "
            "finite-time production result."),
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
