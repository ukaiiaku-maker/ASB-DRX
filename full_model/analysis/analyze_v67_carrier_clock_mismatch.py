#!/usr/bin/env python3
"""Audit lattice capture and interface re-emission clocks on one V67 state.

The Hall--Petch law prescribes ``b**2/d**2`` as its available-site
multiplicity and a fixed increment per event.  This audit verifies the
production event measure and separately reports the factor-free existing-site
hypothesis without promoting that fast-release ablation.
"""

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
    ArrheniusSourceParameters, hall_petch_nucleation_hazard_Hz,
)
from full_model.production.common_tensorial_wall import (
    CommonWallDriving, resolved_driving_components,
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
from full_model.production.multigrain_production import (
    _owner_drivings_from_common_stress,
)
from full_model.production.structural_interface_network import (
    JunctionDisclinationParameters,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify_clock_audit(metrics: dict) -> str:
    """Return a deliberately narrow decision from independently reported clocks."""
    if not metrics["capture_accepted"]:
        return "CAPTURE_CLOCK_NOT_ENERGY_QUALIFIED"
    if not metrics["hall_petch_emission_accepted"]:
        return "HALL_PETCH_EMISSION_NOT_ENERGY_QUALIFIED"
    if metrics["hypothetical_existing_to_new_opportunity_rate_ratio"] < 1.0e6:
        return "SOURCE_OPPORTUNITY_MEASURE_NOT_RATE_SEPARATING"
    if metrics["neutral_interface_pair_to_mobile_inventory_ratio"] <= 1.0:
        return "INTERFACE_RESERVOIR_NOT_CARRIER_DOMINANT"
    if metrics["emitted_line_fraction_of_mobile_inventory_per_interval"] >= 1e-4:
        return "FIXED_EVENT_INTERFACE_SUPPLY_NOT_NEGLIGIBLE"
    return "ANALYTICAL_SITE_MULTIPLICITY_PRESERVED_FIXED_EVENT_SUPPLY_INSUFFICIENT"


def audit(root: Path) -> dict:
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
    mean_strain = np.array([[0.0, .5*load], [.5*load, 0.0]])
    source = ArrheniusSourceParameters(grain_size_m=2.4e-6)
    drivings = _owner_drivings_from_common_stress(
        state, CommonWallDriving(mean_strain=mean_strain), systems, wall)
    hazards = []
    for owner, driving in zip(state.owners, drivings):
        stress = resolved_driving_components(
            owner, driving, systems, (), wall)["raw_stress_Pa"]
        hazards.append(hall_petch_nucleation_hazard_Hz(
            stress, np.asarray(owner.temperature_K)[..., None], source))
    positive_hazard = np.concatenate([
        value[value > 0.0].ravel() for value in hazards])

    capture = evaluate_lattice_interface_absorption(
        state, mean_strain=mean_strain, interval_s=config["dt_s"],
        spacing_m=config["length_m"]/config["n"],
        represented_thickness_m=thickness, systems=systems,
        wall_parameters=wall, energy_kwargs=energy,
        parameters=LatticeInterfaceAbsorptionParameters(
            maximum_fraction_per_interval=config["lattice_maximum_fraction"],
            capture_distance_m=config["lattice_capture_distance_m"],
            complete_physical_clock=True))
    emission = evaluate_interface_mobile_pair_emission(
        state, mean_strain=mean_strain, interval_s=config["dt_s"],
        systems=systems, wall_parameters=wall,
        represented_thickness_m=thickness, source_parameters=source,
        energy_kwargs=energy)

    mobile = float(sum(np.sum(
        np.asarray(support)[..., None]
        *(np.asarray(owner.mobile_plus_m2)+np.asarray(owner.mobile_minus_m2)),
        dtype=np.longdouble)
        for support, owner in zip(state.supports, state.owners)))
    neutral_pairs = float(sum(np.sum(np.minimum(
        np.asarray(component.boundary_plus_m2)[..., 0, :],
        np.asarray(component.boundary_minus_m2)[..., 0, :]),
        dtype=np.longdouble) for component in state.interfaces))
    multiplicity = source.burgers_m**2/source.grain_size_m**2
    metrics = {
        "capture_accepted": capture.accepted,
        "capture_classification": capture.classification,
        "capture_weighted_event_rate_s": capture.event_rate_s,
        "capture_maximum_uncapped_fraction": capture.maximum_uncapped_fraction,
        "capture_selected_hazard_scale": capture.selected_hazard_scale,
        "capture_local_kinetics_mode": capture.local_kinetics_mode,
        "capture_relative_first_law_residual": capture.relative_first_law_residual,
        "hall_petch_emission_accepted": emission.accepted,
        "hall_petch_emission_classification": emission.classification,
        "hall_petch_emission_maximum_local_capacity_fraction": (
            emission.maximum_kinetic_fraction),
        "hall_petch_aggregate_emitted_reservoir_pair_fraction": (
            emission.projection.created_lattice_pair_cell_sum_m2
            /max(neutral_pairs, np.finfo(float).tiny)),
        "emitted_line_fraction_of_mobile_inventory_per_interval": (
            2.0*emission.projection.created_lattice_pair_cell_sum_m2
            /max(mobile, np.finfo(float).tiny)),
        "hall_petch_emission_relative_first_law_residual": (
            emission.relative_first_law_residual),
        "hall_petch_hazard_minimum_positive_s": float(np.min(positive_hazard)),
        "hall_petch_hazard_median_s": float(np.median(positive_hazard)),
        "hall_petch_hazard_maximum_s": float(np.max(positive_hazard)),
        "new_source_site_multiplicity_b2_over_d2": multiplicity,
        "hypothetical_existing_to_new_opportunity_rate_ratio": 1.0/multiplicity,
        "hypothetical_existing_pair_hazard_maximum_s": (
            float(np.max(positive_hazard))/multiplicity),
        "mobile_line_cell_sum_m2": mobile,
        "neutral_interface_pair_cell_sum_m2": neutral_pairs,
        "neutral_interface_pair_to_mobile_inventory_ratio": (
            neutral_pairs/max(mobile, np.finfo(float).tiny)),
    }
    return {
        "directory": str(root.resolve()),
        "checkpoint_sha256": _sha(checkpoint),
        "parent_provenance": provenance,
        "step": step,
        "load": load,
        "physical_time_s": runtime.ledger.physical_time_s,
        "metrics": metrics,
        "classification": classify_clock_audit(metrics),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    result = audit(args.case)
    payload = {
        "schema": "v67-carrier-clock-mismatch-audit-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "immutable_mechanics": (
            "EXP-floor Arrhenius Peierls-Taylor-Hall-Petch mechanics retained"),
        "case": result,
        "decision": {
            "classification": result["classification"],
            "production_change_authorized": False,
        },
        "claim_boundary": (
            "Azizi--Dillon defines b^2/d^2 as the nucleation-site multiplicity "
            "and a fixed strain increment per event. The production emission "
            "therefore retains that factor and debits one fixed pair increment "
            "per event, capacity-limited by the stored reservoir. The reported "
            "factor-free hazard is a quarantined hypothetical existing-site "
            "ablation, not a production clock or finite-time flow recovery."),
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
