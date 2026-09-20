#!/usr/bin/env python3
"""Cancellation-aware geometry affinity and physical reservoir audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from full_model.analysis.run_v46_geometry_representation import (
    LENGTH_M, REPRESENTATION_LENGTH_M, THICKNESS_M, energy_terms_J,
    prepare_represented_block, represented_translation,
)
from full_model.production.arrhenius_kinetics import ActivatedProcess
from full_model.production.extensive_wall import (
    extensive_wall_energy_components_J_m3,
    ordered_gradient_increment_J_m3_cells,
)
from full_model.production.lattice_line_geometry import propose_plaquette_sweep
from full_model.production.v24_mechanical_wall import (
    V24MechanicalWallState, V43GeometryKinetics,
    accepted_geometry_plaquette_transaction, synchronize_common,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def propose(state, data, cell, extent):
    systems, topologies = data[4], data[5]
    candidate = propose_plaquette_sweep(
        state.geometry, state.density, state.reservoir_alignment,
        state.common, systems, state.common.orientation_rad,
        cell, 0, 1, extent,
        continuum_representation_length_m=REPRESENTATION_LENGTH_M)
    return synchronize_common(V24MechanicalWallState(
        candidate[3], candidate[1], candidate[2], candidate[0]), topologies)


def stable_event(state, candidate, data, extent):
    systems, topologies, extensive, dx = data[4], data[5], data[7], data[9]
    zero = np.zeros(state.common.orientation_rad.shape+(3, 3))
    before = extensive_wall_energy_components_J_m3(
        state.density, systems, topologies, state.common.orientation_rad,
        zero, extensive)
    after = extensive_wall_energy_components_J_m3(
        candidate.density, systems, topologies, candidate.common.orientation_rad,
        zero, extensive)
    endpoint = {name: float(np.sum(after[name]-before[name],
                                    dtype=np.longdouble))
                for name in before if name != "total"}
    direct_gradient = ordered_gradient_increment_J_m3_cells(
        state.density, candidate.density, extensive)
    stable = dict(endpoint); stable["ordered_gradient"] = direct_gradient
    volume = dx*dx*THICKNESS_M
    endpoint_total_J = float(np.sum(after["total"]-before["total"],
                                    dtype=np.longdouble)*volume)
    stable_total_J = sum(stable.values())*volume
    displacement = extent*dx
    return {
        "extent": extent, "displacement_m": displacement,
        "endpoint_component_changes_J_m3_cells": endpoint,
        "direct_gradient_change_J_m3_cells": direct_gradient,
        "gradient_endpoint_cancellation_residual_J_m3_cells": (
            direct_gradient-endpoint["ordered_gradient"]),
        "endpoint_total_change_J": endpoint_total_J,
        "stable_complete_change_J": stable_total_J,
        "directional_force_N": -stable_total_J/displacement,
    }


def rigid_translation(state, data, start, width):
    before = energy_terms_J(state, data); initial = state
    for j in range(start, start+width):
        state = propose(state, data, (start, j), -1.0)
        state = propose(state, data, (start+width, j), 1.0)
    after = energy_terms_J(state, data)
    scale = max(abs(before["total"]), abs(after["total"]), 1e-300)
    density_error = float(np.linalg.norm(
        state.density.wall_ordered_plus_m2
        -np.roll(initial.density.wall_ordered_plus_m2, 1, axis=0))
        /max(np.linalg.norm(initial.density.wall_ordered_plus_m2), 1e-300))
    return {
        "displacement_m": data[9],
        "energy_change_J": after["total"]-before["total"],
        "energy_relative_to_stored": (after["total"]-before["total"])/scale,
        "translated_ordered_density_relative_l2": density_error,
        "a_priori_absolute_tolerance_J": 256*np.finfo(float).eps*scale,
        "zero_self_force_passed": bool(
            abs(after["total"]-before["total"])
            <= 256*np.finfo(float).eps*scale),
    }


def reservoir_controls(state, data, cell):
    systems, topologies, common, extensive = data[4], data[5], data[6], data[7]
    rows = {}
    for label, mu in (("negative_mu", -1e-20), ("zero_mu", 0.0),
                      ("positive_mu", 1e-20)):
        kinetics = V43GeometryKinetics(
            ActivatedProcess("v47-physical-vacancy-reservoir", 2e5),
            enthalpy_J=0.0, critical_stress_Pa=1e9,
            material_exchange_model="equilibrated_point_defect_reservoir",
            chemical_species="vacancy", chemical_potential_J_per_defect=mu,
            atomic_volume_m3_per_atom=1.8e-29,
            exchange_stoichiometry_defects_per_atom=1.0,
            continuum_representation_length_m=REPRESENTATION_LENGTH_M)
        _, ledger = accepted_geometry_plaquette_transaction(
            state, {"cell": cell, "family": 0, "burgers_sign": 1,
                    "proposed_extent": .1},
            systems, topologies, data[1], common, extensive, kinetics, 1e-6)
        rows[label] = {
            "accepted": bool(ledger["accepted"]),
            "classification": ledger["classification"],
            "event_rate_s": float(ledger["event_rate_s"]),
            "signed_material_exchange_volume_m3": float(
                ledger["signed_material_exchange_volume_m3"]),
            "signed_material_exchange_count": float(
                ledger["signed_material_exchange_count"]),
            "chemical_reservoir_work_J": float(
                ledger["chemical_reservoir_work_J"]),
            "chemical_reservoir_work_J_m3_cells": float(
                ledger["chemical_reservoir_work_J_m3_cells"]),
            "complete_energy_change_J_m3_cells": float(
                ledger["complete_energy_change_J_m3_cells"]),
            "chemical_work_mode": ledger["chemical_work_mode"],
        }
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n128-row", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [represented_translation(n)[2] for n in (32, 64)]
    rows.append(json.loads(args.n128_row.read_text()))
    state, data, start, width = prepare_represented_block(64)
    cell = (start+width, start)
    directional = []
    for extent in (.025, .05, .1, .2):
        directional.append(stable_event(
            state, propose(state, data, cell, extent), data, extent))
    removal_cell = (start, start)
    reverse = stable_event(
        state, propose(state, data, removal_cell, -.1), data, -.1)
    force_values = [row["directional_force_N"] for row in directional]
    force_spread = (max(force_values)-min(force_values))/max(
        max(abs(x) for x in force_values), 1e-300)
    payload = {
        "schema": "asb-drx/v47/geometry-force/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "actual_configuration": {
            "domain_m": LENGTH_M, "section_thickness_m": THICKNESS_M,
            "representation_length_m": REPRESENTATION_LENGTH_M,
            "v46_reported_1um_was_attribution_error": True,
        },
        "production_dependency_diff_geometry_artifact_to_scientific_source": (
            "no changes under full_model/production"),
        "fixed_scale_refinement": rows,
        "n64_directional_force": directional,
        "n64_removal_direction": reverse,
        "n64_directional_force_fractional_extent_spread": force_spread,
        "n64_complete_loop_rigid_translation": rigid_translation(
            state, data, start, width),
        "n64_physical_chemical_controls": reservoir_controls(state, data, cell),
        "rate_limitation": (
            "current EXP-floor proposal rate uses resolved stress and is "
            "independent of the complete geometry affinity; energy currently "
            "acts as an atomic accept/reject guard"),
        "force_qualified": bool(force_spread <= .05),
        "rate_qualified": False,
    }
    payload["classification"] = (
        "GEOMETRY_FORCE_QUALIFIED_RATE_NOT_QUALIFIED" if payload["force_qualified"]
        else "GEOMETRY_FORCE_AND_RATE_NOT_QUALIFIED")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"classification": payload["classification"],
                      "force_spread": force_spread,
                      "sha256": digest(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
