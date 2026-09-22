#!/usr/bin/env python3
"""Term-resolved fixed-scale geometry/continuum representation audit."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from full_model.analysis.run_v24_mechanical_supply import build_case
from full_model.production.arrhenius_kinetics import ActivatedProcess
from full_model.production.extensive_wall import extensive_wall_energy_components_J_m3
from full_model.production.lattice_line_geometry import (
    apply_physical_reconstruction,
    empty_lattice_geometry,
    initialize_closed_swept_surface,
    propose_plaquette_sweep,
)
from full_model.production.tensorial_nye import nye_from_plastic_distortion
from full_model.production.v24_mechanical_wall import (
    V24MechanicalWallState,
    V43GeometryKinetics,
    _elastic_energy_sum_J_m3_cells,
    accepted_geometry_plaquette_transaction,
    mechanical_checkpoint_arrays,
    synchronize_common,
)


LENGTH_M = 3.2e-6
THICKNESS_M = 3.2e-6
REPRESENTATION_LENGTH_M = 4.0e-7
CHEMICAL_RESERVOIR_WORK_J_M3_CELLS_PER_EXTENT = 2.0e5


def digest_state(state):
    sha = hashlib.sha256()
    for name, value in sorted(mechanical_checkpoint_arrays(state).items()):
        array = np.ascontiguousarray(value)
        sha.update(name.encode()); sha.update(array.dtype.str.encode())
        sha.update(str(array.shape).encode()); sha.update(array.tobytes())
    return sha.hexdigest()


def energy_terms_J(state, data):
    _, driving, _, _, systems, topologies, _, extensive, _, dx = data
    zero = np.zeros(state.common.orientation_rad.shape+(3, 3))
    wall = extensive_wall_energy_components_J_m3(
        state.density, systems, topologies, state.common.orientation_rad,
        zero, extensive)
    volume = dx**2*THICKNESS_M
    terms = {name: float(np.sum(value, dtype=np.longdouble)*volume)
             for name, value in wall.items() if name != "total"}
    elastic_cells = _elastic_energy_sum_J_m3_cells(
        state.common, state.common.beta_p, driving, data[6])
    terms["recoverable_elastic"] = float(elastic_cells*volume)
    terms["total"] = sum(terms.values())
    return terms


def prepare_represented_block(n):
    data = build_case(n, length_m=LENGTH_M, periodic_nye_consistent=True)
    base, _, _, _, systems, topologies, _, _, _, dx = data
    geometry = empty_lattice_geometry(
        (n, n), len(systems), dx, section_thickness_m=THICKNESS_M)
    width = n//4; start = (n-width)//2
    swept = np.zeros_like(geometry.swept_quanta)
    swept[start:start+width, start:start+width, 0, 0] = 1.0
    initialized = initialize_closed_swept_surface(
        geometry, base.density, base.reservoir_alignment, base.common,
        systems, base.common.orientation_rad, swept,
        continuum_representation_length_m=REPRESENTATION_LENGTH_M)
    state = synchronize_common(V24MechanicalWallState(
        initialized[3], initialized[1], initialized[2], initialized[0]),
        topologies)
    return state, data, start, width


def represented_translation(n):
    state, data, start, width = prepare_represented_block(n)
    _, _, _, _, systems, topologies, _, extensive, _, dx = data
    before = energy_terms_J(state, data)
    initial = state
    displacement = 1e-8
    extent = displacement/dx
    for j in range(start, start+width):
        candidate = propose_plaquette_sweep(
            state.geometry, state.density, state.reservoir_alignment,
            state.common, systems, state.common.orientation_rad,
            (start+width, j), 0, 1, extent,
            continuum_representation_length_m=REPRESENTATION_LENGTH_M)
        state = synchronize_common(V24MechanicalWallState(
            candidate[3], candidate[1], candidate[2], candidate[0]),
            topologies)
    after = energy_terms_J(state, data)
    delta = {key: after[key]-before[key] for key in before}
    dbeta = state.common.beta_p-initial.common.beta_p
    dnye = nye_from_plastic_distortion(dbeta, dx)
    raw_dbeta = np.zeros_like(dbeta)
    # The exact zero mode is sufficient for the extensive exchange audit.
    volume = dx**2*THICKNESS_M
    line_before = float(np.sum(np.abs(initial.geometry.edge_x_quanta)
                               +np.abs(initial.geometry.edge_y_quanta))*dx)
    line_after = float(np.sum(np.abs(state.geometry.edge_x_quanta)
                              +np.abs(state.geometry.edge_y_quanta))*dx)
    return state, data, {
        "grid": n, "spacing_m": dx,
        "represented_section_thickness_m": THICKNESS_M,
        "continuum_representation_length_m": REPRESENTATION_LENGTH_M,
        "representation_cells": REPRESENTATION_LENGTH_M/dx,
        "physical_boundary_length_m": width*dx,
        "prescribed_displacement_m": displacement,
        "swept_area_m2": displacement*width*dx,
        "line_length_before_m": line_before,
        "line_length_after_m": line_after,
        "term_energy_before_J": before,
        "term_energy_after_J": after,
        "term_energy_change_J": delta,
        "plastic_increment_integral_m3": (
            np.sum(dbeta, axis=(0, 1))*volume).tolist(),
        "nye_increment_rms_m1": float(np.sqrt(np.mean(dnye*dnye))),
        "state_digest": digest_state(state),
    }


def adjoint_and_commutation_audit():
    rng = np.random.default_rng(46)
    shape = (32, 32, 3, 3)
    dx = LENGTH_M/32
    left = rng.normal(size=shape); right = rng.normal(size=shape)
    kl = apply_physical_reconstruction(
        left, dx, REPRESENTATION_LENGTH_M)
    kr = apply_physical_reconstruction(
        right, dx, REPRESENTATION_LENGTH_M)
    adjoint = abs(float(np.sum(kl*right)-np.sum(left*kr)))/max(
        abs(float(np.sum(kl*right))), abs(float(np.sum(left*kr))), 1.0)
    curl_left = nye_from_plastic_distortion(kl, dx)
    curl_right = apply_physical_reconstruction(
        nye_from_plastic_distortion(left, dx), dx,
        REPRESENTATION_LENGTH_M)
    commute = float(np.sqrt(np.mean((curl_left-curl_right)**2))/max(
        np.sqrt(np.mean(curl_right**2)), 1.0))
    return {"adjoint_relative_residual": adjoint,
            "curl_commutation_relative_rms": commute,
            "zero_mode_exact": bool(np.allclose(
                apply_physical_reconstruction(
                    np.ones((32, 32)), dx, REPRESENTATION_LENGTH_M),
                1.0, rtol=0.0, atol=4e-16))}


def transaction_sequence():
    state, data, start, width = prepare_represented_block(32)
    _, driving, _, _, systems, topologies, common, extensive, _, dx = data
    kinetics = V43GeometryKinetics(
        ActivatedProcess("v46-fixed-scale-geometry", 2e5),
        enthalpy_J=0.0, critical_stress_Pa=1e9,
        material_exchange_model="equilibrated_point_defect_reservoir",
        chemical_work_J_m3_cells_per_extent=(
            CHEMICAL_RESERVOIR_WORK_J_M3_CELLS_PER_EXTENT),
        continuum_representation_length_m=REPRESENTATION_LENGTH_M)
    favorable = {
        "cell": (start+width, start), "family": 0, "burgers_sign": 1,
        "proposed_extent": .2,
    }
    first, one = accepted_geometry_plaquette_transaction(
        state, favorable, systems, topologies, driving, common, extensive,
        kinetics, 1e-6)
    second_event = {**favorable, "cell": (start+width, start+1)}
    second, two = accepted_geometry_plaquette_transaction(
        first, second_event, systems, topologies, driving, common, extensive,
        kinetics, 1e-6)
    rejected_event = {**favorable, "cell": (0, 0)}
    blocked_kinetics = V43GeometryKinetics(
        ActivatedProcess("v46-fixed-scale-geometry-blocked", 2e5),
        enthalpy_J=0.0, critical_stress_Pa=1e9,
        material_exchange_model="equilibrated_point_defect_reservoir",
        chemical_work_J_m3_cells_per_extent=0.0,
        continuum_representation_length_m=REPRESENTATION_LENGTH_M)
    rejected, no = accepted_geometry_plaquette_transaction(
        second, rejected_event, systems, topologies, driving, common,
        extensive, blocked_kinetics, 1e-6)
    return {
        "clock_s": 2e-6,
        "predicted_extent_rate_s": 2e5,
        "predicted_boundary_speed_m_s": 2e5*dx,
        "external_work_J_m3_cells": 0.0,
        "chemical_reservoir_work_J_m3_cells_per_extent": (
            CHEMICAL_RESERVOIR_WORK_J_M3_CELLS_PER_EXTENT),
        "chemical_reservoir_scope": (
            "declared equilibrated point-defect reservoir; uncertain fixture "
            "parameter, not material calibration or artificial pressure"),
        "first": {key: one.get(key) for key in (
            "accepted", "classification", "consumed_duration_s",
            "committed_swept_area_m2", "complete_energy_change_J_m3_cells")},
        "second": {key: two.get(key) for key in (
            "accepted", "classification", "consumed_duration_s",
            "committed_swept_area_m2", "complete_energy_change_J_m3_cells")},
        "rejected": {key: no.get(key) for key in (
            "accepted", "classification", "consumed_duration_s",
            "committed_swept_area_m2", "complete_energy_change_J_m3_cells")},
        "rejection_exact_rollback": digest_state(rejected) == digest_state(second),
        "accepted_event_count": int(second.geometry.accepted_event_count),
    }


def main():
    rows = [represented_translation(n)[2] for n in (16, 32, 64)]
    keys = tuple(rows[0]["term_energy_change_J"])
    convergence = {}
    for key in keys:
        values = [row["term_energy_change_J"][key] for row in rows]
        convergence[key] = {
            "values_J": values,
            "n32_n64_relative_difference": abs(values[2]-values[1])/max(
                abs(values[2]), abs(values[1]), 1e-300),
        }
    stored_total_values = [row["term_energy_after_J"]["total"]
                           for row in rows]
    stored_total_finest_pair = abs(
        stored_total_values[2]-stored_total_values[1])/max(
            abs(stored_total_values[2]), abs(stored_total_values[1]), 1e-300)
    incremental_total_scale = max(
        abs(rows[2]["term_energy_before_J"]["total"]),
        abs(rows[1]["term_energy_before_J"]["total"]), 1e-300)
    incremental_total_absolute_discrepancy = abs(
        convergence["total"]["values_J"][2]
        -convergence["total"]["values_J"][1])
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip()
    payload = {
        "schema": "asb-drx/v46/geometry-representation/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": source,
        "architecture": "fixed_physical_positive_periodic_reconstruction",
        "physical_owner_partition": {
            "raw_geometry": "exact topology, swept surfaces, and line links",
            "continuum_map": "one K_ell map for scalar line, moment, beta_p, and Nye",
            "self_energy": "line/log/gradient continuum functional evaluated once on represented fields",
            "unresolved_extra_core_energy": "zero in this minimal architecture; no duplicate self term",
        },
        "lengths_m": {"representation": REPRESENTATION_LENGTH_M,
                      "section_thickness": THICKNESS_M,
                      "domain": LENGTH_M},
        "refinement": rows,
        "term_convergence": convergence,
        "energy_observable_qualification": {
            "stored_total_n32_n64_relative_difference": (
                stored_total_finest_pair),
            "incremental_total_n32_n64_relative_difference": (
                convergence["total"]["n32_n64_relative_difference"]),
            "incremental_total_n32_n64_absolute_discrepancy_J": (
                incremental_total_absolute_discrepancy),
            "incremental_discrepancy_over_stored_energy": (
                incremental_total_absolute_discrepancy
                /incremental_total_scale),
            "interpretation": (
                "stored-energy comparison qualifies; the nearly cancelling "
                "translation increment does not satisfy a relative 5 percent "
                "claim and remains separately unqualified"),
        },
        "variational_map": adjoint_and_commutation_audit(),
        "transaction_sequence": transaction_sequence(),
        "material_calibration_claimed": False,
        "spontaneous_wall_or_drx_claimed": False,
    }
    payload["classification"] = {
        "representation_resolved_on_finest_pair": bool(
            rows[1]["representation_cells"] >= 4.0
            and rows[2]["representation_cells"] >= 4.0),
        "stored_total_energy_grid_converged_5pct": bool(
            stored_total_finest_pair <= .05),
        "incremental_total_energy_grid_converged_5pct": bool(
            convergence["total"]["n32_n64_relative_difference"] <= .05),
        "nonzero_repeated_motion": bool(
            payload["transaction_sequence"]["first"]["accepted"]
            and payload["transaction_sequence"]["second"]["accepted"]),
        "atomic_rejection": bool(
            payload["transaction_sequence"]["rejection_exact_rollback"]),
    }
    output = Path("full_model/verification/v46_geometry_representation.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"classification": payload["classification"],
                      "sha256": hashlib.sha256(output.read_bytes()).hexdigest()},
                     sort_keys=True))


if __name__ == "__main__":
    main()
