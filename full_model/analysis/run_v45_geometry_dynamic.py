#!/usr/bin/env python3
"""Dynamic physical-measure and blocked-channel checks for V45 geometry."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from full_model.analysis.run_v43_geometry_verification import prepare_block
from full_model.production.arrhenius_kinetics import ActivatedProcess
from full_model.production.extensive_wall import extensive_wall_energy_components_J_m3
from full_model.production.lattice_line_geometry import propose_plaquette_sweep
from full_model.production.v24_mechanical_wall import (
    V24MechanicalWallState, V43GeometryKinetics,
    _elastic_energy_sum_J_m3_cells, accepted_geometry_plaquette_transaction,
    accepted_v24_mechanical_step, mechanical_checkpoint_arrays,
    synchronize_common,
)


def digest_state(state):
    sha = hashlib.sha256()
    for name, value in sorted(mechanical_checkpoint_arrays(state).items()):
        array = np.ascontiguousarray(value)
        sha.update(name.encode()); sha.update(array.dtype.str.encode())
        sha.update(str(array.shape).encode()); sha.update(array.tobytes())
    return sha.hexdigest()


def energy_J(state, data):
    _, driving, _, _, systems, topologies, common, extensive, _, dx = data
    zero = np.zeros(state.common.orientation_rad.shape+(3, 3))
    wall = extensive_wall_energy_components_J_m3(
        state.density, systems, topologies, state.common.orientation_rad,
        zero, extensive)["total"]
    elastic = _elastic_energy_sum_J_m3_cells(
        state.common, state.common.beta_p, driving, common)
    volume = dx**2*float(state.geometry.section_thickness_m)
    return float((np.sum(wall, dtype=np.longdouble)+elastic)*volume)


def scripted_path(n, reverse=False, split=False):
    width = n//4; start = (n-width)//2
    cells = [(i, j) for i in range(start, start+width)
             for j in range(start, start+width)]
    state, data, _ = prepare_block(n, cells)
    _, _, _, _, systems, topologies, _, extensive, _, dx = data
    initial = state
    initial_energy = energy_J(initial, data)
    initial_line_length = float(np.sum(
        np.abs(initial.geometry.edge_x_quanta)
        +np.abs(initial.geometry.edge_y_quanta))*dx)
    displacement_m = 1e-8
    elapsed_s = 1e-6
    extent = displacement_m/dx
    sites = [(start+width, j) for j in range(start, start+width)]
    if reverse:
        sites = list(reversed(sites))
    pieces = (0.5*extent, 0.5*extent) if split else (extent,)
    swept = 0.0
    for cell in sites:
        for piece in pieces:
            candidate = propose_plaquette_sweep(
                state.geometry, state.density, state.reservoir_alignment,
                state.common, systems, state.common.orientation_rad,
                cell, 0, 1, piece)
            swept += abs(float(candidate[-1]["swept_area_m2"]))
            state = synchronize_common(V24MechanicalWallState(
                candidate[3], candidate[1], candidate[2], candidate[0]),
                topologies)
    dbeta = state.common.beta_p-initial.common.beta_p
    cell_volume = dx**2*float(state.geometry.section_thickness_m)
    line_length = float(np.sum(
        np.abs(state.geometry.edge_x_quanta)
        +np.abs(state.geometry.edge_y_quanta))*dx)
    return state, {
        "grid": n, "spacing_m": dx,
        "physical_boundary_length_m": width*dx,
        "prescribed_displacement_m": displacement_m,
        "elapsed_time_s": elapsed_s,
        "physical_velocity_m_s": displacement_m/elapsed_s,
        "eligible_site_count": len(sites),
        "site_population_interpretation": (
            "one numerical patch represents boundary line dx; fractional "
            "weight q=displacement/dx and all eligible patches evolve "
            "concurrently over one shared deterministic clock"),
        "per_site_fractional_extent": extent,
        "per_site_fractional_rate_s": extent/elapsed_s,
        "integrated_swept_area_m2": swept,
        "expected_swept_area_m2": displacement_m*width*dx,
        "plastic_distortion_increment_rms": float(np.sqrt(np.mean(dbeta**2))),
        "plastic_distortion_increment_volume_integral_m3": (
            np.sum(dbeta, axis=(0, 1))*cell_volume).tolist(),
        "signed_material_exchange_volume_m3": float(
            np.sum(np.trace(dbeta, axis1=-2, axis2=-1))*cell_volume),
        "signed_exchange_convention": (
            "positive trace(dbeta_p) is positive represented plastic volume "
            "exchange with the equilibrated point-defect reservoir"),
        "chemical_reference": (
            "zero point-defect reservoir work; spatially equilibrated "
            "reservoir; no vacancy transient"),
        "weighted_line_length_m": line_length,
        "geometric_line_energy_J": extensive.line_energy_J_m*line_length,
        "geometric_line_energy_change_J": (
            extensive.line_energy_J_m*(line_length-initial_line_length)),
        "mechanical_plus_line_energy_J": energy_J(state, data),
        "mechanical_plus_line_energy_change_J": energy_J(state, data)-initial_energy,
        "accepted_motion": True,
        "enumeration": "reverse" if reverse else "forward",
        "patch_subdivision": 2 if split else 1,
        "state_digest": digest_state(state),
    }


def blocked_channel_continuation():
    state, data, _ = prepare_block(16, [(7, 7)])
    _, driving, _, support, systems, topologies, common, extensive, oldk, _ = data
    kinetics = V43GeometryKinetics(
        ActivatedProcess("v45-clock-fixture", 1e12),
        enthalpy_J=0.0, critical_stress_Pa=1e9)
    rejected_event = {
        "cell": (8, 7), "family": 0, "burgers_sign": 1,
        "proposed_extent": 0.25,
        "external_work_J_m3_cells": -1e12,
        "verification_external_work": True,
    }
    rejected_state, rejection = accepted_geometry_plaquette_transaction(
        state, rejected_event, systems, topologies, driving, common, extensive,
        kinetics, 1e-9)
    before = digest_state(rejected_state)
    evolved, ledger = accepted_v24_mechanical_step(
        rejected_state, driving, support, systems, topologies, common,
        extensive, oldk, 1e-12, topology_route_enabled=False,
        mura_transport_operator="compatible_dealiased")
    continuum_dt = float(ledger["accepted_dt_s"])
    accepted_event = {
        **rejected_event, "cell": (8, 8),
        "external_work_J_m3_cells": 1e12,
    }
    final, accepted = accepted_geometry_plaquette_transaction(
        evolved, accepted_event, systems, topologies, driving, common,
        extensive, kinetics, 1e-9)
    return {
        "rejected_geometry": {
            "classification": rejection["classification"],
            "proposed_duration_s": rejection["proposed_duration_s"],
            "consumed_duration_s": rejection["consumed_duration_s"],
            "proposed_swept_area_m2": rejection["proposed_swept_area_m2"],
            "committed_swept_area_m2": rejection["committed_swept_area_m2"],
            "heat_sum_J_m3_cells": float(np.sum(
                rejection["irreversible_heat_increment_J_m3"])),
            "exact_rollback": digest_state(rejected_state) == digest_state(state),
        },
        "continuum": {
            "accepted_duration_s": continuum_dt,
            "state_changed": digest_state(evolved) != before,
            "mura_operator": ledger.get(
                "mura_transport_operator", "compatible_dealiased"),
        },
        "second_geometry_after_continuum": {
            "accepted": bool(accepted["accepted"]),
            "consumed_duration_s": accepted["consumed_duration_s"],
            "committed_swept_area_m2": accepted["committed_swept_area_m2"],
            "final_state_changed": digest_state(final) != digest_state(evolved),
        },
    }


def main():
    rows = []; controls = []
    for n in (16, 32, 64):
        forward_state, forward = scripted_path(n)
        reverse_state, reverse = scripted_path(n, reverse=True)
        split_state, split = scripted_path(n, split=True)
        forward_arrays = mechanical_checkpoint_arrays(forward_state)
        reverse_arrays = mechanical_checkpoint_arrays(reverse_state)
        split_arrays = mechanical_checkpoint_arrays(split_state)
        physical_keys = [key for key in forward_arrays
                         if not key.endswith("accepted_event_count")]
        order_max = max(float(np.max(np.abs(
            forward_arrays[key]-reverse_arrays[key]))) for key in physical_keys)
        controls.append({
            "grid": n,
            "site_order_byte_identical": (
                digest_state(forward_state) == digest_state(reverse_state)),
            "site_order_physical_arrays_close": all(np.allclose(
                forward_arrays[key], reverse_arrays[key],
                rtol=2e-14, atol=1e-30) for key in physical_keys),
            "site_order_maximum_absolute_array_difference": order_max,
            "patch_split_physical_arrays_close": all(np.allclose(
                forward_arrays[key], split_arrays[key],
                rtol=2e-14, atol=1e-30)
                for key in physical_keys),
            "reverse": reverse, "split": split,
        })
        rows.append(forward)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    payload = {
        "schema": "asb-drx/v45/dynamic-geometry-measure/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": source, "dynamic_refinement": rows,
        "order_and_subdivision_controls": controls,
        "blocked_channel_continuation": blocked_channel_continuation(),
        "scripted_path_is_spontaneous_organization": False,
        "drx_claimed": False, "material_calibration_claimed": False,
    }
    swept = [row["integrated_swept_area_m2"] for row in rows]
    line = [row["weighted_line_length_m"] for row in rows]
    geometric_energy = [row["geometric_line_energy_change_J"] for row in rows]
    coupled_energy = [row["mechanical_plus_line_energy_change_J"] for row in rows]
    payload["classification"] = {
        "swept_measure_grid_invariant": bool(np.allclose(
            swept, swept[0], rtol=2e-14, atol=1e-30)),
        "weighted_line_length_grid_invariant": bool(np.allclose(
            line, line[0], rtol=2e-14, atol=1e-30)),
        "geometric_line_energy_grid_invariant": bool(np.allclose(
            geometric_energy, geometric_energy[0], rtol=2e-14, atol=1e-30)),
        "continuum_coupled_energy_grid_converged": bool(
            abs(coupled_energy[-1]-coupled_energy[-2])
            <= .05*max(abs(coupled_energy[-1]), abs(coupled_energy[-2]), 1e-300)),
        "coupled_energy_interpretation": (
            "a one-cell singular line deposited into the continuum gradient "
            "energy is unresolved; geometric event measure qualifies but "
            "coupled energetic mesh convergence does not"),
    }
    output = Path("full_model/verification/v45_dynamic_geometry_measure.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps({
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "grids": [row["grid"] for row in rows],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
