#!/usr/bin/env python3
"""Non-vacuous verification of the V43 persistent lattice geometry route."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from full_model.analysis.run_v24_mechanical_supply import build_case
from full_model.production.arrhenius_kinetics import ActivatedProcess
from full_model.production.lattice_line_geometry import (
    empty_lattice_geometry, geometry_reservoir_fields,
    link_node_balance, propose_plaquette_sweep,
)
from full_model.production.v24_mechanical_wall import (
    V24MechanicalWallState, V43GeometryKinetics,
    accepted_geometry_plaquette_transaction, accepted_v24_mechanical_step,
    mechanical_checkpoint_arrays, mechanical_from_checkpoint_arrays,
    synchronize_common,
)


def digest_arrays(payload):
    sha = hashlib.sha256()
    for key in sorted(payload):
        value = np.ascontiguousarray(payload[key])
        sha.update(key.encode()); sha.update(value.dtype.str.encode())
        sha.update(str(value.shape).encode()); sha.update(value.tobytes())
    return sha.hexdigest()


def prepare_block(n, cells):
    data = build_case(n, length_m=3.2e-6, periodic_nye_consistent=True)
    state, driving, _, support, systems, topologies, common, extensive, oldk, dx = data
    geometry = empty_lattice_geometry((n, n), len(systems), dx)
    state = V24MechanicalWallState(
        state.common, state.density, state.reservoir_alignment, geometry)
    ledgers = []
    for cell in cells:
        candidate = propose_plaquette_sweep(
            state.geometry, state.density, state.reservoir_alignment,
            state.common, systems, state.common.orientation_rad, cell, 0, 1, 1.0)
        state = synchronize_common(V24MechanicalWallState(
            candidate[3], candidate[1], candidate[2], candidate[0]), topologies)
        ledgers.append(candidate[-1])
    state.validate(systems, topologies)
    return state, data, ledgers


def compact(ledger):
    keys = (
        "operator", "accepted", "classification", "accepted_extent",
        "swept_area_m2", "line_length_before_m", "line_length_after_m",
        "line_length_change_m", "maximum_node_balance_residual",
        "nye_curl_increment_rms_residual_m1", "event_rate_s",
        "kinetic_extent_capacity", "wall_energy_change_J_m3_cells",
        "elastic_energy_change_J_m3_cells", "external_work_J_m3_cells",
        "complete_energy_change_J_m3_cells", "energy_tolerance_J_m3_cells",
        "heat_plus_complete_energy_residual_J_m3_cells", "rejection_is_atomic",
        "reason",
    )
    return {key: ledger.get(key) for key in keys if key in ledger}


def main():
    root = Path("full_model/verification")
    state, data, preparation = prepare_block(16, [(7, 7)])
    _, driving, _, support, systems, topologies, common, extensive, oldk, dx = data
    kinetics = V43GeometryKinetics(
        ActivatedProcess("represented-plaquette-sweep", 1e12,
                         entropy_over_kB=-.2, negative_barrier_mode="drag"),
        enthalpy_J=0.0, critical_stress_Pa=1e9)
    event = {
        "cell": (8, 7), "family": 0, "burgers_sign": 1,
        "proposed_extent": 1.0, "external_work_J_m3_cells": 1e12,
        "verification_external_work": True,
    }
    first, first_ledger = accepted_geometry_plaquette_transaction(
        state, event, systems, topologies, driving, common, extensive,
        kinetics, 1e-9)
    rejected_event = {**event, "cell": (8, 8),
                      "external_work_J_m3_cells": -1e12}
    rejected, rejection = accepted_geometry_plaquette_transaction(
        first, rejected_event, systems, topologies, driving, common,
        extensive, kinetics, 1e-9)
    restart_payload = mechanical_checkpoint_arrays(first)
    restored = mechanical_from_checkpoint_arrays(
        restart_payload, systems, topologies)
    second_event = {**event, "cell": (8, 8)}
    continuous, second_a = accepted_geometry_plaquette_transaction(
        first, second_event, systems, topologies, driving, common, extensive,
        kinetics, 1e-9)
    restarted, second_b = accepted_geometry_plaquette_transaction(
        restored, second_event, systems, topologies, driving, common, extensive,
        kinetics, 1e-9)
    exact_restart = digest_arrays(mechanical_checkpoint_arrays(continuous)) == \
        digest_arrays(mechanical_checkpoint_arrays(restarted))
    exact_rollback = digest_arrays(mechanical_checkpoint_arrays(first)) == \
        digest_arrays(mechanical_checkpoint_arrays(rejected))

    off, off_ledger = accepted_v24_mechanical_step(
        state, driving, support, systems, topologies, common, extensive, oldk,
        1e-9, topology_route_enabled=False)
    enabled, on_ledger = accepted_v24_mechanical_step(
        state, driving, support, systems, topologies, common, extensive, oldk,
        1e-9, topology_route_enabled=False, geometry_event=event,
        geometry_kinetics=kinetics)
    rho, moment = geometry_reservoir_fields(continuous.geometry)
    node = link_node_balance(continuous.geometry.edge_x_quanta,
                             continuous.geometry.edge_y_quanta)

    # The same 0.8-um square loop is represented by 4x4 and 8x8 plaquettes.
    coarse_cells = [(i, j) for i in range(6, 10) for j in range(6, 10)]
    fine_cells = [(i, j) for i in range(12, 20) for j in range(12, 20)]
    coarse, _, _ = prepare_block(16, coarse_cells)
    fine, _, _ = prepare_block(32, fine_cells)
    def line_length(geometry):
        return float(np.sum(np.abs(geometry.edge_x_quanta)
                            +np.abs(geometry.edge_y_quanta))*geometry.spacing_m)
    lc, lf = line_length(coarse.geometry), line_length(fine.geometry)

    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    evidence = {
        "schema": "asb-drx/v43/nonzero-geometry-event/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": source,
        "prepared_state_scope": (
            "declared one-plaquette closed loop; representation initialization "
            "only, with no physical heat and no claim of simulated prior history"),
        "preparation": compact(preparation[-1]),
        "first_accepted_event": compact(first_ledger),
        "rejected_event": compact(rejection),
        "second_event_from_evolved_state": compact(second_a),
        "second_event_after_restart": compact(second_b),
        "exact_rejection_rollback": exact_rollback,
        "exact_checkpoint_continuation": exact_restart,
        "accepted_event_count_after_second": int(continuous.geometry.accepted_event_count),
        "maximum_endpoint_balance_residual": float(np.max(np.abs(node))),
        "maximum_moment_realizability_excess_m2": float(
            np.max(np.linalg.norm(moment, axis=-1)-rho)),
        "full_production_comparison": {
            "disabled_event_count": int(off.geometry.accepted_event_count),
            "enabled_event_count": int(enabled.geometry.accepted_event_count),
            "enabled_event": compact(on_ledger["geometry_event_energy_kinematics"]),
            "enabled_nye_hard_invariant_passed": bool(
                on_ledger["nye_suboperator_audit"][
                    "accepted_step_hard_invariant_passed"]),
            "beta_p_paths_differ": bool(not np.array_equal(
                off.common.beta_p, enabled.common.beta_p)),
        },
        "segment_refinement": {
            "physical_square_width_m": 8e-7,
            "n16_line_length_m": lc, "n32_line_length_m": lf,
            "relative_difference": abs(lc-lf)/max(lc, lf, 1e-300),
        },
        "scientific_claim": "MANUFACTURED_NONZERO_GEOMETRY_EVENT_QUALIFIED",
        "spontaneous_wall_or_drx_claimed": False,
    }
    (root/"v43_nonzero_geometry_event.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True)+"\n")
    capabilities = {
        "schema": "asb-drx/v43/representation-capabilities/v1",
        "source_sha": source,
        "representation": "periodic oriented-link boundary of swept plaquettes",
        "represented": [
            "persistent closed line links", "Burgers family and sign",
            "periodic winding", "swept surface", "scalar line length",
            "first line moment", "plastic distortion increment",
            "family-resolved Nye increment", "finite line stretching/shrinkage",
        ],
        "not_represented": [
            "free off-lattice node coordinates", "three-dimensional junction products",
            "cross-family Burgers reactions", "spontaneous nucleation law",
        ],
        "geometry_neutral_v42_comparator_retained": True,
        "publication_safety": "qualified for represented plaquette events only",
        "material_calibration": False,
    }
    (root/"v43_representation_capabilities.json").write_text(
        json.dumps(capabilities, indent=2, sort_keys=True)+"\n")

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), constrained_layout=True)
    q = np.sum(continuous.geometry.swept_quanta, axis=(2, 3))
    ex = np.sum(continuous.geometry.edge_x_quanta, axis=(2, 3))
    ey = np.sum(continuous.geometry.edge_y_quanta, axis=(2, 3))
    axes[0].imshow(q.T, origin="lower"); axes[0].set_title("swept plaquettes")
    axes[1].quiver(ex.T, ey.T); axes[1].set_title("oriented line links")
    axes[2].imshow(np.linalg.norm(np.sum(
        continuous.common.family_nye_m1, axis=2), axis=(-2, -1)).T,
        origin="lower"); axes[2].set_title("Nye norm [m$^{-1}$]")
    axes[3].imshow(np.linalg.norm(continuous.common.beta_p, axis=(-2, -1)).T,
                   origin="lower"); axes[3].set_title("plastic distortion")
    fig.savefig(root/"v43_nonzero_geometry_event.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"source_sha": source,
                      "first_accepted": first_ledger["accepted"],
                      "restart_exact": exact_restart,
                      "rollback_exact": exact_rollback}, sort_keys=True))


if __name__ == "__main__":
    main()
