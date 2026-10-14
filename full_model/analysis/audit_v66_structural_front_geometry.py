#!/usr/bin/env python3
"""Audit fixed owner orientations against conserved structural content."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from full_model.analysis.run_v58_three_grain_production import (
    _load_checkpoint, network_interfaces,
)
from full_model.analysis.run_v66_coupled_rx_asb import _physics
from full_model.analysis.run_v66_multigrain_handoff import _advance, _initial
from full_model.production.multigrain_production import MultiGrainProductionRuntime
from full_model.production.structural_interface_network import (
    conservative_geometry_update, initialize_structural_interfaces,
    interface_lengths_m,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"full_model"/"verification"/"v66_structural_front_geometry.json"


def _serial_ledger(ledger, fixed_angles):
    old_c = np.asarray([
        item.signed_integrated_content for item in ledger.interfaces])
    fixed_theta = np.asarray((fixed_angles[1]-fixed_angles[0],
                              fixed_angles[2]-fixed_angles[0],
                              fixed_angles[2]-fixed_angles[1]))
    # Interface order below is 10-20, 10-30, 20-30.
    fixed_content = ledger.new_lengths_m*fixed_theta/2.48e-10
    return {
        "old_lengths_m": ledger.old_lengths_m.tolist(),
        "new_lengths_m": ledger.new_lengths_m.tolist(),
        "conservative_old_misorientation_deg": np.degrees(
            ledger.old_misorientation_rad).tolist(),
        "conservative_new_misorientation_deg": np.degrees(
            ledger.new_misorientation_rad).tolist(),
        "fixed_label_content_change": (fixed_content-old_c).tolist(),
        "fixed_label_content_relative_change": np.divide(
            fixed_content-old_c, np.maximum(np.abs(old_c), 1e-300)).tolist(),
        "conservative_boundary_energy_change_J_per_m": (
            ledger.energy_change_J_per_m),
        "fitted_owner_orientations_deg": np.degrees(
            ledger.compatibility.fitted_orientation_rad).tolist(),
        "edge_compatibility_residual_deg": np.degrees(
            ledger.compatibility.edge_compatibility_residual_rad).tolist(),
        "weighted_rms_compatibility_residual_deg": float(np.degrees(
            ledger.compatibility.weighted_rms_compatibility_residual_rad)),
        "maximum_cycle_residual_deg": float(np.degrees(
            ledger.compatibility.maximum_cycle_residual_rad)),
    }


def main():
    args = SimpleNamespace(
        n=16, length=4.8e-6, temperature=1000.0,
        shear_rate=4e4, dt=2e-9)
    spacing, thickness, systems, wall, kinetics, energy, _ = _physics(args)
    reduced = json.loads((ROOT/"full_model"/"verification"/
                          "v66_subgrain_maturation"/"result.json").read_text())
    contrast = reduced["cases"]["T1000_rate1"]["stored_energy_contrast_Pa"]
    initial, _, _ = _initial(16, args.length, args.temperature, contrast)
    prepared, _, _, _ = _advance(
        initial, MultiGrainProductionRuntime(network_interfaces(initial)),
        20, 2e-8, spacing=spacing, thickness=thickness, wall=wall,
        kinetics=kinetics, energy=dict(energy, mean_strain=np.zeros((2, 2))),
        systems=systems)
    coupled, *_ = _load_checkpoint(
        ROOT/"full_model"/"verification"/"v66_coupled_source_disabled_v2"/
        "checkpoint_latest.npz")
    ids = initial.grain_ids
    edges = ((10, 20), (10, 30), (20, 30))
    edge_records = tuple(network_interfaces(initial))
    # network_interfaces uses the same sorted edge order declared above.
    assert tuple((item.grain_a_id, item.grain_b_id)
                 for item in edge_records) == edges
    lengths0 = interface_lengths_m(initial.supports, edge_records, ids, spacing)
    lengths_prepared = interface_lengths_m(
        prepared.supports, edge_records, ids, spacing)
    lengths_coupled = interface_lengths_m(
        coupled.supports, edge_records, ids, spacing)

    lagb_angles = np.radians((0.0, 2.5, 5.0))
    lagb_interfaces = initialize_structural_interfaces(
        ids, lagb_angles, edges, lengths0, wall.burgers_m)
    prepared_ledger = conservative_geometry_update(
        ids, lagb_interfaces, lengths0, lengths_prepared, wall.burgers_m)
    # The coupled production exposure starts from the prepared handoff. Reset
    # its structural ledger there so this audit isolates only the subsequent
    # four-percent-shear front motion.
    lagb_prepared_interfaces = initialize_structural_interfaces(
        ids, lagb_angles, edges, lengths_prepared, wall.burgers_m)
    coupled_ledger = conservative_geometry_update(
        ids, lagb_prepared_interfaces, lengths_prepared, lengths_coupled,
        wall.burgers_m)

    mature_angles = np.radians((0.0, 15.0, 30.0))
    mature_interfaces = initialize_structural_interfaces(
        ids, mature_angles, edges, lengths_prepared, wall.burgers_m)
    mature_coupled = conservative_geometry_update(
        ids, mature_interfaces, lengths_prepared, lengths_coupled,
        wall.burgers_m)
    payload = {
        "schema": "v66-structural-front-geometry-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": "2be1347",
        "geometry": {
            "grid": [16, 16], "spacing_m": spacing,
            "interface_measure": "symmetric diffuse pair-contour length",
            "edge_order": [list(edge) for edge in edges],
        },
        "low_angle_structural_test": {
            "initial_owner_orientations_deg": [0.0, 2.5, 5.0],
            "prepared_static_handoff": _serial_ledger(
                prepared_ledger, lagb_angles),
            "coupled_four_percent_shear": _serial_ledger(
                coupled_ledger, lagb_angles),
        },
        "mature_owner_geometry_only_diagnostic": {
            "initial_owner_orientations_deg": [0.0, 15.0, 30.0],
            "coupled_four_percent_shear": _serial_ledger(
                mature_coupled, mature_angles),
            "scope": (
                "geometry/content conservation diagnostic only; scalar low-angle "
                "Read-Shockley structure is not asserted for the 30 degree edge"),
        },
        "decision": {
            "fixed_owner_labels_conserve_structural_content": False,
            "conservative_motion_generates_rotation": True,
            "differential_edge_motion_generates_cycle_residual": bool(
                coupled_ledger.compatibility.maximum_cycle_residual_rad > 1e-14),
            "production_implication": (
                "A front cannot simultaneously keep every owner angle fixed, conserve "
                "each edge's structural content, and preserve Frank compatibility after "
                "differential interface motion. Production needs a structural-content "
                "state plus conservative redistribution/reaction at network cycles."),
        },
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload["decision"], indent=2))
    print(json.dumps(payload["low_angle_structural_test"][
        "coupled_four_percent_shear"], indent=2))


if __name__ == "__main__":
    main()
