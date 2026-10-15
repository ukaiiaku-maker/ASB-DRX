#!/usr/bin/env python3
"""Compare explicit interface and near-boundary lattice Burgers capacity."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from full_model.analysis.run_v58_three_grain_production import _load_checkpoint
from full_model.production.interface_structural_absorption import (
    project_interface_burgers_to_structural_content,
)
from full_model.production.structural_interface_network import (
    initialize_structural_interfaces, interface_lengths_m,
)
from full_model.production.tensorial_nye import (
    bcc_four_family_systems, rotated_system_fields, spectral_derivatives,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    state, runtime, step, load, _, configuration, provenance = _load_checkpoint(
        args.checkpoint)
    spacing = float(configuration["length_m"])/int(configuration["n"])
    burgers_m = 2.48e-10
    lengths = interface_lengths_m(
        state.supports, runtime.interfaces, state.grain_ids, spacing)
    orientations = np.asarray([
        np.mean(owner.orientation_rad) for owner in state.owners])
    structural = initialize_structural_interfaces(
        state.grain_ids, orientations,
        tuple((edge.grain_a_id, edge.grain_b_id) for edge in runtime.interfaces),
        lengths, burgers_m)
    explicit = project_interface_burgers_to_structural_content(
        state, structural, spacing_m=spacing, fraction=1.0,
        burgers_m=burgers_m)
    explicit_by_edge = {
        edge.interface_id: float(value)
        for edge, value in zip(structural, explicit.structural_content_change)}
    owner_index = {owner: index for index, owner in enumerate(state.grain_ids)}
    systems = bcc_four_family_systems(burgers_m)
    rows = []
    for edge in runtime.interfaces:
        ia = owner_index[edge.grain_a_id]
        ib = owner_index[edge.grain_b_id]
        eta = state.supports[ib]-state.supports[ia]
        gx, gy = spectral_derivatives(eta, spacing)
        magnitude = np.sqrt(gx*gx+gy*gy)
        active = magnitude > 1e-12*max(float(np.max(magnitude)), 1.0)
        normal = np.stack((
            np.divide(gx, magnitude, out=np.zeros_like(gx), where=active),
            np.divide(gy, magnitude, out=np.zeros_like(gy), where=active),
            np.zeros_like(gx)), axis=-1)
        # Symmetric diffuse capture band. It is a capacity measure, not yet a
        # kinetic sink or a production debit rule.
        capture = 4.0*state.supports[ia]*state.supports[ib]
        contributions = []
        for owner_position in (ia, ib):
            owner = state.owners[owner_position]
            burgers, _, _ = rotated_system_fields(
                systems, owner.orientation_rad)
            signed = sum(
                np.asarray(getattr(owner, plus))-np.asarray(getattr(owner, minus))
                for plus, minus in (
                    ("mobile_plus_m2", "mobile_minus_m2"),
                    ("forest_plus_m2", "forest_minus_m2"),
                    ("wall_plus_m2", "wall_minus_m2")))
            vector_density = np.einsum(
                "...a,...ai->...i", signed, burgers)
            projection = np.sum(vector_density*normal, axis=-1)
            signed_content = float(np.sum(
                capture*projection, dtype=np.longdouble)*spacing**2/burgers_m)
            absolute_content = float(np.sum(
                capture*np.abs(projection), dtype=np.longdouble)
                *spacing**2/burgers_m)
            contributions.append({
                "owner_id": int(state.grain_ids[owner_position]),
                "signed_common_axis_content": signed_content,
                "absolute_common_axis_capacity": absolute_content,
            })
        lattice_signed = sum(row["signed_common_axis_content"]
                             for row in contributions)
        lattice_absolute = sum(row["absolute_common_axis_capacity"]
                               for row in contributions)
        explicit_content = explicit_by_edge[edge.component_id]
        rows.append({
            "interface": edge.component_id,
            "owner_contributions": contributions,
            "lattice_signed_common_axis_content": lattice_signed,
            "lattice_absolute_common_axis_capacity": lattice_absolute,
            "explicit_interface_signed_capacity": explicit_content,
            "absolute_lattice_over_explicit_ratio": (
                lattice_absolute/max(abs(explicit_content),
                                     np.finfo(float).tiny)),
        })
    payload = {
        "schema": "v67-lattice-versus-interface-absorption-capacity-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": int(step),
        "applied_shear_strain": float(load),
        "checkpoint_provenance": provenance,
        "capture_band": "4 eta_a eta_b; symmetric diffuse-interface capacity",
        "interfaces": rows,
        "decision": {
            "interface_reservoir_capacity_is_rate_limiting": bool(all(
                row["absolute_lattice_over_explicit_ratio"] > 1e3
                for row in rows)),
            "next_hypothesis": (
                "energy-complete selective lattice-to-structural absorption "
                "with joint owner capacity and explicit incompatible residual"),
            "production_promotion": False,
            "reason": (
                "This audit establishes accessible signed capacity only. A "
                "production reaction must jointly debit positive/negative owner "
                "inventories, conserve the full Burgers vector, retain tangential "
                "content, and pass complete work/heat accounting."),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
