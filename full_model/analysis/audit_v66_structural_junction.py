#!/usr/bin/env python3
"""Quantify the cycle defect created by conservative three-owner motion."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from full_model.production.structural_boundary import StructuralBoundaryParameters
from full_model.production.structural_interface_network import (
    JunctionDisclinationParameters, initialize_structural_interfaces,
    network_junction_disclination_energy_J_per_m,
    project_content_to_compatible_orientations, relax_junction_cycle_interval,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/"full_model"/"verification"/"v66_structural_front_geometry.json"
OUT = ROOT/"full_model"/"verification"/"v66_structural_junction_audit.json"


def main():
    source = json.loads(SOURCE.read_text())
    boundary = StructuralBoundaryParameters()
    ids = (10, 20, 30)
    edges = ((10, 20), (10, 30), (20, 30))
    cases = {}
    for name in ("prepared_static_handoff", "coupled_four_percent_shear"):
        record = source["low_angle_structural_test"][name]
        old_lengths = np.asarray(record["old_lengths_m"])
        new_lengths = np.asarray(record["new_lengths_m"])
        interfaces = initialize_structural_interfaces(
            ids, np.radians((0.0, 2.5, 5.0)), edges, old_lengths,
            boundary.burgers_m)
        projection = project_content_to_compatible_orientations(
            ids, interfaces, new_lengths, boundary.burgers_m,
            parameters=boundary)
        cutoff = {}
        for radius in (1e-7, 5e-7, 1e-6):
            p = JunctionDisclinationParameters(outer_cutoff_m=radius)
            cutoff[f"R{radius:.0e}_m"] = (
                network_junction_disclination_energy_J_per_m(
                    ids, interfaces, new_lengths, boundary.burgers_m, p))
        extrinsic = (boundary.extrinsic_line_energy_J_m
                     *float(np.sum(np.abs(
                         projection.junction_reservoir_change))))
        reaction = relax_junction_cycle_interval(
            ids, interfaces, new_lengths, boundary.burgers_m,
            temperature_K=1000.0, dt_s=1e-6,
            represented_thickness_m=2*boundary.burgers_m,
            boundary_parameters=boundary)
        cases[name] = {
            "fundamental_cycle_residual_deg": np.degrees(
                projection.before_compatibility.fundamental_cycle_residual_rad
            ).tolist(),
            "conservative_boundary_energy_change_J_per_m": (
                record["conservative_boundary_energy_change_J_per_m"]),
            "junction_disclination_energy_J_per_m_by_cutoff": cutoff,
            "instant_projection_edge_content_change": (
                projection.edge_content_change.tolist()),
            "instant_projection_boundary_energy_change_J_per_m": (
                projection.delta_boundary_energy_J_per_m),
            "instant_projection_extrinsic_reservoir_energy_J_per_m": extrinsic,
            "instant_projection_total_change_J_per_m": (
                projection.delta_boundary_energy_J_per_m+extrinsic),
            "projection_content_balance_max_abs": float(np.max(np.abs(
                projection.content_balance_residual))),
            "one_microsecond_arrhenius_reaction": {
                "classification": reaction.classification,
                "accepted": reaction.accepted,
                "frank_backstress_Pa": reaction.frank_backstress_Pa,
                "activation_enthalpy_eV": (
                    reaction.activation_enthalpy_J/1.602176634e-19),
                "rate_s": reaction.rate_s,
                "reacted_fraction": reaction.reacted_fraction,
                "absolute_reacted_content": reaction.absolute_reacted_content,
                "delta_helmholtz_J": reaction.delta_helmholtz_J,
                "generated_heat_J": reaction.generated_heat_J,
            },
        }
    payload = {
        "schema": "v66-structural-junction-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": "2be1347+structural-development",
        "model": {
            "cycle_measure": "fundamental scalar common-axis Frank closure",
            "wedge_energy": "mu*Omega^2*R^2/[16*pi*(1-nu)]",
            "shear_modulus_Pa": 70e9,
            "poisson_ratio": .34,
            "cutoff_interpretation": "declared screening radius; sensitivity, not calibration",
        },
        "cases": cases,
        "decision": {
            "fixed_labels_rejected": True,
            "silent_projection_rejected": True,
            "conservative_cycle_reservoir_required": True,
            "production_status": (
                "bounded EXP-floor reaction is evaluable and discriminates "
                "an uphill static rollback from a downhill deformed-state "
                "event; coupled runtime promotion is next"),
        },
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
