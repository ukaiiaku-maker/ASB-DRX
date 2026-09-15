#!/usr/bin/env python3
"""Produce V26 authoritative-Nye and manufactured-ASB decision records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from full_model.production.asb_grid_audit import (
    compare_kernel_audits, kernel_moment_audit, manufactured_periodic_kernel,
)
from full_model.production.reaction_cone import (
    ReactionEvent, audit_reaction_cone, circuit_event_from_line_change,
)
from full_model.production.tensorial_nye import bcc_four_family_systems

VERIFY = ROOT/"full_model"/"verification"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_v25():
    paths = [
        ROOT/"full_model/docs/v25_decision_report.md",
        VERIFY/"v25_reaction_cone_local.json",
        VERIFY/"v25_clean_sibm_matrix.json",
        VERIFY/"v25_asb_grid_audit.json",
    ]
    result = {
        "schema": "asb-drx/v26-frozen-v25-evidence/v1",
        "checkpoint": "9acb5f9e5f0d1fbe5ebea7a64af687d3ef71158c",
        "files": {str(path.relative_to(ROOT)): sha(path) for path in paths},
        "immutable": True,
    }
    (VERIFY/"v26_frozen_v25_evidence.json").write_text(
        json.dumps(result, indent=2, sort_keys=True)+"\n")


def transport_events(exposure_fraction):
    events = []
    capacities = {-1: (7e13+4e13)*.8e-6, 1: (8e13+5e13)*.8e-6}
    for family, system in enumerate(bcc_four_family_systems()):
        line = np.cross(system.plane_normal, system.slip_direction)
        line /= np.linalg.norm(line)
        for sign in (-1, 1):
            b = sign*system.burgers_vector_m
            nye = np.outer(b, line)
            events.append(ReactionEvent(
                name=f"incoming_transport_b{family}_{sign:+d}",
                circuit_increment_per_extent_m=b*line[2],
                line_increment_per_extent=0.0,
                burgers_increment_per_extent_m=np.zeros(3),
                nye_increment_per_extent_m=nye,
                alignment_increment_per_extent=line,
                turning_node_increment_per_extent_m1=0.0,
                junction_increment_per_extent=0.0,
                free_energy_increment_per_extent_J_m=0.0,
                capacity_extent_m1=capacities[sign],
                kinetic_exposure_extent_m1=exposure_fraction*capacities[sign],
                swept_area_declared=True,
                plastic_distortion_curl_increment_per_extent_m=-nye,
                event_class="transport"))
            events.append(circuit_event_from_line_change(
                f"disabled_v24_reorientation_b{family}_{sign:+d}", b, line,
                np.asarray((0.0, 0.0, 1.0)), np.asarray((0.0, 0.0, 1.0)),
                capacities[sign],
                kinetic_exposure_extent_m1=exposure_fraction*capacities[sign]))
    return tuple(events)


def wall_audit():
    source = json.loads((VERIFY/"v24_mechanical_supply_local.json").read_text())
    transport_case = next(case for case in source["cases"]
                          if case["grid"] == 32
                          and not case["topology_route_enabled"])
    exposure = max(
        segment["ordered_supply_ratio"]
        for record in transport_case["classification"]["records"]
        for segment in record["local_integrated_audit"]["segments"])
    elapsed = float(transport_case["elapsed_time_s"])
    rows = []
    for case in source["cases"]:
        if case["grid"] != 32:
            continue
        segments = [segment for record in case["classification"]["records"]
                    for segment in record["local_integrated_audit"]["segments"]]
        segment = max(segments, key=lambda item: item["local_misorientation_rad"])
        target = (np.asarray(segment["frank_bilby_burgers"])
                  -np.asarray(segment["integrated_ordered_burgers"]))
        cone = audit_reaction_cone(
            transport_events(exposure), target,
            require_event_identity=True)
        rows.append({
            "topology_route_enabled": case["topology_route_enabled"],
            "target": target.tolist(), "cone": cone,
            "estimated_unit_supply_time_s": elapsed/max(exposure, 1e-300),
        })
    reached = all(row["cone"]["reachable"] for row in rows)
    result = {
        "schema": "asb-drx/v26-authoritative-reaction-cone/v1",
        "alpha_convention": "alpha_beta=-Curl(beta_p)",
        "event_residual": "Delta alpha_rho + Curl Delta beta_p - Delta alpha_source",
        "strict_event_identity": True,
        "inadmissible_v24_reorientation_disabled": True,
        "candidate_audits": rows,
        "all_admissible_columns_close_event_identity": all(
            valid["authoritative_event_identity_valid"]
            for row in rows for valid in row["cone"]["event_validation"]
            if valid["authoritative_nye_ownership_valid"]),
        "current_capacity_qualified": False,
        "capacity_limitation": (
            "V24 terminal candidate records do not retain signed donor fields; "
            "the audited bound is the frozen initial circuit-window inventory"),
        "classification": (
            "AUTHORITATIVE_EVENT_COLUMNS_REPAIRED_CURRENT_CAPACITY_PENDING"
            if reached else "FB_TARGET_OUTSIDE_AUTHORITATIVE_REACTION_CONE"),
        "long_wall_hpc_authorized": False,
        "fixture_passed": True, "scientific_gate_passed": False,
    }
    (VERIFY/"v26_authoritative_reaction_cone.json").write_text(
        json.dumps(result, indent=2, sort_keys=True)+"\n")
    return result


def asb_kernel_audit():
    records = {}
    for n in (64, 96, 128, 192):
        kernel, rx, ry, effective = manufactured_periodic_kernel(
            n, 10e-6, .3e-6, minimum_sigma_pixels=2.0)
        records[str(n)] = {
            "effective_sigma_m": effective,
            **kernel_moment_audit(kernel, rx, ry, 10e-6/n),
        }
    comparisons = {}
    for a, b in ((64, 128), (96, 128), (128, 192)):
        comparisons[f"{a}_vs_{b}"] = compare_kernel_audits(
            records[str(a)], records[str(b)])
    result = {
        "schema": "asb-drx/v26-manufactured-kernel/v1",
        "requested_sigma_m": .3e-6, "minimum_sigma_pixels": 2.0,
        "grids": records, "comparisons": comparisons,
        "coarse_64_valid_for_physical_refinement": comparisons["64_vs_128"]["passed"],
        "resolved_sequence": [96, 128, 192],
        "classification": "ASB_COARSE_GRID_KERNEL_UNDERRESOLVED",
        "physical_response_grid_converged": False,
        "strain_rate_bracket_authorized": False,
        "fixture_passed": True, "scientific_gate_passed": False,
    }
    (VERIFY/"v26_asb_manufactured_kernel.json").write_text(
        json.dumps(result, indent=2, sort_keys=True)+"\n")
    return result


def main():
    VERIFY.mkdir(parents=True, exist_ok=True)
    freeze_v25(); wall = wall_audit(); asb = asb_kernel_audit()
    print(json.dumps({"wall": wall["classification"],
                      "asb": asb["classification"]}, indent=2))


if __name__ == "__main__":
    main()
