#!/usr/bin/env python3
"""Replay a mechanical transition band with an online current-capacity cone."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from full_model.analysis.run_v24_mechanical_supply import build_case, snapshot
from full_model.production.reaction_cone import ReactionEvent, audit_reaction_cone
from full_model.production.tensorial_nye import rotated_system_fields
from full_model.production.v24_mechanical_wall import accepted_v24_mechanical_step
from full_model.production.wall_circuit_diagnostics import (
    classify_persistent_wall_history,
)
from full_model.production.wall_topology_supply import reservoir_nye_m1


SIGNED = ("mobile_plus_m2", "mobile_minus_m2", "forest_plus_m2",
          "forest_minus_m2", "wall_tangle_plus_m2", "wall_tangle_minus_m2",
          "wall_ordered_plus_m2", "wall_ordered_minus_m2")


def candidate_segment(state, systems, topologies, dx, time_s):
    classified = classify_persistent_wall_history(
        [snapshot(state, systems, topologies, dx, time_s, True)],
        required_release_persistence_s=0.0)
    segments = [segment for record in classified["records"]
                for segment in record["local_integrated_audit"]["segments"]]
    return (max(segments, key=lambda item: item["local_misorientation_rad"])
            if segments else None)


def periodic_strip(shape, centroid, normal, dx, half_width_m):
    ii = ((np.arange(shape[0])-centroid[0]+shape[0]/2) % shape[0]
          -shape[0]/2)*dx
    jj = ((np.arange(shape[1])-centroid[1]+shape[1]/2) % shape[1]
          -shape[1]/2)*dx
    distance = normal[0]*ii[:, None]+normal[1]*jj[None, :]
    return np.abs(distance) <= half_width_m


def online_record(state, segment, systems, topologies, dx, elapsed,
                  cumulative_capture):
    centroid = np.asarray(segment["centroid_index"], float)
    normal = np.asarray(segment["normal_xy"], float)
    strip = periodic_strip(
        state.common.orientation_rad.shape, centroid, normal, dx, .4e-6)
    length = max(float(segment["estimated_segment_length_m"]), dx)
    i = int(round(centroid[0])) % strip.shape[0]
    j = int(round(centroid[1])) % strip.shape[1]
    burgers, slip, plane = rotated_system_fields(
        systems, state.common.orientation_rad)
    line = np.cross(plane, slip)
    events = []
    capacities = {}
    exposures = {}
    for family in range(len(systems)):
        for sign_name, sign_value in (("minus", -1), ("plus", 1)):
            mobile = np.asarray(getattr(
                state.density, f"mobile_{sign_name}_m2"))[..., family]
            capacity = float(np.sum(mobile[strip], dtype=np.longdouble)
                             *dx*dx/length)
            exposure = float(cumulative_capture[sign_name][family]*dx*dx/length)
            name = f"incoming_transport_b{family}_{sign_value:+d}"
            capacities[name] = capacity; exposures[name] = exposure
            b = sign_value*burgers[i, j, family]
            direction = line[i, j, family]
            nye = np.outer(b, direction)
            events.append(ReactionEvent(
                name=name,
                circuit_increment_per_extent_m=b*direction[2],
                line_increment_per_extent=0.0,
                burgers_increment_per_extent_m=np.zeros(3),
                nye_increment_per_extent_m=nye,
                alignment_increment_per_extent=direction,
                turning_node_increment_per_extent_m1=0.0,
                junction_increment_per_extent=0.0,
                free_energy_increment_per_extent_J_m=0.0,
                capacity_extent_m1=capacity,
                kinetic_exposure_extent_m1=exposure,
                swept_area_declared=True,
                plastic_distortion_curl_increment_per_extent_m=-nye,
                event_class="measured_incoming_transport"))
    target = (np.asarray(segment["frank_bilby_burgers"])
              -np.asarray(segment["integrated_ordered_burgers"]))
    cone = audit_reaction_cone(
        tuple(events), target, require_event_identity=True)
    nye = reservoir_nye_m1(
        state.reservoir_alignment, systems,
        state.common.orientation_rad, topologies)
    curl_beta = np.sum(state.common.family_nye_m1, axis=2)
    reservoir_means = {
        name: np.mean(np.asarray(getattr(state.density, name))[strip], axis=0).tolist()
        for name in SIGNED}
    alignment_means = {
        name: np.mean(np.asarray(getattr(state.reservoir_alignment, name))[strip],
                      axis=0).tolist() for name in SIGNED}
    topology_capacity = []
    for topology in topologies:
        aname = f"wall_tangle_{'plus' if topology.sign_a > 0 else 'minus'}_m2"
        bname = f"wall_tangle_{'plus' if topology.sign_b > 0 else 'minus'}_m2"
        first = np.asarray(getattr(state.density, aname))[..., topology.parent_a]
        second = np.asarray(getattr(state.density, bname))[..., topology.parent_b]
        topology_capacity.append(float(np.sum(
            np.minimum(first, second)[strip], dtype=np.longdouble)*dx*dx/length))
    return {
        "time_s": elapsed, "segment": segment,
        "reservoir_means_m2_by_family": reservoir_means,
        "alignment_means_m2_by_family_xyz": alignment_means,
        "junction_means_m2": np.mean(
            state.density.junction_m2[strip], axis=0).tolist(),
        "beta_p_mean": np.mean(state.common.beta_p[strip], axis=0).tolist(),
        "authoritative_curl_beta_mean_m1": np.mean(curl_beta[strip], axis=0).tolist(),
        "reservoir_nye_mean_m1": np.mean(nye["total"][strip], axis=0).tolist(),
        "dual_nye_relative_rms": float(np.sqrt(np.mean(
            (curl_beta[strip]-nye["total"][strip])**2))/max(
                float(np.sqrt(np.mean(curl_beta[strip]**2))), 1.0)),
        "current_transport_capacity_m1": capacities,
        "cumulative_captured_exposure_m1": exposures,
        "junction_topology_capacity_m1": topology_capacity,
        "finite_segment_reorientation_enabled": False,
        "frank_bilby_deficit": target.tolist(),
        "online_cone": cone,
    }


def main():
    n = 32
    (state, driving, _, support, systems, topologies,
     common, extensive, kinetics, dx) = build_case(n)
    elapsed = 0.0
    cumulative = {sign: np.zeros(len(systems)) for sign in ("plus", "minus")}
    records = []
    for step in range(300):
        state, ledger = accepted_v24_mechanical_step(
            state, driving, support, systems, topologies, common, extensive,
            kinetics, 2e-9, topology_route_enabled=False)
        elapsed += float(ledger["accepted_dt_s"])
        for sign in ("plus", "minus"):
            captured = ledger["transport_capture"]["sign"][sign]["captured_line_m2"]
            cumulative[sign] += np.sum(captured, axis=(0, 1))
        if step in (49, 99, 149, 199, 249, 299):
            segment = candidate_segment(
                state, systems, topologies, dx, elapsed)
            if segment is not None:
                records.append(online_record(
                    state, segment, systems, topologies, dx, elapsed, cumulative))
    classifications = [row["online_cone"]["classification"] for row in records]
    dual_nye_consistent = bool(records) and all(
        row["dual_nye_relative_rms"] <= .05 for row in records)
    if not dual_nye_consistent:
        decision = "REACTION_CONE_INVALID_DUE_TO_DUAL_NYE_MISMATCH"
    elif any(value == "FB_TARGET_REACHABLE_WITH_CURRENT_CAPACITY"
           for value in classifications):
        decision = "FB_TARGET_REACHABLE_WITH_CURRENT_CAPACITY"
    elif any(value == "FB_TARGET_REACHABLE_BUT_KINETICALLY_UNEXPOSED"
             for value in classifications):
        decision = "FB_TARGET_REACHABLE_BUT_KINETICALLY_UNEXPOSED"
    elif records and all(row["online_cone"]["event_identity_required"]
                         for row in records):
        decision = "FB_TARGET_OUTSIDE_AUTHORITATIVE_REACTION_CONE"
    else:
        decision = "REACTION_CONE_INVALID_DUE_TO_DUAL_NYE_MISMATCH"
    result = {
        "schema": "asb-drx/v27-current-capacity-wall/v1",
        "grid": n, "spacing_m": dx, "elapsed_time_s": elapsed,
        "phase_and_grain_allocation_enabled": False,
        "future_orientation_target_used": False,
        "finite_segment_reorientation_enabled": False,
        "cone_optimizer_changes_state": False,
        "records": records, "classification": decision,
        "dual_nye_consistency_tolerance": .05,
        "dual_nye_consistent": dual_nye_consistent,
        "fixture_passed": bool(records),
        "scientific_gate_passed": bool(decision in (
            "FB_TARGET_REACHABLE_WITH_CURRENT_CAPACITY",
            "FB_TARGET_REACHABLE_BUT_KINETICALLY_UNEXPOSED")),
        "source_sha256": {name: hashlib.sha256(
            (ROOT/"full_model/production"/name).read_bytes()).hexdigest()
            for name in ("v24_mechanical_wall.py", "wall_topology_supply.py",
                         "reaction_cone.py", "common_tensorial_wall.py")},
    }
    output = ROOT/"full_model/verification/v27_current_capacity_wall.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(decision)


if __name__ == "__main__":
    main()
