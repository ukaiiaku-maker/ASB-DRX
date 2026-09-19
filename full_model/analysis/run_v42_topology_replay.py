#!/usr/bin/env python3
"""Classify the first legacy and repaired topology increments from 5% strain."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from full_model.analysis.run_v30_mura_tier_b1_case import create_case, load_checkpoint
from full_model.production.extensive_wall import extensive_wall_energy_components_J_m3
from full_model.production.tensorial_nye import divergence_of_nye, nye_from_plastic_distortion
from full_model.production.wall_topology_supply import reservoir_nye_m1


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def last_history(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    return rows[-1]


def state_record(path, context):
    _, _, _, systems, topologies, _, extensive, _, dx = context
    state, metadata = load_checkpoint(path, systems, topologies)
    zero = np.zeros(state.common.orientation_rad.shape+(3, 3))
    parts = extensive_wall_energy_components_J_m3(
        state.density, systems, topologies, state.common.orientation_rad,
        zero, extensive)
    alpha = np.sum(state.common.family_nye_m1, axis=2)
    curl = nye_from_plastic_distortion(state.common.beta_p, dx)
    reservoir = reservoir_nye_m1(
        state.reservoir_alignment, systems, state.common.orientation_rad,
        topologies)["total"]
    scale = max(float(np.sqrt(np.mean(alpha*alpha))), 1.0)
    div_scale = max(scale/dx, 1.0)
    return state, {
        "path": str(Path(path).resolve()), "sha256": digest(path),
        "step": int(metadata["step"]),
        "strain": float(metadata["applied_strain"]),
        "energy_J_per_m_thickness": {
            name: float(np.sum(value, dtype=np.longdouble)*dx*dx)
            for name, value in parts.items()},
        "authoritative_source_offset_relative_rms": float(
            np.sqrt(np.mean((alpha-curl)**2))/scale),
        "normalized_line_continuity_residual": float(
            np.sqrt(np.mean(divergence_of_nye(alpha, dx)**2))/div_scale),
        "reservoir_to_plastic_nye_relative_rms": float(
            np.sqrt(np.mean((reservoir-alpha)**2))/scale),
    }


def state_equal(left, right):
    differences = {}
    exact = True
    for group in ("common", "density", "reservoir_alignment"):
        a = getattr(left, group); b = getattr(right, group)
        for name in a.__dict__:
            if group == "common" and name == "temperature_K":
                continue
            av = np.asarray(getattr(a, name)); bv = np.asarray(getattr(b, name))
            same = np.array_equal(av, bv)
            exact &= same
            if not same:
                differences[f"{group}.{name}"] = float(np.max(np.abs(av-bv)))
    return bool(exact), differences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-off", type=Path, required=True)
    parser.add_argument("--legacy-on", type=Path, required=True)
    parser.add_argument("--repaired-on", type=Path, required=True)
    parser.add_argument("--legacy-off-history", type=Path, required=True)
    parser.add_argument("--legacy-on-history", type=Path, required=True)
    parser.add_argument("--repaired-on-history", type=Path, required=True)
    parser.add_argument("--first-failure-output", type=Path, required=True)
    parser.add_argument("--repair-output", type=Path, required=True)
    args = parser.parse_args()
    context = create_case(64, "mechanical_heterogeneity", 42, 1e-5)
    off_state, off = state_record(args.legacy_off, context)
    old_state, old = state_record(args.legacy_on, context)
    new_state, new = state_record(args.repaired_on, context)
    off_hist = last_history(args.legacy_off_history)
    old_hist = last_history(args.legacy_on_history)
    new_hist = last_history(args.repaired_on_history)
    exact, differences = state_equal(off_state, new_state)
    temperature_increment = (np.asarray(new_state.common.temperature_K)
                             -np.asarray(off_state.common.temperature_K))
    generated = datetime.now(timezone.utc).isoformat()
    old_delta = {name: old["energy_J_per_m_thickness"][name]
                 -off["energy_J_per_m_thickness"][name]
                 for name in off["energy_J_per_m_thickness"]}
    first = {
        "schema": "asb-drx/v42/topology-first-failure/v1",
        "generated_utc": generated,
        "source_checkpoint": "4b6da0dfbb6f9e8b9a4e009e1eeb39c3b67ca03a",
        "legacy_control": off, "legacy_topology_on": old,
        "legacy_control_metrics": off_hist,
        "legacy_topology_on_metrics": old_hist,
        "legacy_on_minus_off_energy_J_per_m_thickness": old_delta,
        "first_offending_suboperator": "finite_segment_kink_pair_reorientation",
        "independent_following_violation": "explicit_junction_topology",
        "diagnosis": (
            "local moments were rotated and their measured reservoir-Nye change "
            "was injected into family_nye without persistent endpoints or swept "
            "plastic area; complete ordered-gradient energy was not an acceptance input"),
        "classification": "HARD_INVALID_LEGACY_TOPOLOGY_TRANSACTION",
    }
    repair = {
        "schema": "asb-drx/v42/topology-repair-decision/v1",
        "generated_utc": generated,
        "repair_source_commit": "ee4b805e4576af5110301692d72bc4c4d76a68c7",
        "control": off, "repaired_topology_on": new,
        "control_metrics": off_hist, "repaired_metrics": new_hist,
        "repaired_full_state_bitwise_equal_to_control": bool(
            exact and np.array_equal(temperature_increment,
                                     np.zeros_like(temperature_increment))),
        "repaired_nonthermal_state_bitwise_equal_to_control": exact,
        "topology_heat_temperature_increment_K": {
            "minimum": float(np.min(temperature_increment)),
            "maximum": float(np.max(temperature_increment)),
            "mean": float(np.mean(temperature_increment)),
        },
        "nonmatching_arrays": differences,
        "complete_transaction": {
            "represented_operation": "geometry-neutral reservoir conversion",
            "unrepresented_operations_fail_closed": [
                "line reorientation", "junction creation", "endpoint creation",
                "swept plastic area"],
            "actual_discrete_ordered_gradient_energy_in_acceptance": True,
            "failed_candidate_leaves_authoritative_state_and_heat_unchanged": True,
        },
        "classification": (
            "VALID_NEGATIVE_AT_TESTED_CONDITION;LEGACY_APPARENT_ORDER_REJECTED;"
            "NO_QUALIFIED_BOUNDARY" if exact else "REPAIR_MISMATCH_UNRESOLVED"),
    }
    for path, payload in ((args.first_failure_output, first),
                          (args.repair_output, repair)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
