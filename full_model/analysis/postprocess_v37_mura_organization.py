#!/usr/bin/env python3
"""Classify matched one-grain V37 Mura organization controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from full_model.analysis.run_v30_mura_tier_b1_case import create_case, load_checkpoint
from full_model.production.density_state_map import SIGNED_RESERVOIRS, derived_density_fields
from full_model.production.extensive_wall import extensive_wall_energy_components_J_m3
from full_model.production.wall_topology_supply import reservoir_nye_m1


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def last_checkpoint(case_dir):
    return sorted(Path(case_dir).glob("checkpoint_step_*.npz"))[-1]


def summarize(case_dir):
    case_dir = Path(case_dir)
    config = json.loads((case_dir/"case_config.json").read_text())
    status = json.loads((case_dir/"status.json").read_text())
    history = [json.loads(line) for line in (case_dir/"history.jsonl").read_text(
    ).splitlines() if line.strip()]
    (_, _, _, systems, topologies, _, extensive, _, spacing) = create_case(
        config["grid"], config["condition"], config["seed"], config["length_m"])
    checkpoint = last_checkpoint(case_dir)
    state, metadata = load_checkpoint(checkpoint, systems, topologies)
    fields = derived_density_fields(state.density, topologies)
    nye = reservoir_nye_m1(
        state.reservoir_alignment, systems, state.common.orientation_rad,
        topologies)
    energy = extensive_wall_energy_components_J_m3(
        state.density, systems, topologies, state.common.orientation_rad,
        np.zeros_like(nye["total"]), extensive)
    area = spacing**2
    inventories = {
        name: float(np.sum(getattr(state.density, name))*area)
        for name in SIGNED_RESERVOIRS}
    inventories["junction_m2"] = float(np.sum(state.density.junction_m2)*area)
    hard = all(row["accepted_step_hard_invariant_passed"]
               and not row["post_step_projection_used"]
               and row["minimum_heat_increment_J_m3"] >= 0.0
               for row in history)
    final = history[-1]
    orientation_span = final["orientation_span_deg"]
    ordered = final["ordered_line_m2_cells"]
    wall = final["total_wall_line_m2_cells"]
    if wall == 0.0:
        classification = "NO_CAPTURED_WALL_LINE"
    elif orientation_span < 1.0:
        classification = "CAPTURED_ORDERED_LINE_WITHOUT_ORIENTATION_WALL"
    else:
        classification = "ORIENTATION_WALL_REQUIRES_PERSISTENCE_AUDIT"
    return {
        "case_dir": str(case_dir),
        "condition": config["condition"],
        "topology_route_enabled": config["topology_route_enabled"],
        "source_sha": config["source_sha"],
        "status": status["status"],
        "grid": config["grid"], "spacing_m": spacing,
        "initial_strain": config["initial_strain"],
        "final_strain": status["applied_strain"],
        "strain_exposure": status["applied_strain"]-config["initial_strain"],
        "accepted_intervals": status["step"],
        "wall_seconds": status["wall_seconds_this_invocation"],
        "seconds_per_accepted_interval": (
            status["wall_seconds_this_invocation"]/max(status["step"], 1)),
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256(checkpoint),
        "history_sha256": sha256(case_dir/"history.jsonl"),
        "hard_invariants_passed": hard,
        "no_phase_or_grain_state_present": not (
            hasattr(state, "phase") or hasattr(state, "grain")),
        "classification": classification,
        "ordered_line_m2_cells": ordered,
        "total_wall_line_m2_cells": wall,
        "ordered_fraction_wall_local": final["ordered_fraction_wall_local"],
        "ordered_polarization_wall_local_mean": final[
            "ordered_polarization_wall_local_mean"],
        "orientation_span_deg": orientation_span,
        "maximum_signed_density_m2": final["maximum_signed_density_m2"],
        "dominant_signed_wavelength_m": final["dominant_signed_wavelength_m"],
        "structure_factor_peak_fraction": final[
            "structure_factor_peak_fraction"],
        "dual_nye_relative_rms": final["dual_nye_relative_rms"],
        "normalized_line_continuity_residual": final[
            "normalized_line_continuity_residual"],
        "integrated_inventory_m_per_m_thickness": inventories,
        "energy_J_per_m_thickness": {
            name: float(np.sum(value)*area) for name, value in energy.items()},
        "metadata_source_sha": metadata.get("source_sha", "UNRECORDED"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [summarize(path) for path in args.case_dir]
    paired = []
    for condition in sorted(set(case["condition"] for case in cases)):
        pair = [case for case in cases if case["condition"] == condition]
        off = next(case for case in pair if not case["topology_route_enabled"])
        on = next(case for case in pair if case["topology_route_enabled"])
        paired.append({
            "condition": condition,
            "topology_on_minus_off": {
                key: on[key]-off[key] for key in (
                    "ordered_line_m2_cells", "total_wall_line_m2_cells",
                    "ordered_fraction_wall_local", "orientation_span_deg",
                    "maximum_signed_density_m2")},
            "states_identical_in_selected_observables": all(
                on[key] == off[key] for key in (
                    "ordered_line_m2_cells", "total_wall_line_m2_cells",
                    "orientation_span_deg", "maximum_signed_density_m2")),
        })
    hard = all(case["hard_invariants_passed"] for case in cases)
    result = {
        "schema": "asb-drx/v37-mura-organization/v1",
        "analysis_sha256": sha256(Path(__file__)),
        "fixture_passed": hard and all(case["status"] == "COMPLETED"
                                       for case in cases),
        "scientific_gate_passed": False,
        "scientific_gate_not_passed_reason": (
            "matched controls have only 0.02 percent strain exposure and do "
            "not form an orientation-compatible persistent wall"),
        "claim_boundary": (
            "one-grain organization controls only; no phase support, grain "
            "label, nucleation, DRX, or intrinsic wall spacing is claimed"),
        "cases": cases,
        "topology_pairs": paired,
        "selected_long_continuation": {
            "condition": "mechanical_heterogeneity",
            "topology_route_enabled": False,
            "reason": (
                "one-grain physical organization route with capture support; "
                "the explicit topology route remains the existing-boundary "
                "comparator and must not globally veto organization"),
            "launch_state": "AWAITING_COORDINATION_BEFORE_HPC_SUBMISSION",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"output": str(args.output),
                      "fixture_passed": result["fixture_passed"],
                      "cases": len(cases)}, sort_keys=True))


if __name__ == "__main__":
    main()
