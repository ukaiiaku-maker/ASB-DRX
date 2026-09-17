#!/usr/bin/env python3
"""Audit the physical size and timestep sensitivity of ordered reservoirs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from full_model.analysis.run_v36_mura_rate_limit import advance, context, state_difference
from full_model.production.density_state_map import derived_density_fields
from full_model.production.extensive_wall import extensive_wall_energy_components_J_m3
from full_model.production.wall_topology_supply import reservoir_nye_m1


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_observables(state, systems, topologies, extensive, spacing_m):
    fields = derived_density_fields(state.density, topologies)
    wall = fields["rho_wall_m2"]
    ordered = fields["rho_wall_ordered_m2"]
    threshold = max(float(np.quantile(wall, .9)), 1e12)
    mask = wall >= threshold
    cell_area = spacing_m**2
    nye = reservoir_nye_m1(
        state.reservoir_alignment, systems, state.common.orientation_rad,
        topologies)
    ordered_nye = nye["wall_ordered"]
    bmean = float(np.mean([system.burgers_m for system in systems]))
    polarization = np.linalg.norm(ordered_nye, axis=(-2, -1))/np.maximum(
        bmean*ordered, 1e-300)
    energies = extensive_wall_energy_components_J_m3(
        state.density, systems, topologies, state.common.orientation_rad,
        np.zeros_like(ordered_nye), extensive)
    moment_absolute = sum(float(np.sum(np.linalg.norm(
        getattr(state.reservoir_alignment, f"wall_ordered_{sign}_m2"),
        axis=-1))*cell_area) for sign in ("plus", "minus"))
    return {
        "wall_local_threshold_m2": threshold,
        "wall_local_cell_count": int(np.count_nonzero(mask)),
        "ordered_line_integral_m_per_m_thickness": float(
            np.sum(ordered)*cell_area),
        "total_wall_line_integral_m_per_m_thickness": float(
            np.sum(wall)*cell_area),
        "ordered_fraction_global": float(
            np.sum(ordered)/max(float(np.sum(wall)), 1e-300)),
        "ordered_fraction_wall_local": float(
            np.sum(ordered[mask])/max(float(np.sum(wall[mask])), 1e-300)),
        "ordered_moment_absolute_integral_m_per_m_thickness": moment_absolute,
        "ordered_polarization_maximum": float(np.max(polarization)),
        "ordered_polarization_wall_local_mean": float(
            np.mean(polarization[mask]) if np.any(mask) else 0.0),
        "ordered_boundary_energy_J_m3_cells": float(np.sum(
            energies["ordered_boundary"])),
        "ordered_boundary_energy_J_per_m_thickness": float(np.sum(
            energies["ordered_boundary"])*cell_area),
        "complete_extensive_defect_energy_J_per_m_thickness": float(
            np.sum(energies["total"])*cell_area),
        "ordered_energy_fraction_absolute": float(np.sum(np.abs(
            energies["ordered_boundary"]))/max(float(np.sum(np.abs(
                energies["total"]))), 1e-300)),
    }


def ordered_error(left, right, systems, topologies, extensive, spacing_m):
    cell_area = spacing_m**2
    ld = left.density; rd = right.density
    line_error = sum(float(np.sum(np.abs(
        getattr(ld, f"wall_ordered_{sign}_m2")
        -getattr(rd, f"wall_ordered_{sign}_m2")))*cell_area)
        for sign in ("plus", "minus"))
    moment_error = sum(float(np.sum(np.linalg.norm(
        getattr(left.reservoir_alignment, f"wall_ordered_{sign}_m2")
        -getattr(right.reservoir_alignment, f"wall_ordered_{sign}_m2"),
        axis=-1))*cell_area) for sign in ("plus", "minus"))
    lf = derived_density_fields(ld, topologies)
    rf = derived_density_fields(rd, topologies)
    wall_ref = np.maximum(lf["rho_wall_m2"], rf["rho_wall_m2"])
    mask = wall_ref >= max(float(np.quantile(wall_ref, .9)), 1e12)
    ordered_difference = np.abs(lf["rho_wall_ordered_m2"]
                                -rf["rho_wall_ordered_m2"])
    wall_local_error = float(np.sum(ordered_difference[mask])*cell_area)
    wall_local_line = float(np.sum(wall_ref[mask])*cell_area)
    lo = ordered_observables(left, systems, topologies, extensive, spacing_m)
    ro = ordered_observables(right, systems, topologies, extensive, spacing_m)
    return {
        "absolute_ordered_line_error_m_per_m_thickness": line_error,
        "absolute_ordered_moment_error_m_per_m_thickness": moment_error,
        "wall_local_ordered_error_over_total_wall_line": (
            wall_local_error/max(wall_local_line, 1e-300)),
        "ordered_energy_difference_J_per_m_thickness": abs(
            lo["ordered_boundary_energy_J_per_m_thickness"]
            -ro["ordered_boundary_energy_J_per_m_thickness"]),
        "coarser": lo,
        "finer": ro,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon-s", type=float, default=4e-10)
    parser.add_argument("--dt-s", type=float, nargs="+",
                        default=(2e-9, 1e-10, 5e-11))
    args = parser.parse_args()
    (_, metadata, _, _, systems, topologies, _, extensive, _, spacing) = (
        context(args.checkpoint))
    states = []; records = []
    for dt_s in args.dt_s:
        state, record = advance(args.checkpoint, dt_s, 12, args.horizon_s)
        states.append(state)
        record["ordered_observables"] = ordered_observables(
            state, systems, topologies, extensive, spacing)
        records.append(record)
    comparisons = []
    for index in range(len(states)-1):
        comparisons.append({
            "coarser_dt_s": args.dt_s[index],
            "finer_dt_s": args.dt_s[index+1],
            "all_state_error": state_difference(states[index], states[index+1]),
            "ordered_reservoir_error": ordered_error(
                states[index], states[index+1], systems, topologies,
                extensive, spacing),
        })
    finest = records[-1]["ordered_observables"]
    result = {
        "schema": "asb-drx/v37-mura-ordered-audit/v1",
        "source_sha": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True).strip(),
        "script_sha256": sha256(Path(__file__)),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_source_sha": metadata.get("source_sha", "UNRECORDED"),
        "records": records,
        "comparisons": comparisons,
        "hard_invariants_passed": all(record["hard_invariants_passed"]
                                      for record in records),
        "ordered_reservoir_physically_negligible_for_total_energy": bool(
            finest["ordered_energy_fraction_absolute"] < 1e-4),
        "wall_formation_supported": False,
        "wall_formation_not_supported_reason": (
            "short rate-limit audit does not establish a persistent, "
            "polarized, orientation-compatible wall"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"output": str(args.output),
                      "hard_invariants_passed": result[
                          "hard_invariants_passed"]}, sort_keys=True))


if __name__ == "__main__":
    main()
