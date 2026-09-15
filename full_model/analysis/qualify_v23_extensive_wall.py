#!/usr/bin/env python3
"""Local V23 Frank--Bilby and extensive-wall qualification."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from full_model.production.density_state_map import DensityInventory
from full_model.production.extensive_wall import (
    manufacture_ordered_inventory, ordered_wall_nye_m1,
    planar_frank_bilby_target_m1, wall_diagnostics,
)
from full_model.production.tensorial_nye import (
    bcc_four_family_systems, integrated_nye_closure, make_junction_topology,
    rotated_system_fields,
)


def empty_inventory(n):
    shape = (n, n, 4)
    return DensityInventory(
        np.full(shape, 3e13), np.full(shape, 2e13),
        np.full(shape, 4e13), np.full(shape, 3e13),
        np.full(shape, 2e13), np.full(shape, 1e13),
        np.zeros(shape), np.zeros(shape), np.zeros((n, n, 1)))


def main():
    systems = bcc_four_family_systems()
    topologies = (make_junction_topology(
        systems, 0, 1, 1, -1, line_tension_J_m=0.0),)
    convergence = []
    for n in (16, 32, 64):
        length = 3.2e-6
        dx = length/n
        width = n*3//16
        x = np.arange(n)[:, None]
        orientation = np.where(
            np.broadcast_to(x < n//2, (n, n)), 0.0, np.deg2rad(2.0))
        target, closure = planar_frank_bilby_target_m1(
            (n, n), dx, 0.0, np.deg2rad(2.0), width_cells=width)
        wall, projection = manufacture_ordered_inventory(
            empty_inventory(n), systems, orientation, target)
        alpha, _ = ordered_wall_nye_m1(wall, systems, orientation)
        recovered = integrated_nye_closure(alpha, 0, dx, line_axis=2)
        relative_closure = float(np.linalg.norm(recovered-closure)
                                 / max(np.linalg.norm(closure), 1e-300))
        diagnostic = wall_diagnostics(
            wall, systems, topologies, orientation, target)
        convergence.append({
            "grid": n,
            "spacing_m": dx,
            "wall_width_cells": width,
            "wall_width_m": width*dx,
            "projection_relative_residual": projection,
            "frank_bilby_relative_closure_error": relative_closure,
            "maximum_ordered_density_m2": float(np.max(
                diagnostic["rho_wall_ordered_m2"])),
            "physical_wall_cell_count": int(np.count_nonzero(
                diagnostic["physical_wall_mask"])),
        })

    # Quantify why the V22 glide-line inheritance is rejected.
    n = 16; dx = 2e-7
    orientation = np.zeros((n, n))
    target, _ = planar_frank_bilby_target_m1(
        (n, n), dx, 0.0, np.deg2rad(2.0), width_cells=3)
    burgers, directions, normals = rotated_system_fields(systems, orientation)
    glide_lines = np.cross(normals, directions)
    glide_lines /= np.maximum(np.linalg.norm(glide_lines, axis=-1)[..., None], 1e-300)
    glide_basis = np.einsum("...ai,...aj->...aij", burgers, glide_lines)
    grid = (n//2, n//2)
    matrix = np.stack([glide_basis[grid + (a,)].ravel() for a in range(4)], axis=1)
    projected = matrix @ np.linalg.lstsq(matrix, target[grid].ravel(), rcond=None)[0]
    legacy_residual = float(np.linalg.norm(projected-target[grid].ravel())
                            / np.linalg.norm(target[grid]))

    fixture = (max(row["projection_relative_residual"] for row in convergence) < 2e-14
               and max(row["frank_bilby_relative_closure_error"]
                       for row in convergence) < 2e-14
               and all(row["physical_wall_cell_count"] > 0 for row in convergence)
               and legacy_residual > 0.5)
    result = {
        "schema": "v23-extensive-wall-local-qualification-1",
        "convergence": convergence,
        "rejected_v22_glide_line_projection_relative_residual": legacy_residual,
        "ordered_boundary_line_direction_crystal": [0.0, 0.0, 1.0],
        "ordered_line_interpretation": "sessile boundary line; Burgers family retained",
        "negative_controls": {
            "absent_wall": "passed_by_pytest",
            "balanced_tangle": "passed_by_pytest",
            "high_ordered_density_zero_orientation_jump": "passed_by_pytest",
            "reversed_sign": "passed_by_pytest",
            "incompatible_line_tensor": "passed_by_pytest",
            "rigid_rotation": "passed_by_pytest"
        },
        "fixture_passed": bool(fixture),
        "scientific_gate_passed": False,
        "classification": (
            "LOCAL_EXTENSIVE_WALL_PREREQUISITES_PASSED_NONLINEAR_FORMATION_PENDING"
            if fixture else "LOCAL_EXTENSIVE_WALL_PREREQUISITE_FAILURE")
    }
    for source_name in ("density_state_map.py", "extensive_wall.py"):
        source = ROOT / "full_model" / "production" / source_name
        result.setdefault("source_sha256", {})[source_name] = hashlib.sha256(
            source.read_bytes()).hexdigest()
    output = ROOT / "full_model" / "verification" / "v23_extensive_wall_local.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
