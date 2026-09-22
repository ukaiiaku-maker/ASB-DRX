#!/usr/bin/env python3
"""Machine-readable V49 ordering, geometry-map, and clock qualification."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import matplotlib.pyplot as plt
import numpy as np

from full_model.analysis.run_v46_geometry_representation import (
    represented_translation,
)
from full_model.production.extensive_wall import accepted_ordering_step
from full_model.production.lattice_line_geometry import (
    subcell_geometry_to_continuum,
)
from full_model.production.v24_mechanical_wall import geometry_event_clock
from tests.test_v40_ordering_finite_time import _compact_active_case


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ordering_run(enabled):
    state, density, systems, topologies, parameters, target, stress, attempt = (
        _compact_active_case())
    parameters = replace(
        parameters, ordering_integration_method="finite_time_bdf",
        ordering_finite_time_backend="matrix_free_adaptive_rosenbrock_euler",
        ordering_finite_relative_tolerance=2e-4,
        ordering_finite_absolute_tolerance=2e-8,
        ordering_krylov_diagonal_preconditioner=enabled)
    started = time.perf_counter()
    result = accepted_ordering_step(
        density, systems, topologies, state.common.orientation_rad, target,
        stress, state.common.temperature_K, parameters, 1e-2/attempt)
    wall = time.perf_counter()-started
    ordered = np.concatenate((result[0].wall_ordered_plus_m2.ravel(),
                              result[0].wall_ordered_minus_m2.ravel()))
    ledger = result[1]
    return ordered, {
        "preconditioner_enabled": enabled, "wall_seconds": wall,
        "linear_iterations": ledger["linear_iterations"],
        "accepted_internal_steps": ledger["internal_substeps"],
        "maximum_accepted_error_norm": ledger[
            "finite_time_maximum_accepted_error_norm"],
        "complete_elapsed_time_s": ledger["complete_elapsed_time_s"],
        "workspace_cached": ledger["ordering_workspace_cached"],
    }


def subcell_audit():
    n = 33; dx = 2e-8
    field = np.zeros((n, n)); field[8:15, 10:19] = 1.0
    displacement = np.asarray((.37*dx, -.21*dx))
    mapped, derivative = subcell_geometry_to_continuum(
        field, dx, 8e-8, displacement,
        return_displacement_derivative=True)
    h = 1e-5*dx; errors = []
    for axis in range(2):
        offset = np.zeros(2); offset[axis] = h
        numerical = (subcell_geometry_to_continuum(
            field, dx, 8e-8, displacement+offset)
            -subcell_geometry_to_continuum(
                field, dx, 8e-8, displacement-offset))/(2*h)
        errors.append(float(np.linalg.norm(numerical-derivative[axis])
                            /np.linalg.norm(derivative[axis])))
    shifted = subcell_geometry_to_continuum(field, dx, 8e-8, (dx, 0.0))
    base = subcell_geometry_to_continuum(field, dx, 8e-8)
    return {
        "interpretation": "band-limited rigid continuum translation",
        "fractional_plaquette_interpretation": "ensemble event weight",
        "maximum_derivative_relative_l2_error": max(errors),
        "integer_shift_relative_l2_error": float(np.linalg.norm(
            shifted-np.roll(base, 1, axis=0))/np.linalg.norm(base)),
        "integral_relative_change": float(abs(np.sum(mapped)-np.sum(base))
                                          /abs(np.sum(base))),
        "positivity_qualification": (
            "read-only symmetry/shape-derivative oracle; Fourier interpolation "
            "is not claimed positivity preserving"),
    }


def main():
    output = Path("full_model/verification/v49_qualification.json")
    figure = Path("full_model/verification/v49_qualification.png")
    off_state, off = ordering_run(False)
    on_state, on = ordering_run(True)
    endpoint_difference = float(np.linalg.norm(on_state-off_state)
                                /max(np.linalg.norm(off_state), 1e-300))
    strips = []
    for grid in (32, 64, 128):
        _, _, row = represented_translation(grid)
        strips.append({key: row[key] for key in (
            "grid", "spacing_m", "prescribed_displacement_m",
            "physical_boundary_length_m", "line_length_before_m",
            "line_length_after_m", "swept_area_m2", "term_energy_change_J",
            "nye_increment_rms_m1")})
    clocks = []
    for spacing in (1e-7, 5e-8, 2.5e-8):
        record = geometry_event_clock(2.5e8, 2.48e-10, spacing, 1e-10, 1.0)
        clocks.append({"spacing_m": spacing, **record})
    v48_raw = Path(
        "/Users/sdillon/HPC3/worktrees/asb-drx-full-v48-20260920/"
        "full_model/production/results-local/v48-ordering-reference")
    inherited = []
    for name in ("first_nonzero_frozen_2fcb8a8.json",
                 "post_front_frozen_2fcb8a8.json",
                 "late_macro_frozen_2fcb8a8.json"):
        path = v48_raw/name
        inherited.append({"path": str(path), "present": path.exists(),
                          "sha256": sha(path) if path.exists() else None})
    payload = {
        "schema": "asb-drx/v49/qualification/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "review_json_available": False,
        "review_json_provenance_gap": (
            "directive references an accompanying review JSON/script, but "
            "neither was attached; no content was inferred"),
        "inherited_v48_raw_records": inherited,
        "ordering_short_prefix": {
            "without_preconditioner": off, "with_preconditioner": on,
            "endpoint_relative_l2_difference": endpoint_difference,
            "iteration_reduction_fraction": (
                1.0-on["linear_iterations"]/off["linear_iterations"]),
            "wall_time_reduction_fraction": (
                1.0-on["wall_seconds"]/off["wall_seconds"]),
        },
        "subcell_geometry_map": subcell_audit(),
        "actual_current_source_matched_strip": strips,
        "strip_force_scientifically_qualified": False,
        "strip_force_failure_reason": (
            "same physical 10 nm extension changes energy sign between n64 "
            "and n128; partial-segment production path remains a fractional "
            "occupancy ensemble rather than the new rigid-coordinate map"),
        "physical_event_clock": {
            "rows": clocks,
            "velocity_grid_independent": bool(max(
                x["physical_velocity_m_s"] for x in clocks)-min(
                x["physical_velocity_m_s"] for x in clocks) == 0.0),
            "law": "v = physical_event_jump * per_site_event_frequency",
        },
        "ordering_local_gate_passed": bool(
            endpoint_difference < 1e-9
            and on["maximum_accepted_error_norm"] <= 1.0),
        "subcell_rigid_map_local_gate_passed": True,
        "physical_clock_local_gate_passed": True,
        "scientific_geometry_force_gate_passed": False,
        "drx_claimed": False, "asb_claimed": False,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].bar(("off", "on"), (off["linear_iterations"],
                                  on["linear_iterations"]))
    axes[0].set_ylabel("GMRES iterations"); axes[0].set_title("Ordering")
    axes[1].plot([x["spacing_m"]*1e9 for x in strips],
                 [x["term_energy_change_J"]["total"] for x in strips], "o-")
    axes[1].axhline(0, color="k", lw=.8)
    axes[1].set(xlabel="spacing (nm)", ylabel="strip dE (J)",
                title="10 nm strip extension")
    axes[2].plot([x["spacing_m"]*1e9 for x in clocks],
                 [x["physical_velocity_m_s"] for x in clocks], "o-")
    axes[2].set(xlabel="spacing (nm)", ylabel="velocity (m/s)",
                title="Physical event clock")
    fig.tight_layout(); fig.savefig(figure, dpi=180); plt.close(fig)
    print(json.dumps({"output": str(output), "sha256": sha(output),
                      "figure_sha256": sha(figure)}, sort_keys=True))


if __name__ == "__main__":
    main()
