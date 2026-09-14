#!/usr/bin/env python3
"""Compact local v13 geometry and pinned-cap thermodynamic preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parents[1] / "production"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from arrhenius_kinetics import ActivatedProcess
from sibm_boundary import BoundaryGraphState, SIBMParameters, advance_boundary, select_resolved_hagb
from sibm_geometry import displace_pair_boundary, pinned_cap_energy, validate_post_seed_pair


def bicrystal(grid: int, width_m: float) -> tuple[np.ndarray, float]:
    dx = 10e-6 / grid
    x = (np.arange(grid) - grid / 2 + 0.5)[:, None] * dx
    child = np.broadcast_to(0.5 * (1 + np.tanh(x / (np.sqrt(2) * width_m))),
                            (grid, grid))
    eta = np.zeros((grid, grid, 2))
    eta[..., 0], eta[..., 1] = 1-child, child
    return eta, dx


def graph_matrix() -> dict:
    process = ActivatedProcess("canonical-HAGB", 1e7, 0.0, 1e7)
    def params(mobility=2e-17):
        return SIBMParameters(0.5, mobility, process, 8e-20, 2e8,
                              1.0, 1.0, 0.1, 1e-6)
    def step(height, drive, mobility=2e-17):
        state = BoundaryGraphState(np.asarray(height, float), 0, 1, 0.0, np.pi/6)
        return advance_boundary(state, spacing_m=1e-7, dt_s=1e-5,
                                temperature_K=1100.0, parameters=params(mobility),
                                stored_difference_Pa=drive)[0]
    flat = np.zeros(64)
    equal = step(flat, 0.0)
    favorable = step(flat, 2e7)
    reverse = step(flat, -2e7)
    off = step(flat, 2e7, 0.0)
    bulge = 2e-7*np.cos(np.arange(64)*2*np.pi/64)
    smoothed = step(bulge, 2e7)
    return {
        "flat_equal_stationary": bool(np.array_equal(equal.height_m, flat)),
        "flat_favorable_translates_positive": bool(np.mean(favorable.height_m) > 0),
        "flat_reverse_translates_negative": bool(np.mean(reverse.height_m) < 0),
        "mobility_off_stationary": bool(np.array_equal(off.height_m, flat)),
        "unpinned_bulge_smooths_relative_to_mean": bool(
            np.ptp(smoothed.height_m-np.mean(smoothed.height_m)) < np.ptp(bulge)),
        "energy_ledgers_close": all(abs(s.ledger.energy_closure_J) <= 1e-25
                                    for s in (equal, favorable, reverse, off, smoothed)),
    }


def run(source: Path | None = None) -> dict:
    width = 3.16e-7
    geometry = {}
    for grid in (128, 192):
        eta, dx = bicrystal(grid, width)
        zero = displace_pair_boundary(
            eta, parent=0, child=1, spacing_m=dx, interface_width_m=width,
            centre_index=(grid//2, grid//2), advance_direction_index=(-1, 0),
            amplitude_m=0.0, half_chord_m=1.5e-6,
            active_window_radius_m=3.5e-6)
        seed = displace_pair_boundary(
            eta, parent=0, child=1, spacing_m=dx, interface_width_m=width,
            centre_index=(grid//2, grid//2), advance_direction_index=(-1, 0),
            amplitude_m=0.6e-6, half_chord_m=1.5e-6,
            active_window_radius_m=3.5e-6)
        report = validate_post_seed_pair(
            seed.eta, parent=0, child=1, spacing_m=dx, interface_width_m=width,
            centre_index=(grid//2, grid//2), advance_direction_index=(-1, 0),
            seed_amplitude_m=0.6e-6, half_chord_m=1.5e-6,
            active_window_radius_m=3.5e-6)
        geometry[str(grid)] = {
            "zero_amplitude_bitwise_exact": bool(np.array_equal(zero.eta, eta)),
            "post_seed": report.to_dict(),
            "seed": {key: getattr(seed, key) for key in (
                "amplitude_m", "half_chord_m", "chord_length_m", "arc_length_m",
                "swept_area_m2", "tip_curvature_m_1", "maximum_simplex_error",
                "maximum_nonpair_change")},
        }
    a, b, gamma = 0.6e-6, 1.5e-6, 0.5
    no_drive = pinned_cap_energy(a, b, boundary_energy_J_m2=gamma,
                                 stored_pressure_Pa=0.0)
    critical = gamma*no_drive["d_arc_length_da"]/no_drive["d_swept_area_da_m"]
    sub = pinned_cap_energy(a, b, boundary_energy_J_m2=gamma,
                            stored_pressure_Pa=0.8*critical)
    sup = pinned_cap_energy(a, b, boundary_energy_J_m2=gamma,
                            stored_pressure_Pa=1.2*critical)
    h = 1e-10
    fd = (pinned_cap_energy(a+h, b, boundary_energy_J_m2=gamma,
                            stored_pressure_Pa=critical)["total_energy_J"]
          - pinned_cap_energy(a-h, b, boundary_energy_J_m2=gamma,
                              stored_pressure_Pa=critical)["total_energy_J"])/(2*h)
    source_audit = None
    if source:
        with np.load(source, allow_pickle=True) as state:
            eta = state["eta"]; rho = state["rho"]; angles = state["psi_gv"]
        selected = select_resolved_hagb(eta, angles, rho, purity_threshold=0.8,
                                        min_pure_core_cells=16,
                                        min_misorientation_deg=15.0)
        dx = 10e-6/eta.shape[0]
        seed = displace_pair_boundary(
            eta, parent=selected.parent_label, child=selected.child_label,
            spacing_m=dx, interface_width_m=width,
            centre_index=selected.centre_index,
            advance_direction_index=selected.advance_direction_index,
            amplitude_m=0.75e-6, half_chord_m=1.5e-6,
            active_window_radius_m=3e-6)
        source_audit = validate_post_seed_pair(
            seed.eta, parent=selected.parent_label, child=selected.child_label,
            spacing_m=dx, interface_width_m=width,
            centre_index=selected.centre_index,
            advance_direction_index=selected.advance_direction_index,
            seed_amplitude_m=0.75e-6, half_chord_m=1.5e-6,
            active_window_radius_m=3e-6).to_dict()
    graph = graph_matrix()
    passed = (all(item["zero_amplitude_bitwise_exact"] and item["post_seed"]["valid"]
                  for item in geometry.values())
              and sub["d_total_energy_da_J_m"] > 0
              and sup["d_total_energy_da_J_m"] < 0
              and abs(fd) < 1e-7
              and all(graph.values()))
    return {
        "schema": "full-v34-v13-valid-pair-preflight/v1",
        "classification": ("CANONICAL_PINNED_CAP_PREFLIGHT_PASSED" if passed
                           else "CANONICAL_PINNED_CAP_PREFLIGHT_FAILED"),
        "full_model_sibm_qualified": False,
        "canonical_geometry": geometry,
        "pinned_cap_criticality": {
            "amplitude_m": a, "half_chord_m": b,
            "critical_stored_pressure_Pa": critical,
            "subcritical": sub, "supercritical": sup,
            "finite_difference_derivative_at_critical_J_m": fd,
        },
        "flat_and_unpinned_matrix": graph,
        "retired_v12_source_post_seed": source_audit,
        "next_requirement": "full-driver canonical bicrystal and compact five-case HPC requalification",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    result = run(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(result["classification"])


if __name__ == "__main__":
    main()
