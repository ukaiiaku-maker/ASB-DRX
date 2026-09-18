#!/usr/bin/env python3
"""Discrete flat-interface translation/growth/profile-energy audit for V39."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np

from full_model.production.tensorial_nye import spectral_derivatives


SCHEMA = "asb-drx/v39/interface-residual-audit/v1"


def phase_profile(grid, length_m, width_m, translation_m=0.0,
                  half_slab_change_m=0.0):
    spacing = length_m/grid
    x = np.arange(grid)*spacing
    left = .25*length_m+translation_m-half_slab_change_m
    right = .75*length_m+translation_m+half_slab_change_m
    # Periodic images are negligible at the declared separation/width; this
    # is exactly the production bicrystal initialization profile.
    fraction = .5*(np.tanh((x-left)/width_m)
                   -np.tanh((x-right)/width_m))
    chi = np.broadcast_to(np.clip(fraction[:, None], 0.0, 1.0),
                          (grid, grid)).copy()
    return np.stack((1.0-chi, chi), axis=2)


def phase_energy_J(eta, spacing_m, thickness_m, barrier_J_m3=5e6,
                   gradient_J_m=5e-7):
    sum_e2 = np.sum(eta*eta, axis=2)
    sum_e4 = np.sum(eta**4, axis=2)
    local = .5*barrier_J_m3*np.maximum(sum_e2*sum_e2-sum_e4, 0.0)
    volume = spacing_m*spacing_m*thickness_m
    local_J = float(np.sum(local, dtype=np.longdouble)*volume)
    gradient_J = 0.0
    for index in range(eta.shape[2]):
        gx, gy = spectral_derivatives(eta[..., index], spacing_m)
        gradient_J += float(.5*gradient_J_m*np.sum(
            gx*gx+gy*gy, dtype=np.longdouble)*volume)
    return {"phase_local_J": local_J, "phase_gradient_J": gradient_J,
            "phase_total_J": local_J+gradient_J}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length-um", type=float, default=3.2)
    parser.add_argument("--interface-width-um", type=float, default=.4)
    parser.add_argument("--grid", type=int, nargs="+", default=(32, 64, 128))
    args = parser.parse_args()
    length = args.length_um*1e-6; width = args.interface_width_um*1e-6
    thickness = 2.0*2.48e-10
    rows = []
    for grid in args.grid:
        spacing = length/grid
        offsets = []
        for fraction in (0.0, .25, .5, .75):
            offset = fraction*spacing
            eta = phase_profile(grid, length, width, translation_m=offset)
            offsets.append({"offset_cells": fraction,
                            "offset_m": offset,
                            **phase_energy_J(eta, spacing, thickness)})
        delta = .01*spacing
        translated_minus = phase_energy_J(phase_profile(
            grid, length, width, translation_m=-delta), spacing, thickness)
        translated_plus = phase_energy_J(phase_profile(
            grid, length, width, translation_m=delta), spacing, thickness)
        growth_minus = phase_energy_J(phase_profile(
            grid, length, width, half_slab_change_m=-delta), spacing, thickness)
        growth_plus = phase_energy_J(phase_profile(
            grid, length, width, half_slab_change_m=delta), spacing, thickness)
        narrower = phase_energy_J(phase_profile(
            grid, length, .95*width), spacing, thickness)
        baseline = phase_energy_J(phase_profile(
            grid, length, width), spacing, thickness)
        wider = phase_energy_J(phase_profile(
            grid, length, 1.05*width), spacing, thickness)
        energies = np.asarray([item["phase_total_J"] for item in offsets])
        rows.append({
            "grid": grid, "spacing_m": spacing,
            "interface_width_m": width,
            "interface_cells": width/spacing,
            "subcell_offsets": offsets,
            "translation_energy_range_J": float(np.max(energies)-np.min(energies)),
            "translation_energy_range_relative": float(
                (np.max(energies)-np.min(energies))/max(abs(np.mean(energies)), 1e-300)),
            "central_translation_derivative_J_m": float(
                (translated_plus["phase_total_J"]
                 -translated_minus["phase_total_J"])/(2*delta)),
            "central_slab_growth_derivative_J_m": float(
                (growth_plus["phase_total_J"]
                 -growth_minus["phase_total_J"])/(2*delta)),
            "profile_width_energy_J": {"0.95": narrower["phase_total_J"],
                                       "1.00": baseline["phase_total_J"],
                                       "1.05": wider["phase_total_J"]},
        })
    result = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "length_m": length, "interface_width_m": width,
        "phase_barrier_J_m3": 5e6, "phase_gradient_J_m": 5e-7,
        "rows": rows,
        "scope": "phase-only actual discrete production energy",
        "equal_state_projection_used": False,
        "fitted_drift_subtracted": False,
        "artificial_pressure_added": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
