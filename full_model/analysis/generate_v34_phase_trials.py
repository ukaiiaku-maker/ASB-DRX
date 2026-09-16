#!/usr/bin/env python3
"""Generate production-qualified forward/reverse phase trials for I3."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from full_model.analysis.run_v34_finite_coupled_response import resolved_bicrystal
from full_model.production.phase_proposal import (
    adaptive_phase_proposal, phase_structure_diagnostics,
    preserve_existing_pair_components, spectral_phase_energy)
from full_model.production.stored_energy_coupling import (
    common_variational_stored_energy)


def generate_trials(grid, *, length_m=1.0e-5, interface_width_m=3.0e-7,
                    pressure_Pa=2.0e8, dt_s=1.0e-7, mobility=3.0e-3,
                    kappa_J_m=5.0e-7, barrier_J_m3=5.0e6):
    context = resolved_bicrystal(
        grid=grid, length_m=length_m,
        interface_width_m=interface_width_m)
    initial = context["state"].eta.copy()
    spacing = float(context["spacing_m"])
    kx = 2.0*np.pi*np.fft.fftfreq(grid, d=spacing)
    ky = 2.0*np.pi*np.fft.fftfreq(grid, d=spacing)
    k2 = kx[:, None]**2+ky[None, :]**2
    results = {}
    diagnostics = {}
    for direction, signed_pressure in (
            ("forward", pressure_Pa), ("reverse", -pressure_Pa)):
        phase_energy = np.empty_like(initial)
        # child-parent energy difference = -signed pressure
        phase_energy[:, :, 0] = max(float(signed_pressure), 0.0)
        phase_energy[:, :, 1] = max(float(-signed_pressure), 0.0)

        def local_force(eta):
            total_sq = np.sum(eta**2, axis=2)
            _, stored = common_variational_stored_energy(eta, phase_energy)
            return (2.0*barrier_J_m3*eta
                    *(total_sq[:, :, None]-eta**2)+stored)

        def full_force(eta):
            result = local_force(eta)
            for phase in range(2):
                result[:, :, phase] -= kappa_J_m*np.real(np.fft.ifft2(
                    -k2*np.fft.fft2(eta[:, :, phase])))
            return result

        def stored_density(eta):
            return common_variational_stored_energy(eta, phase_energy)[0]

        def energy(eta):
            return spectral_phase_energy(
                eta, spacing_m=spacing, kappa_J_m=kappa_J_m,
                bulk_barrier_J_m3=barrier_J_m3,
                stored_energy_density=stored_density)

        trial, diag = adaptive_phase_proposal(
            initial, dt_s=dt_s, spacing_m=spacing, mobility=mobility,
            kappa_J_m=kappa_J_m, force=full_force,
            local_force=local_force, energy=energy,
            admissibility_projector=lambda before, candidate: (
                preserve_existing_pair_components(
                    before, candidate, parent_label=0, child_label=1)))
        results[direction+"_eta"] = trial
        diagnostics[direction] = diag.to_dict()
        diagnostics[direction].update(phase_structure_diagnostics(
            initial, trial, spacing_m=spacing,
            parent_label=0, child_label=1))
    metadata = {
        "schema": "asb-drx/v34-qualified-phase-trials/v1",
        "created_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"),
        "grid": int(grid), "length_m": float(length_m),
        "interface_width_m": float(interface_width_m),
        "pressure_Pa": float(pressure_Pa), "dt_s": float(dt_s),
        "mobility_m3_J_s": float(mobility),
        "kappa_J_m": float(kappa_J_m),
        "barrier_J_m3": float(barrier_J_m3),
        "diagnostics": diagnostics,
    }
    return initial, results, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pressure-Pa", type=float, default=2.0e8)
    parser.add_argument("--dt-s", type=float, default=1.0e-7)
    args = parser.parse_args()
    initial, trials, metadata = generate_trials(
        args.grid, pressure_Pa=args.pressure_Pa, dt_s=args.dt_s)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, initial_eta=initial,
        forward_eta=trials["forward_eta"],
        reverse_eta=trials["reverse_eta"],
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
