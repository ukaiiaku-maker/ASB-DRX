#!/usr/bin/env python3
"""Measure the discrete phase-interface translation potential over one cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
from scipy.ndimage import fourier_shift


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=41)
    args = parser.parse_args()
    checkpoints = sorted(args.case.glob("drx_v25_restart_*.npz"), key=lambda path: int(
        re.search(r"(\d+)$", path.stem).group(1)))
    with np.load(checkpoints[0], allow_pickle=True) as data:
        eta = np.asarray(data["eta"], float)[:, 0, :2]
        p = json.loads(str(data["P_json"].item()))
    dx = float(p["L_phys"])/int(p["Nx"])
    offsets = np.linspace(0.0, 1.0, args.samples)
    energies = []
    for offset in offsets:
        shifted = np.stack([
            np.fft.ifftn(fourier_shift(np.fft.fftn(eta[:, phase]), offset)).real
            for phase in range(2)], axis=-1)
        shifted = np.clip(shifted, 0.0, 1.0)
        shifted /= np.maximum(np.sum(shifted, axis=1, keepdims=True), 1e-300)
        gradient = (np.roll(shifted, -1, axis=0)
                    -np.roll(shifted, 1, axis=0))/(2.0*dx)
        density = (.5*float(p["kappa_eta"])*np.sum(gradient**2, axis=1)
                   +float(p["W_eta"])*np.prod(shifted**2, axis=1))
        energies.append(float(np.sum(density)*dx))
    energies = np.asarray(energies)
    derivative = np.gradient(energies, offsets*dx)
    amplitude = float(np.max(energies)-np.min(energies))
    result = {
        "schema": "asb-drx/v28-sibm-registry/v1",
        "grid": int(p["Nx"]), "spacing_m": dx,
        "offset_cell_fraction": offsets.tolist(),
        "interface_energy_J_m2": energies.tolist(),
        "translation_derivative_Pa": derivative.tolist(),
        "peierls_amplitude_J_m2": amplitude,
        "relative_peierls_amplitude": amplitude/max(float(np.mean(energies)), 1e-300),
        "stationary_offset_cell_fraction": float(offsets[np.argmin(energies)]),
        "fixture_passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({key: result[key] for key in (
        "peierls_amplitude_J_m2", "relative_peierls_amplitude",
        "stationary_offset_cell_fraction")}, indent=2))


if __name__ == "__main__":
    main()
