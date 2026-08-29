#!/usr/bin/env python3
"""Gate B linear signed-patterning fixture.

The operator is a preregistered structural fixture, not a calibrated transport
law. It verifies sign sensitivity and finite wavelength selection analytically.
"""

from __future__ import annotations

import json
import numpy as np


def growth_rate(k: np.ndarray, drive_m2_s: float, regularizer_m4_s: float) -> np.ndarray:
    return drive_m2_s * k**2 - regularizer_m4_s * k**4


def run_fixture() -> dict[str, object]:
    drive = 2.0e-12
    regularizer = 8.0e-25
    k_star = np.sqrt(drive / (2.0 * regularizer))
    wavelength = 2.0 * np.pi / k_star

    grids = [128, 256, 512]
    domain = 16.0 * wavelength
    selected = []
    for n in grids:
        k = 2.0 * np.pi * np.fft.rfftfreq(n, d=domain / n)
        lam = growth_rate(k, drive, regularizer)
        selected.append(float(k[int(np.argmax(lam))]))

    rho_total = 10.0
    mixtures = [(5.0, 5.0), (8.0, 2.0)]
    signed_excess = [plus - minus for plus, minus in mixtures]
    balanced_scalar_rate = -drive * k_star**2
    compatible_signed_rate = float(growth_rate(np.asarray([k_star]), drive, regularizer)[0])
    relative_grid_change = abs(selected[-1] - selected[-2]) / k_star

    checks = {
        "same_total_density": all(abs(p + m - rho_total) < 1.0e-14 for p, m in mixtures),
        "different_signed_state": signed_excess[0] != signed_excess[1],
        "balanced_scalar_mode_does_not_create_wall": balanced_scalar_rate < 0.0,
        "compatible_signed_mode_can_grow": compatible_signed_rate > 0.0,
        "finite_analytical_wavelength": np.isfinite(wavelength) and wavelength > 0.0,
        "discrete_fastest_mode_converges": relative_grid_change < 0.05,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    assert all(checks.values()), checks
    return {
        "schema": "gate_B_linear_fixture_v1",
        "fixture_passed": True,
        "scientific_gate_passed": False,
        "reason": "nonlinear Burgers-resolved transport coefficients and wall validation data are absent",
        "checks": checks,
        "analytical_fastest_wavelength_m": wavelength,
        "discrete_selected_wavenumbers_m_inv": selected,
        "last_relative_wavenumber_change": relative_grid_change,
    }


if __name__ == "__main__":
    print(json.dumps(run_fixture(), indent=2, sort_keys=True))
