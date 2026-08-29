#!/usr/bin/env python3
"""Gate D static Frank--Bilby and rotation-ablation fixture."""

from __future__ import annotations

import json
import math
import numpy as np


def rotation(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.asarray([[c, -s], [s, c]])


def frank_bilby(theta: float, tangent: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    relative = rotation(theta)
    target = (np.eye(2) - relative.T) @ tangent
    coefficients, *_ = np.linalg.lstsq(basis, target, rcond=None)
    reconstructed = basis @ coefficients
    scale = max(float(np.linalg.norm(target)), np.finfo(float).eps)
    residual = float(np.linalg.norm(target - reconstructed) / scale)
    return target, coefficients, residual


def read_shockley(theta: float, theta_m: float, gamma_m: float) -> float:
    if theta <= 0.0:
        return 0.0
    x = theta / theta_m
    return gamma_m * x * (1.0 - math.log(x)) if x <= 1.0 else gamma_m


def boundary_rotation_rate(mobility: float, shear_coupling: float, driving: float, radius: float) -> float:
    return mobility * shear_coupling * driving / radius


def run_fixture() -> dict[str, object]:
    theta = math.radians(7.0)
    tangent = np.asarray([1.0, 0.0])
    basis = np.eye(2)
    target, coefficients, residual = frank_bilby(theta, tangent, basis)
    energy = read_shockley(theta, math.radians(15.0), 0.65)
    zero_energy = read_shockley(0.0, math.radians(15.0), 0.65)
    coupled_rate = boundary_rotation_rate(2.0, 0.4, 3.0, 5.0)
    ablated_rate = boundary_rotation_rate(2.0, 0.0, 3.0, 5.0)

    checks = {
        "static_frank_bilby_reconstructs_target": residual < 1.0e-13,
        "nonzero_misorientation_requires_boundary_content": np.linalg.norm(coefficients) > 0.0,
        "zero_misorientation_has_zero_read_shockley_energy": zero_energy == 0.0,
        "lagb_energy_is_positive_and_bounded": 0.0 < energy <= 0.65,
        "shear_coupling_generates_rotation": coupled_rate > 0.0,
        "shear_coupling_ablation_is_exact": ablated_rate == 0.0,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    assert all(checks.values()), checks
    return {
        "schema": "gate_D_static_fixture_v1",
        "fixture_passed": True,
        "scientific_gate_passed": False,
        "reason": "dynamic disconnection benchmark, boundary mobility, and reaction law are absent",
        "checks": checks,
        "misorientation_rad": theta,
        "frank_bilby_target": target.tolist(),
        "boundary_coefficients": coefficients.tolist(),
        "normalized_residual": residual,
    }


if __name__ == "__main__":
    print(json.dumps(run_fixture(), indent=2, sort_keys=True))
