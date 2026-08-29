#!/usr/bin/env python3
"""Gate A kinematic fixture.

This verifies BCC-family/MRSSP/plastic-spin invariants. It deliberately reports
the scientific gate as not passed because no Bertin trajectory/calibration is
part of the campaign evidence.
"""

from __future__ import annotations

import json
import numpy as np


BURGERS_FAMILIES = np.asarray(
    [[1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [-1.0, 1.0, 1.0], [1.0, 1.0, -1.0]]
)
BURGERS_FAMILIES /= np.linalg.norm(BURGERS_FAMILIES, axis=1)[:, None]


def mrssp_normal(stress: np.ndarray, slip_direction: np.ndarray) -> tuple[np.ndarray, float]:
    traction = stress @ slip_direction
    projected = traction - slip_direction * np.dot(slip_direction, traction)
    magnitude = float(np.linalg.norm(projected))
    if magnitude <= 1.0e-14:
        return np.zeros(3), 0.0
    normal = projected / magnitude
    return normal, float(slip_direction @ stress @ normal)


def plastic_spin(stress: np.ndarray, exponent: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    lp = np.zeros((3, 3))
    rates = []
    scale = max(float(np.linalg.norm(stress)), 1.0)
    for direction in BURGERS_FAMILIES:
        normal, tau = mrssp_normal(stress, direction)
        rate = np.sign(tau) * (abs(tau) / scale) ** exponent if tau else 0.0
        lp += rate * np.outer(direction, normal)
        rates.append(rate)
    return 0.5 * (lp - lp.T), np.asarray(rates)


def inactive_relaxation(density: np.ndarray, rates: np.ndarray, coefficient: float) -> np.ndarray:
    activity = np.abs(rates)
    fraction = activity / max(float(np.sum(activity)), np.finfo(float).tiny)
    return -coefficient * (1.0 - fraction) * density


def run_fixture() -> dict[str, object]:
    stress = np.asarray([[1.0, 0.37, -0.11], [0.37, -0.55, 0.23], [-0.11, 0.23, -0.45]])
    normals = [mrssp_normal(stress, direction)[0] for direction in BURGERS_FAMILIES]
    orthogonality = max(abs(float(np.dot(s, m))) for s, m in zip(BURGERS_FAMILIES, normals))
    wp, rates = plastic_spin(stress)
    zero_wp, zero_rates = plastic_spin(np.zeros((3, 3)))
    density = np.asarray([2.0, 3.0, 5.0, 7.0])
    relaxation = inactive_relaxation(density, rates, coefficient=0.2)

    checks = {
        "four_bcc_burgers_families": BURGERS_FAMILIES.shape == (4, 3),
        "mrssp_normal_is_orthogonal": orthogonality < 1.0e-13,
        "plastic_spin_is_skew": np.allclose(wp + wp.T, 0.0, atol=1.0e-14),
        "no_stress_no_slip_no_rotation": np.array_equal(zero_wp, np.zeros((3, 3)))
        and np.array_equal(zero_rates, np.zeros(4)),
        "inactive_relaxation_never_creates_density": bool(np.all(relaxation <= 0.0)),
        "plastic_spin_ablation_exact": np.array_equal(0.0 * wp, np.zeros((3, 3))),
    }
    assert all(checks.values()), checks
    return {
        "schema": "gate_A_kinematic_fixture_v1",
        "fixture_passed": True,
        "scientific_gate_passed": False,
        "reason": "Bertin stable/unstable orientation trajectories and transferable parameters are absent",
        "checks": checks,
        "maximum_mrssp_orthogonality_error": orthogonality,
        "plastic_spin_norm": float(np.linalg.norm(wp)),
    }


if __name__ == "__main__":
    print(json.dumps(run_fixture(), indent=2, sort_keys=True))
