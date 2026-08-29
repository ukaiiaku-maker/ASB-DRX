#!/usr/bin/env python3
"""Gate C mechanism-separation and conservative-transfer fixture."""

from __future__ import annotations

import json
import math
import numpy as np


KB = 1.380649e-23
EV = 1.602176634e-19


def arrhenius(prefactor: float, barrier_eV: float, temperature_K: float) -> float:
    return prefactor * math.exp(-barrier_eV * EV / (KB * temperature_K))


def transfer_step(state: np.ndarray, dt: float, capture: float, emission: float) -> np.ndarray:
    mobile, junction, boundary = state
    dm = (-capture * mobile + emission * boundary) * dt
    dj = 0.35 * capture * mobile * dt
    db = 0.65 * capture * mobile * dt - emission * boundary * dt
    return state + np.asarray([dm, dj, db])


def run_fixture() -> dict[str, object]:
    initial = np.asarray([8.0, 1.0, 1.0])
    glide = transfer_step(initial, 1.0e-3, capture=20.0, emission=2.0)
    climb_low = arrhenius(1.0e5, 1.2, 700.0)
    climb_high = arrhenius(1.0e5, 1.2, 1100.0)
    cross_slip = arrhenius(1.0e5, 0.6, 900.0)
    theta_before = 0.17
    theta_after = theta_before  # reservoir recovery has no orientation source

    checks = {
        "conservative_capture_transfer": abs(float(np.sum(glide) - np.sum(initial))) < 1.0e-14,
        "mobile_interior_depletes": glide[0] < initial[0],
        "boundary_content_increases": glide[2] > initial[2],
        "climb_is_temperature_activated": climb_high > climb_low > 0.0,
        "cross_slip_channel_is_separate": cross_slip > 0.0 and not math.isclose(cross_slip, climb_high),
        "recovery_does_not_invent_orientation": theta_after == theta_before,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    assert all(checks.values()), checks
    return {
        "schema": "gate_C_mechanism_fixture_v1",
        "fixture_passed": True,
        "scientific_gate_passed": False,
        "reason": "glide/cross-slip/climb parameters and a spatial polygonization benchmark are absent",
        "checks": checks,
        "initial_reservoirs": initial.tolist(),
        "post_capture_reservoirs": glide.tolist(),
        "climb_rate_ratio_1100K_to_700K": climb_high / climb_low,
    }


if __name__ == "__main__":
    print(json.dumps(run_fixture(), indent=2, sort_keys=True))
