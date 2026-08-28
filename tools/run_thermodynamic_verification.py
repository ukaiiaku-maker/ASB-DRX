from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from asb_drx.thermodynamics import (
    CircularNucleusLimit,
    GrainEnergyParameters,
    energy_checked_allen_cahn_step,
    free_energy_1d_J_m2,
)


def main() -> None:
    parameters = GrainEnergyParameters(2.0e6, 2.0e-6, 2.0e5, 5.0e-7)
    coordinate = np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)
    eta = 0.45 + 0.08 * np.sin(coordinate) + 0.03 * np.cos(3.0 * coordinate)
    dx_m = 1.0e-7
    initial = free_energy_1d_J_m2(eta, dx_m, parameters)
    halvings = []
    for _ in range(100):
        step = energy_checked_allen_cahn_step(eta, dx_m, 1.0e-4, parameters)
        eta = step.eta
        halvings.append(step.halvings)
    final = free_energy_1d_J_m2(eta, dx_m, parameters)

    nucleus = CircularNucleusLimit(0.5, 2.0e6, 3.0e-10)
    critical = nucleus.critical_radius_m
    report = {
        "schema": "asb-drx-thermodynamic-verification/v1",
        "scientific_disposition": "dimension/sign/conservation verification fixture; not material calibration",
        "relaxation": {
            "initial_free_energy_J_m2": initial,
            "final_free_energy_J_m2": final,
            "relative_change": (final - initial) / abs(initial),
            "steps": 100,
            "maximum_step_halvings": max(halvings),
        },
        "circular_nucleus": {
            "critical_radius_m": critical,
            "subcritical_radius_rate_m_s": nucleus.radius_rate_m_s(0.8 * critical),
            "critical_radius_rate_m_s": nucleus.radius_rate_m_s(critical),
            "supercritical_radius_rate_m_s": nucleus.radius_rate_m_s(1.2 * critical),
        },
    }
    output = Path("output/thermodynamic_verification.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
