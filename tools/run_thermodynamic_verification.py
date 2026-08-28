from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from asb_drx.thermodynamics import (
    CircularNucleusLimit,
    GrainEnergyParameters,
    PhaseFieldState2D,
    advance_phase_field_2d,
    diffuse_circle_2d,
    energy_checked_allen_cahn_step,
    equivalent_support_radius_m,
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

    diffuse_gamma = (
        2.0 * parameters.gradient_coefficient_J_m * parameters.well_height_J_m3
    ) ** 0.5 / 6.0
    diffuse_limit = CircularNucleusLimit(diffuse_gamma, parameters.bulk_driving_J_m3, 1.0)
    interface_length_m = (
        parameters.gradient_coefficient_J_m / parameters.well_height_J_m3
    ) ** 0.5

    def radius_change(grid_points: int, ratio: float, dt_s: float, steps: int) -> float:
        dx = 1.6e-5 / grid_points
        field = diffuse_circle_2d(
            grid_points,
            dx,
            ratio * diffuse_limit.critical_radius_m,
            interface_length_m,
        )
        initial_radius = equivalent_support_radius_m(field, dx)
        evolved = advance_phase_field_2d(
            PhaseFieldState2D(field, 0.0, 0), dx, dt_s, steps, parameters
        )
        return equivalent_support_radius_m(evolved.eta, dx) - initial_radius

    grid_changes = [radius_change(points, 1.35, 1.0e-4, 100) for points in (64, 96, 128)]
    time_changes = [
        radius_change(128, 1.35, dt_s, steps)
        for dt_s, steps in ((2.0e-4, 50), (1.0e-4, 100), (5.0e-5, 200))
    ]
    report["diffuse_2d_nucleus"] = {
        "derived_gb_energy_J_m2": diffuse_gamma,
        "derived_critical_radius_m": diffuse_limit.critical_radius_m,
        "subcritical_radius_change_m": radius_change(128, 0.72, 1.0e-4, 200),
        "supercritical_radius_change_m": radius_change(128, 1.35, 1.0e-4, 200),
        "grid_points": [64, 96, 128],
        "grid_radius_changes_m": grid_changes,
        "final_grid_relative_change": abs(grid_changes[-1] - grid_changes[-2]) / abs(grid_changes[-1]),
        "timesteps_s": [2.0e-4, 1.0e-4, 5.0e-5],
        "time_radius_changes_m": time_changes,
        "final_timestep_relative_change": abs(time_changes[-1] - time_changes[-2]) / abs(time_changes[-1]),
    }
    output = Path("output/thermodynamic_verification.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
