#!/usr/bin/env python3
"""Map the normalized local Hessian of the v18 density/wall functional."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from full_model.production.dislocation_free_energy import (
    DislocationFreeEnergyParameters, free_energy_J_m3,
)
from full_model.production.wall_ordering_energy import (
    WallOrderingParameters, local_wall_energy_J_m3,
)


def numerical_hessian(function, point, step=2.0e-4):
    point = np.asarray(point, dtype=float)
    n = point.size
    result = np.empty((n, n))
    f0 = function(point)
    for i in range(n):
        ei = np.zeros(n); ei[i] = step
        result[i, i] = (function(point+ei)-2.0*f0+function(point-ei))/step**2
        for j in range(i):
            ej = np.zeros(n); ej[j] = step
            value = (function(point+ei+ej)-function(point+ei-ej)
                     -function(point-ei+ej)+function(point-ei-ej))/(4.0*step**2)
            result[i, j] = result[j, i] = value
    return result


def run():
    rho_scale = 5.0e14
    line = 0.5*45.0e9*(2.48e-10)**2
    storage = DislocationFreeEnergyParameters(
        line, 0.06*line, 1.0e14, 1.0, rho_scale, 1.3, 0.3,
        low_density_strength=0.25,
    )
    wall = WallOrderingParameters(
        ordering_amplitude_J_m3=1.2*line*rho_scale*0.3,
        density_scale_m2=rho_scale,
        center_ratio=1.3,
        width_ratio=0.3,
        order_barrier_J_m3=0.08*line*rho_scale,
        wall_partition_coefficient_J_m=0.25*line,
        target_wall_density_m2=0.8*rho_scale,
    )

    def energy(x):
        # x=(rho_m,rho_f,rho_w)/rho_scale, q.  The inherited scalar ordering
        # is omitted from storage and reintroduced exactly once through q.
        densities = x[:3]*rho_scale
        total = float(np.sum(densities))
        return float(free_energy_J_m3(total, storage)
                     +local_wall_energy_J_m3(total, densities[2], x[3], wall))

    states = []
    for total_ratio in (0.35, 0.7, 1.0, 1.3, 1.6, 2.0):
        for q in (0.05, 0.35, 0.65, 0.95):
            point = np.array([0.25*total_ratio, 0.45*total_ratio,
                              0.30*total_ratio, q])
            hessian = numerical_hessian(energy, point)
            eigenvalues = np.linalg.eigvalsh(hessian)
            states.append({
                "total_density_ratio": total_ratio,
                "wall_order": q,
                "eigenvalues_J_m3_per_normalized_state2": eigenvalues.tolist(),
                "minimum_eigenvalue_J_m3_per_normalized_state2": float(eigenvalues[0]),
                "negative_eigenvalue_count": int(np.sum(eigenvalues < -1.0)),
            })
    return {
        "schema": "asb-drx/v18-total-hessian/v1",
        "variables": ["rho_mobile/rho_scale", "rho_forest/rho_scale",
                      "rho_wall/rho_scale", "q_wall"],
        "rho_scale_m2": rho_scale,
        "log_coefficient_J_m": storage.log_coefficient_J_m,
        "log_curvature_sign": "strictly_positive_for_positive_density",
        "scalar_ordering_count": 0,
        "explicit_q_weighted_ordering_count": 1,
        "states": states,
        "interpretation": {
            "convex_storage": "positive rho-log-rho curvature and wall-reservoir partition penalty",
            "nonconvex_organization": "q-weighted inherited ordering dip plus bounded q double-well",
            "low_density_regularization": "positive rho-log-rho curvature and optional exponential soft penalty",
            "wall_state_stabilization": "q-weighted ordering dip and wall-reservoir target",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"output": str(args.output), "states": len(result["states"]),
                      "minimum_eigenvalue": min(x["minimum_eigenvalue_J_m3_per_normalized_state2"] for x in result["states"])}, indent=2))


if __name__ == "__main__":
    main()
