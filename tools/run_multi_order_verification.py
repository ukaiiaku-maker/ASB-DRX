from __future__ import annotations

import json
import math
from pathlib import Path

from asb_drx.multi_order import (
    BinaryCircularLimit,
    MultiOrderParameters,
    MultiOrderState,
    advance_multi_order,
    binary_boundary_energy_J_m2,
    diffuse_binary_circle,
    equivalent_child_radius_m,
    multi_order_free_energy_J_m,
)


def main() -> None:
    parameters = MultiOrderParameters(2.0e6, 1.0e-6, 5.0e-7, (0.0, -2.0e5))
    boundary_energy = binary_boundary_energy_J_m2(parameters)
    limit = BinaryCircularLimit(boundary_energy, 2.0e5, 1.0)
    interface_length_m = 2.0 * math.sqrt(
        parameters.gradient_coefficient_J_m / parameters.pair_penalty_J_m3
    )
    dx_m = 1.6e-5 / 128
    cases = {}
    for name, ratio in (("subcritical", 0.72), ("supercritical", 1.35)):
        fields = diffuse_binary_circle(
            128, dx_m, ratio * limit.critical_radius_m, interface_length_m
        )
        initial_radius = equivalent_child_radius_m(fields, dx_m)
        initial_energy = multi_order_free_energy_J_m(fields, dx_m, parameters)
        final = advance_multi_order(
            MultiOrderState(fields, 0.0, 0), dx_m, 1.0e-4, 200, parameters
        )
        cases[name] = {
            "initial_radius_m": initial_radius,
            "final_radius_m": equivalent_child_radius_m(final.eta_fields, dx_m),
            "initial_energy_J_m": initial_energy,
            "final_energy_J_m": multi_order_free_energy_J_m(
                final.eta_fields, dx_m, parameters
            ),
            "accepted_steps": final.accepted_steps,
            "final_simplex_max_error": float(
                abs(final.eta_fields.sum(axis=0) - 1.0).max()
            ),
        }
    report = {
        "schema": "asb-drx-multi-order-verification/v1",
        "scientific_disposition": (
            "generic constrained phase-field fixture; not nucleation, material, DRX, or ASB validation"
        ),
        "boundary_energy_J_m2": boundary_energy,
        "critical_radius_m": limit.critical_radius_m,
        "interface_length_m": interface_length_m,
        "cases": cases,
    }
    output = Path("output/multi_order_verification.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
