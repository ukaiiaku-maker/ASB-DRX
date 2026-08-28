from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from asb_drx.analytical import ExpFloorLaw
from asb_drx.coupled import CoupledParameters, CoupledState, advance_coupled
from asb_drx.multi_order import BinaryCircularLimit, binary_boundary_energy_J_m2, diffuse_binary_circle


EV_J = 1.602176634e-19


def main() -> None:
    law = ExpFloorLaw(
        1.5 * EV_J, 1.2e9, 1000.0, 0.2, 2.0, 2.5, 1.0e12, 4.0, 2.5e-10
    )
    parameters = CoupledParameters(
        8.0e10, 3.5e6, 5.0e-9, 1.0e14, 2.0e6, 1.0e-6, 5.0e-7
    )
    density = np.array((5.0e13, 1.0e13))
    phase_parameters = parameters.phase_parameters(density)
    boundary_energy = binary_boundary_energy_J_m2(phase_parameters.phase_parameters)
    limit = BinaryCircularLimit(
        boundary_energy, phase_parameters.driving_energy_J_m3(0, 1), 1.0
    )
    interface_length_m = 2.0 * math.sqrt(
        parameters.gradient_coefficient_J_m / parameters.pair_penalty_J_m3
    )
    dx_m = 1.6e-5 / 64
    fields = diffuse_binary_circle(
        64, dx_m, 1.35 * limit.critical_radius_m, interface_length_m
    )
    initial = CoupledState(
        1.0e8, 0.0, np.zeros(2), density, fields, 1000.0
    )
    final, ledgers = advance_coupled(
        initial, 10.0, dx_m, 1.0e-5, 100, law, parameters
    )
    report = {
        "schema": "asb-drx-coupled-verification/v1",
        "scientific_disposition": (
            "generic binary aggregate/global-ledger fixture; not spatial mechanics, material, DRX, or ASB validation"
        ),
        "steps": 100,
        "initial_stress_Pa": initial.stress_Pa,
        "final_stress_Pa": final.stress_Pa,
        "initial_temperature_K": initial.temperature_K,
        "final_temperature_K": final.temperature_K,
        "initial_forest_density_m2": initial.forest_density_m2.tolist(),
        "final_forest_density_m2": final.forest_density_m2.tolist(),
        "cumulative_external_work_J_m3": sum(
            item.external_work_J_m3 for item in ledgers
        ),
        "cumulative_elastic_change_J_m3": sum(
            item.elastic_energy_change_J_m3 for item in ledgers
        ),
        "cumulative_stored_change_J_m3": sum(
            item.stored_energy_change_J_m3 for item in ledgers
        ),
        "cumulative_interface_order_change_J_m3": sum(
            item.interface_order_energy_change_J_m3 for item in ledgers
        ),
        "cumulative_mechanical_heat_J_m3": sum(
            item.mechanical_heat_J_m3 for item in ledgers
        ),
        "cumulative_phase_heat_J_m3": sum(item.phase_heat_J_m3 for item in ledgers),
        "cumulative_global_closure_error_J_m3": sum(
            item.global_closure_error_J_m3 for item in ledgers
        ),
        "cumulative_thermal_closure_error_J_m3": sum(
            item.thermal_closure_error_J_m3 for item in ledgers
        ),
    }
    output = Path("output/coupled_verification.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
