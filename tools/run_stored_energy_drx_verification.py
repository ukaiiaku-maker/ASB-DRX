from __future__ import annotations

import json
import math
from pathlib import Path

from asb_drx.multi_order import (
    BinaryCircularLimit,
    MultiOrderState,
    binary_boundary_energy_J_m2,
    diffuse_binary_circle,
    equivalent_child_radius_m,
)
from asb_drx.stored_energy_drx import (
    StoredEnergyDRXParameters,
    StoredEnergyDRXState,
    advance_stored_energy_drx,
)


def main() -> None:
    parameters = StoredEnergyDRXParameters(
        2.0e6, 1.0e-6, 5.0e-7, 5.0e-9, (5.0e13, 1.0e13), 3.5e6
    )
    boundary_energy = binary_boundary_energy_J_m2(parameters.phase_parameters)
    driving_energy = parameters.driving_energy_J_m3(0, 1)
    limit = BinaryCircularLimit(boundary_energy, driving_energy, 1.0)
    interface_length_m = 2.0 * math.sqrt(
        parameters.gradient_coefficient_J_m / parameters.pair_penalty_J_m3
    )
    dx_m = 1.6e-5 / 128
    fields = diffuse_binary_circle(
        128, dx_m, 1.35 * limit.critical_radius_m, interface_length_m
    )
    initial = StoredEnergyDRXState(MultiOrderState(fields, 0.0, 0), 1000.0)
    initial_radius = equivalent_child_radius_m(fields, dx_m)
    final, ledgers = advance_stored_energy_drx(
        initial, dx_m, 1.0e-4, 200, parameters
    )
    report = {
        "schema": "asb-drx-stored-energy-coupling-verification/v1",
        "scientific_disposition": (
            "generic stored-energy/phase/heat ledger fixture; not material, nucleation, DRX, or ASB validation"
        ),
        "stored_line_energy_J_m": parameters.stored_line_energy_J_m,
        "grain_dislocation_density_m2": parameters.grain_dislocation_density_m2,
        "driving_energy_J_m3": driving_energy,
        "boundary_energy_J_m2": boundary_energy,
        "critical_radius_m": limit.critical_radius_m,
        "initial_radius_m": initial_radius,
        "final_radius_m": equivalent_child_radius_m(final.phase.eta_fields, dx_m),
        "initial_temperature_K": initial.temperature_K,
        "final_temperature_K": final.temperature_K,
        "cumulative_stored_energy_change_J_m": sum(
            item.stored_energy_change_J_m for item in ledgers
        ),
        "cumulative_interfacial_energy_change_J_m": sum(
            item.interfacial_energy_change_J_m for item in ledgers
        ),
        "cumulative_heat_J_m": sum(item.heat_J_m for item in ledgers),
        "cumulative_closure_error_J_m": sum(
            item.closure_error_J_m for item in ledgers
        ),
    }
    output = Path("output/stored_energy_drx_verification.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
