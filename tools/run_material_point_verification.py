from __future__ import annotations

import json
from pathlib import Path

from asb_drx.analytical import ExpFloorLaw
from asb_drx.material_point import (
    MaterialPointParameters,
    MaterialPointState,
    advance_material_point,
)
from asb_drx.thermodynamics import DislocationReservoirs


EV_J = 1.602176634e-19


def main() -> None:
    law = ExpFloorLaw(
        1.5 * EV_J, 1.2e9, 1000.0, 0.2, 2.0, 2.5, 1.0e12, 4.0, 2.5e-10
    )
    parameters = MaterialPointParameters(8.0e10, 3.5e6, 5.0e-9, 1.0e14)
    initial = MaterialPointState(
        1.0e8,
        0.0,
        0.0,
        1000.0,
        DislocationReservoirs(1.0e13, 1.0e14, 0.0, 0.0),
    )
    final, ledgers = advance_material_point(initial, 10.0, 1.0e-5, 100, law, parameters)
    external = sum(item.external_work_J_m3 for item in ledgers)
    closure = sum(item.closure_error_J_m3 for item in ledgers)
    report = {
        "schema": "asb-drx-material-point-verification/v1",
        "scientific_disposition": "generic finite-loading/work-ledger fixture; not material calibration",
        "steps": 100,
        "initial_temperature_K": initial.temperature_K,
        "final_temperature_K": final.temperature_K,
        "initial_stress_Pa": initial.stress_Pa,
        "final_stress_Pa": final.stress_Pa,
        "initial_forest_density_m2": initial.reservoirs.forest_m2,
        "final_forest_density_m2": final.reservoirs.forest_m2,
        "cumulative_external_work_J_m3": external,
        "cumulative_elastic_change_J_m3": sum(item.elastic_energy_change_J_m3 for item in ledgers),
        "cumulative_stored_dislocation_J_m3": sum(item.stored_dislocation_J_m3 for item in ledgers),
        "cumulative_heat_J_m3": sum(item.heat_J_m3 for item in ledgers),
        "cumulative_closure_error_J_m3": closure,
        "relative_closure_error": closure / abs(external),
    }
    output = Path("output/material_point_verification.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
