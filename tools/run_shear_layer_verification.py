from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from asb_drx.analytical import ExpFloorLaw
from asb_drx.shear_layer import (
    ShearLayerParameters,
    ShearLayerState,
    advance_shear_layer,
)


EV_J = 1.602176634e-19


def main() -> None:
    law = ExpFloorLaw(
        1.5 * EV_J, 1.2e9, 1000.0, 0.2, 2.0, 2.5, 1.0e12, 4.0, 2.5e-10
    )
    parameters = ShearLayerParameters(8.0e10, 3.5e6, 25.0, 5.0e-9, 1.0e14)
    points = 32
    coordinate = np.linspace(0.0, 2.0 * np.pi, points, endpoint=False)
    temperature = 1000.0 + 0.25 * np.sin(coordinate)
    initial = ShearLayerState(
        1.0e8, 0.0, np.zeros(points), temperature, np.full(points, 1.0e14)
    )
    final, ledgers = advance_shear_layer(
        initial, 10.0, 2.0e-6, 1.0e-5, 100, law, parameters
    )
    external = sum(item.external_work_J_m3 for item in ledgers)
    report = {
        "schema": "asb-drx-shear-layer-verification/v1",
        "scientific_disposition": "generic common-stress spatial mechanism fixture; not ASB or material validation",
        "points": points,
        "dx_m": 2.0e-6,
        "steps": 100,
        "initial_temperature_mean_K": float(np.mean(initial.temperature_K)),
        "final_temperature_mean_K": float(np.mean(final.temperature_K)),
        "initial_temperature_std_K": float(np.std(initial.temperature_K)),
        "final_temperature_std_K": float(np.std(final.temperature_K)),
        "final_stress_Pa": final.stress_Pa,
        "cumulative_external_work_J_m3": external,
        "cumulative_mechanical_closure_error_J_m3": sum(item.mechanical_closure_error_J_m3 for item in ledgers),
        "cumulative_thermal_closure_error_J_m3": sum(item.thermal_closure_error_J_m3 for item in ledgers),
    }
    output = Path("output/shear_layer_verification.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
