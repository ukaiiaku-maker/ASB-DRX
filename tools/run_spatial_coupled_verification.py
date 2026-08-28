from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np

from asb_drx.analytical import ExpFloorLaw
from asb_drx.multi_order import BinaryCircularLimit, diffuse_binary_circle
from asb_drx.spatial_coupled import SpatialCoupledParameters, SpatialCoupledState, advance_spatial_coupled

EV_J = 1.602176634e-19

def main() -> None:
    law = ExpFloorLaw(1.5 * EV_J, 1.2e9, 1000.0, 0.2, 2.0, 2.5, 1.0e12, 4.0, 2.5e-10)
    p = SpatialCoupledParameters(8.0e10, 3.5e6, 5.0, 5.0e-9, 1.0e14, 2.0e6, 1.0e-6, 5.0e-7)
    points = 32; dx_m = 1.6e-5 / points
    gamma = math.sqrt(p.gradient_coefficient_J_m * p.pair_penalty_J_m3) / 3.0
    limit = BinaryCircularLimit(gamma, p.stored_line_energy_J_m * 4.0e13, 1.0)
    eta = diffuse_binary_circle(points, dx_m, 1.35 * limit.critical_radius_m, 2.0 * math.sqrt(p.gradient_coefficient_J_m / p.pair_penalty_J_m3))
    rho = np.empty_like(eta); rho[0] = 5.0e13; rho[1] = 1.0e13
    coordinate = np.linspace(0.0, 2.0 * math.pi, points, endpoint=False)
    temperature = 1000.0 + 0.25 * np.sin(coordinate)[None, :]
    temperature = np.broadcast_to(temperature, (points, points)).copy()
    initial = SpatialCoupledState(1.0e8, 0.0, np.zeros((points, points)), temperature, rho, eta)
    final, ledgers = advance_spatial_coupled(initial, 10.0, dx_m, 1.0e-5, 100, law, p)
    report = {
        "schema": "asb-drx-spatial-coupled-verification/v1",
        "scientific_disposition": "generic periodic common-stress 2-D fixture; not localization, material, DRX, or ASB validation",
        "points": points, "dx_m": dx_m, "steps": 100,
        "initial_temperature_mean_K": float(np.mean(initial.temperature_K)),
        "final_temperature_mean_K": float(np.mean(final.temperature_K)),
        "initial_temperature_std_K": float(np.std(initial.temperature_K)),
        "final_temperature_std_K": float(np.std(final.temperature_K)),
        "final_stress_Pa": final.stress_Pa,
        "cumulative_external_work_J_m3": sum(x.external_work_J_m3 for x in ledgers),
        "cumulative_global_closure_error_J_m3": sum(x.global_closure_error_J_m3 for x in ledgers),
        "cumulative_thermal_closure_error_J_m3": sum(x.thermal_closure_error_J_m3 for x in ledgers),
    }
    output = Path("output/spatial_coupled_verification.json"); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

if __name__ == "__main__": main()
