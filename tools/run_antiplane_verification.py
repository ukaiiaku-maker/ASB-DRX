from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from asb_drx.antiplane import midpoint_work_ledger_J_m3, solve_periodic_antiplane


def main() -> None:
    points = 64
    domain_m = 1.6e-5
    dx_m = domain_m / points
    modulus = 8.0e10
    coordinate = np.linspace(0.0, 2.0 * np.pi, points, endpoint=False)
    y_band = 0.01 * np.sin(coordinate)[:, None] * np.ones((1, points))
    old = solve_periodic_antiplane(0.02, 0.25 * y_band, modulus, dx_m)
    new = solve_periodic_antiplane(0.021, 0.50 * y_band, modulus, dx_m)
    ledger = midpoint_work_ledger_J_m3(old, new, 0.001, 0.25 * y_band)
    report = {
        "schema": "asb-drx-periodic-antiplane-equilibrium/v1",
        "scientific_disposition": "isolated local-stress-redistribution gate; not yet constitutively coupled or ASB evidence",
        "grid_points": points,
        "domain_m": domain_m,
        "dx_m": dx_m,
        "shear_modulus_Pa": modulus,
        "old": {"mean_stress_Pa": old.mean_stress_Pa, "elastic_energy_J_m3": old.elastic_energy_J_m3, "equilibrium_residual_Pa_m_inv": old.equilibrium_residual_Pa_m_inv},
        "new": {"mean_stress_Pa": new.mean_stress_Pa, "elastic_energy_J_m3": new.elastic_energy_J_m3, "equilibrium_residual_Pa_m_inv": new.equilibrium_residual_Pa_m_inv, "stress_x_min_Pa": float(np.min(new.stress_x_Pa)), "stress_x_max_Pa": float(np.max(new.stress_x_Pa))},
        "midpoint_ledger_J_m3": {"external": ledger[0], "plastic": ledger[1], "elastic_change": ledger[2], "closure": ledger[3]},
    }
    output = Path("output/antiplane_verification.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
