#!/usr/bin/env python3
"""Emit the deterministic V23 density-map audit."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from full_model.production.density_state_map import (
    DensityInventory, derived_density_fields, line_energy_density_J_m3,
    taylor_resistance_Pa,
)
from full_model.production.tensorial_nye import (
    bcc_four_family_systems, make_junction_topology,
)


def main():
    systems = bcc_four_family_systems()
    topologies = (make_junction_topology(
        systems, 0, 1, 1, -1, line_tension_J_m=0.0),)
    shape = (4, 4, 4)
    inventory = DensityInventory(
        *(np.full(shape, value) for value in
          (3e13, 2e13, 4e13, 3e13, 2e13, 1e13, 5e12, 4e12)),
        np.full((4, 4, 1), 5e12))
    fields = derived_density_fields(inventory, topologies)
    line_energy = line_energy_density_J_m3(inventory, topologies, 1.5e-9)
    tau = taylor_resistance_Pa(
        inventory, topologies, 116.5e9, systems[0].burgers_m,
        alpha=.3, wall_weight=2.0)
    expected = sum(4 * x for x in
                   (3e13, 2e13, 4e13, 3e13, 2e13, 1e13, 5e12, 4e12))
    expected += 5e12 * topologies[0].product_line_multiplicity
    closure = float(np.max(np.abs(fields["rho_total_m2"] - expected)))
    result = {
        "schema": "v23-density-map-audit-1",
        "units": {
            "all_line_populations": "m^-2",
            "line_energy": "J m^-1",
            "energy_density": "J m^-3",
            "taylor_resistance": "Pa",
            "area_integral": "line length per out-of-plane thickness"
        },
        "thickness_factor": 1.0,
        "silent_1e3_conversion_present": False,
        "junction_product_line_multiplicity": float(
            topologies[0].product_line_multiplicity),
        "maximum_total_density_reconstruction_residual_m2": closure,
        "mean_total_density_m2": float(np.mean(fields["rho_total_m2"])),
        "mean_line_energy_density_J_m3": float(np.mean(line_energy)),
        "taylor_resistance_range_Pa": [float(np.min(tau)), float(np.max(tau))],
        "fixture_passed": bool(closure == 0 and np.min(tau) > 1e7
                               and np.max(tau) < 2e9),
        "scientific_gate_passed": False,
        "classification": "DENSITY_MAP_QUALIFIED_FB_WALL_CLOSURE_PENDING"
    }
    source = Path("full_model/production/density_state_map.py")
    result["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    output = Path("full_model/verification/v23_density_map_audit.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
