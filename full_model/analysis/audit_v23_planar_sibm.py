#!/usr/bin/env python3
"""V23 frozen planar SIBM and actual cap-direction functional audit."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from full_model.production.sibm_geometry import pinned_cap_energy
from full_model.production.stored_energy_coupling import (
    common_variational_stored_energy, frozen_planar_sibm_response,
)


def main():
    mobility = 2e-12
    cases = {
        "equal": frozen_planar_sibm_response(4e7, 4e7, mobility),
        "favorable": frozen_planar_sibm_response(8e7, 2e7, mobility),
        "reverse": frozen_planar_sibm_response(2e7, 8e7, mobility),
        "mobility_off": frozen_planar_sibm_response(8e7, 2e7, 0.0),
    }
    eta = np.array([[.5, .5]])
    energy = np.array([[8e7, 2e7]])
    _, derivative = common_variational_stored_energy(eta, energy)
    direction = np.array([[-1.0, 1.0]])
    analytical = float(np.sum(derivative*direction))
    epsilon = 1e-7
    fd = float((common_variational_stored_energy(
        eta+epsilon*direction, energy)[0][0]
        -common_variational_stored_energy(
            eta-epsilon*direction, energy)[0][0])/(2*epsilon))
    amplitude = 6e-7; half_chord = 1.5e-6; gamma = .5
    zero = pinned_cap_energy(
        amplitude, half_chord, boundary_energy_J_m2=gamma,
        stored_pressure_Pa=0.0)
    critical = gamma*zero["d_arc_length_da"]/zero["d_swept_area_da_m"]
    caps = []
    for factor in (.8, 1.2):
        kwargs = dict(boundary_energy_J_m2=gamma,
                      stored_pressure_Pa=factor*critical,
                      represented_thickness_m=1e-6)
        cap = pinned_cap_energy(amplitude, half_chord, **kwargs)
        h = 1e-10
        finite = (pinned_cap_energy(amplitude+h, half_chord, **kwargs)["total_energy_J"]
                  -pinned_cap_energy(amplitude-h, half_chord, **kwargs)["total_energy_J"])/(2*h)
        caps.append({
            "pressure_over_cap_critical": factor,
            "pressure_Pa": factor*critical,
            "analytical_dE_da_J_m": cap["d_total_energy_da_J_m"],
            "finite_difference_dE_da_J_m": finite,
            "relative_error": abs(cap["d_total_energy_da_J_m"]-finite)
                              /max(abs(cap["d_total_energy_da_J_m"]), 1e-300),
        })
    passed = (
        cases["equal"]["normal_velocity_m_s"] == 0
        and cases["favorable"]["normal_velocity_m_s"] > 0
        and cases["reverse"]["normal_velocity_m_s"] < 0
        and cases["mobility_off"]["normal_velocity_m_s"] == 0
        and analytical < 0 and abs(analytical-fd)/abs(analytical) < 2e-9
        and caps[0]["analytical_dE_da_J_m"] > 0
        and caps[1]["analytical_dE_da_J_m"] < 0
        and max(x["relative_error"] for x in caps) < 2e-7)
    result = {
        "schema": "v23-planar-sibm-functional-audit-1",
        "external_pressure_Pa": 0.0,
        "planar_cases": cases,
        "common_functional_direction": {
            "analytical_J_m3": analytical,
            "finite_difference_J_m3": fd,
            "relative_error": abs(analytical-fd)/abs(analytical),
        },
        "cap_amplitude_m": amplitude,
        "cap_half_chord_m": half_chord,
        "cap_critical_pressure_Pa": critical,
        "cap_directional_derivatives": caps,
        "fixture_passed": bool(passed),
        "scientific_gate_passed": False,
        "classification": ("PLANAR_SIBM_FUNCTIONAL_PASSED_SEQUENTIAL_HANDOFF_PENDING"
                           if passed else "PLANAR_SIBM_FUNCTIONAL_FAILED"),
    }
    output = ROOT/"full_model"/"verification"/"v23_planar_sibm_audit.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
