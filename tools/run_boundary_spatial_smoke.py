from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from asb_drx.boundary_campaign import BoundarySpatialCase
from asb_drx.fixtures import SingleGliderDDDParameterization
from asb_drx.localization import LocalizationCriteria, localization_history
from asb_drx.mechanism_ladder import (
    MechanismCase,
    classify_mechanism_trace,
    matched_isothermal_case,
    run_mechanism_trace,
)
from asb_drx.spatial_coupled import SpatialMechanismControls


def run_grid(points: int, case: BoundarySpatialCase, fixture: SingleGliderDDDParameterization) -> dict:
    initial, metadata = case.build_state(points, fixture)
    thermal_case = MechanismCase(
        "coupled_boundary_smoke",
        case.shear_rate_s_inv,
        SpatialMechanismControls(True, True),
    )
    control_case = matched_isothermal_case(thermal_case)
    proposed_dt_s = 2.0e-8
    steps = 100
    thermal = run_mechanism_trace(
        initial, thermal_case, metadata["dx_m"], proposed_dt_s, steps,
        fixture.law(), fixture.spatial_parameters(),
    )
    control = run_mechanism_trace(
        initial, control_case, metadata["dx_m"], proposed_dt_s, steps,
        fixture.law(), fixture.spatial_parameters(),
    )
    criteria = LocalizationCriteria(0.4, 20.0, 0.1, 3.0, 3, 0.05)
    decision = classify_mechanism_trace(
        thermal, control, metadata["dx_m"], metadata["interface_width_m"], criteria
    )
    history = localization_history(
        thermal.plastic_rate_s_inv, thermal.temperature_K,
        control.temperature_K, thermal.stress_Pa, metadata["dx_m"],
    )
    final = thermal.states[-1]
    return {
        "points": points,
        "metadata": metadata,
        "steps": steps,
        "proposed_dt_s": proposed_dt_s,
        "accepted_duration_s": final.time_s,
        "applied_shear": final.applied_shear,
        "final_stress_Pa": final.stress_Pa,
        "maximum_temperature_K": float(np.max(thermal.temperature_K)),
        "maximum_matched_temperature_excess_K": max(item.temperature_excess_K for item in history),
        "minimum_active_fraction": min(item.active_fraction for item in history),
        "maximum_softening_fraction": max(item.softening_fraction for item in history),
        "phase_change_l2": float(np.linalg.norm(final.eta_fields - initial.eta_fields)),
        "localized": decision.localized,
        "failed_criteria": list(decision.failed_criteria),
        "maximum_absolute_global_closure_error_J_m3": max(abs(item.global_closure_error_J_m3) for item in thermal.ledgers),
        "maximum_absolute_thermal_closure_error_J_m3": max(abs(item.thermal_closure_error_J_m3) for item in thermal.ledgers),
    }


def main() -> None:
    fixture = SingleGliderDDDParameterization()
    case = BoundarySpatialCase(950.0, 45000.0, 1.0)
    grids = [run_grid(points, case, fixture) for points in (16, 32)]
    coarse, fine = grids
    stress_change = abs(fine["final_stress_Pa"] - coarse["final_stress_Pa"]) / max(abs(fine["final_stress_Pa"]), 1.0)
    temperature_change = abs(fine["maximum_temperature_K"] - coarse["maximum_temperature_K"]) / fine["maximum_temperature_K"]
    report = {
        "schema": "asb-drx-boundary-spatial-smoke/v1",
        "scientific_disposition": (
            "single-job numerical-coupling smoke at an analytically extrapolated generic boundary point; "
            "not DDD validation, material calibration, ASB, DRX, or production-boundary evidence"
        ),
        "condition": case.__dict__,
        "DDD_rate_validity_note": "45000 s^-1 is outside the source DDD campaign rate 4.5 s^-1",
        "grids": grids,
        "final_refinement_relative_change": {
            "stress": stress_change,
            "maximum_temperature": temperature_change,
        },
        "provisional_refinement_target": 0.05,
        "refinement_pass": stress_change < 0.05 and temperature_change < 0.05,
    }
    output = Path("output/boundary_spatial_smoke.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
