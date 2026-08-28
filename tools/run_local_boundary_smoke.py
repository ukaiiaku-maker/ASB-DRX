from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from asb_drx.boundary_campaign import BoundarySpatialCase
from asb_drx.fixtures import SingleGliderDDDParameterization
from asb_drx.local_mechanism import (
    classify_local_mechanism_trace,
    matched_local_isothermal_trace,
    run_local_mechanism_trace,
)
from asb_drx.localization import LocalizationCriteria, localization_history
from asb_drx.mechanism_ladder import MechanismCase
from asb_drx.spatial_coupled import SpatialMechanismControls


def run_grid(
    points: int,
    case: BoundarySpatialCase,
    fixture: SingleGliderDDDParameterization,
    count: int,
) -> dict:
    initial, metadata = case.build_local_state(points, fixture)
    mechanism = MechanismCase(
        "local_antiplane_boundary_smoke",
        case.shear_rate_s_inv,
        SpatialMechanismControls(True, True),
    )
    proposed_dt_s = 2.0e-8
    trace = run_local_mechanism_trace(
        initial, mechanism, metadata["dx_m"], proposed_dt_s, count,
        fixture.law(), fixture.spatial_parameters(),
    )
    control = matched_local_isothermal_trace(
        initial, mechanism, metadata["dx_m"], proposed_dt_s, count,
        fixture.law(), fixture.spatial_parameters(),
    )
    criteria = LocalizationCriteria(0.4, 20.0, 0.1, 3.0, 3, 0.05)
    decision = classify_local_mechanism_trace(
        trace, control, metadata["dx_m"], metadata["interface_width_m"], criteria
    )
    history = localization_history(
        trace.plastic_rate_s_inv, trace.temperature_K, control.temperature_K,
        trace.mean_stress_Pa, metadata["dx_m"],
    )
    final = trace.states[-1]
    final_step = trace.steps[-1]
    return {
        "points": points,
        "metadata": metadata,
        "steps": count,
        "proposed_dt_s": proposed_dt_s,
        "accepted_duration_s": final.time_s,
        "applied_shear": final.applied_shear,
        "final_mean_stress_Pa": final_step.equilibrium.mean_stress_Pa,
        "final_local_stress_min_Pa": float(np.min(final_step.equilibrium.stress_x_Pa)),
        "final_local_stress_max_Pa": float(np.max(final_step.equilibrium.stress_x_Pa)),
        "final_local_stress_std_Pa": float(np.std(final_step.equilibrium.stress_x_Pa)),
        "maximum_temperature_K": float(np.max(trace.temperature_K)),
        "maximum_matched_temperature_excess_K": max(item.temperature_excess_K for item in history),
        "minimum_active_fraction": min(item.active_fraction for item in history),
        "maximum_softening_fraction": max(item.softening_fraction for item in history),
        "phase_change_l2": float(np.linalg.norm(final.eta_fields - initial.eta_fields)),
        "localized": decision.localized,
        "failed_criteria": list(decision.failed_criteria),
        "maximum_equilibrium_residual_Pa_m_inv": max(
            item.equilibrium.equilibrium_residual_Pa_m_inv for item in trace.steps
        ),
        "maximum_absolute_global_closure_error_J_m3": max(
            abs(item.ledger.global_closure_error_J_m3) for item in trace.steps
        ),
        "maximum_absolute_thermal_closure_error_J_m3": max(
            abs(item.ledger.thermal_closure_error_J_m3) for item in trace.steps
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("output/local_boundary_smoke.json"))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    fixture = SingleGliderDDDParameterization()
    case = BoundarySpatialCase(950.0, 45000.0, 1.0)
    grids = [run_grid(points, case, fixture, args.steps) for points in (16, 32)]
    coarse, fine = grids
    changes = {
        "mean_stress": abs(fine["final_mean_stress_Pa"] - coarse["final_mean_stress_Pa"])
        / max(abs(fine["final_mean_stress_Pa"]), 1.0),
        "maximum_temperature": abs(fine["maximum_temperature_K"] - coarse["maximum_temperature_K"])
        / fine["maximum_temperature_K"],
        "local_stress_std": abs(fine["final_local_stress_std_Pa"] - coarse["final_local_stress_std_Pa"])
        / max(abs(fine["final_local_stress_std_Pa"]), 1.0),
    }
    report = {
        "schema": "asb-drx-local-antiplane-boundary-smoke/v1",
        "scientific_disposition": (
            "locally equilibrated integration/refinement smoke at an analytically extrapolated "
            "generic boundary point; not DDD validation, material calibration, ASB, or DRX evidence"
        ),
        "condition": case.__dict__,
        "DDD_rate_validity_note": "45000 s^-1 is outside the source DDD campaign rate 4.5 s^-1",
        "grids": grids,
        "final_refinement_relative_change": changes,
        "provisional_refinement_target": 0.05,
        "refinement_pass": all(value < 0.05 for value in changes.values()),
    }
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
