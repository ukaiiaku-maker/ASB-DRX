from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from asb_drx.boundary_campaign import BoundarySpatialCase
from asb_drx.fixtures import SingleGliderDDDParameterization
from asb_drx.local_mechanism import (
    classify_local_mechanism_trace,
    run_matched_local_strain_pair,
)
from asb_drx.localization import LocalizationCriteria, localization_history
from asb_drx.mechanism_ladder import MechanismCase
from asb_drx.spatial_coupled import SpatialMechanismControls


def run_condition(
    temperature_K: float,
    shear_rate_s_inv: float,
    density_ratio: float,
    points: int,
    steps: int,
    target_shear_increment: float,
    fixture: SingleGliderDDDParameterization,
) -> dict:
    case = BoundarySpatialCase(temperature_K, shear_rate_s_inv, density_ratio)
    initial, metadata = case.build_local_state(points, fixture)
    proposed_dt_s = target_shear_increment / (shear_rate_s_inv * steps)
    mechanism = MechanismCase(
        "local_boundary_matrix",
        shear_rate_s_inv,
        SpatialMechanismControls(True, True),
    )
    trace, control = run_matched_local_strain_pair(
        initial, mechanism, metadata["dx_m"], proposed_dt_s,
        target_shear_increment,
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
    accepted_dt = np.asarray([item.accepted_dt_s for item in trace.steps])
    actual_increment = abs(final.applied_shear - initial.applied_shear)
    return {
        "temperature_K": temperature_K,
        "shear_rate_s_inv": shear_rate_s_inv,
        "density_ratio": density_ratio,
        "analytical_branch": metadata["branch"],
        "nominal_density_m2": metadata["nominal_density_m2"],
        "initial_mean_stress_Pa": metadata["initial_stress_Pa"],
        "proposed_dt_s": proposed_dt_s,
        "minimum_accepted_dt_s": float(np.min(accepted_dt)),
        "maximum_halvings": max(item.halvings for item in trace.steps),
        "accepted_steps": len(trace.steps),
        "accepted_duration_s": final.time_s,
        "applied_shear_increment": actual_increment,
        "final_mean_stress_Pa": final_step.equilibrium.mean_stress_Pa,
        "peak_mean_stress_Pa": float(np.max(np.abs(trace.mean_stress_Pa))),
        "final_local_stress_std_Pa": float(np.std(final_step.equilibrium.stress_x_Pa)),
        "maximum_temperature_K": float(np.max(trace.temperature_K)),
        "maximum_matched_temperature_excess_K": max(item.temperature_excess_K for item in history),
        "minimum_active_fraction": min(item.active_fraction for item in history),
        "maximum_softening_fraction": max(item.softening_fraction for item in history),
        "initial_child_order_fraction": float(np.mean(initial.eta_fields[1])),
        "final_child_order_fraction": float(np.mean(final.eta_fields[1])),
        "phase_change_l2": float(np.linalg.norm(final.eta_fields - initial.eta_fields)),
        "maximum_storage_limited_fraction": max(
            item.storage_limited_fraction for item in trace.steps
        ),
        "steps_with_storage_limiting": sum(
            item.storage_limited_fraction > 0.0 for item in trace.steps
        ),
        "localized": decision.localized,
        "failed_criteria": list(decision.failed_criteria),
        "maximum_absolute_global_closure_error_J_m3": max(
            abs(item.ledger.global_closure_error_J_m3) for item in trace.steps
        ),
        "maximum_absolute_thermal_closure_error_J_m3": max(
            abs(item.ledger.thermal_closure_error_J_m3) for item in trace.steps
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--target-shear", type=float, default=0.9)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--execution-site", default="local")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--rate", type=float)
    parser.add_argument("--density-ratio", type=float)
    args = parser.parse_args()
    if args.points < 8 or args.steps < 1 or args.target_shear <= 0.0:
        parser.error("invalid grid, step count, or target shear")
    selectors = (args.temperature, args.rate, args.density_ratio)
    if any(item is not None for item in selectors) and not all(
        item is not None for item in selectors
    ):
        parser.error("temperature, rate, and density-ratio must be supplied together")
    temperatures = (args.temperature,) if args.temperature is not None else (850.0, 950.0, 1050.0)
    rates = (args.rate,) if args.rate is not None else (4.5, 450.0, 45000.0)
    ratios = (args.density_ratio,) if args.density_ratio is not None else (0.5, 1.0, 2.0)
    fixture = SingleGliderDDDParameterization()
    records = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial_output = args.output.with_name(
        f"{args.output.stem}.partial{args.output.suffix}"
    )
    for temperature in temperatures:
        for rate in rates:
            for ratio in ratios:
                print(
                    f"starting T={temperature:g} K rate={rate:g} s^-1 "
                    f"density_ratio={ratio:g}",
                    flush=True,
                )
                records.append(
                    run_condition(
                        temperature, rate, ratio, args.points, args.steps,
                        args.target_shear, fixture,
                    )
                )
                partial_output.write_text(
                    json.dumps(
                        {
                            "schema": "asb-drx-local-antiplane-boundary-matrix/partial-v1",
                            "source_commit": args.source_commit,
                            "execution_site": args.execution_site,
                            "completed_records": records,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                print(f"completed records={len(records)}", flush=True)
    report = {
        "schema": "asb-drx-local-antiplane-boundary-matrix/v1",
        "source_commit": args.source_commit,
        "execution_site": args.execution_site,
        "scientific_disposition": (
            "single deterministic generic screening matrix; analytical extrapolation outside the "
            "source DDD rate; not material validation or a converged production regime map"
        ),
        "temperature_axis_K": temperatures,
        "shear_rate_axis_s_inv": rates,
        "density_ratio_axis": ratios,
        "grid_points": args.points,
        "steps": args.steps,
        "target_shear_increment": args.target_shear,
        "localization_criteria": LocalizationCriteria(0.4, 20.0, 0.1, 3.0, 3, 0.05).__dict__,
        "records": records,
        "localized_count": sum(item["localized"] for item in records),
        "explicit_limitations": [
            "one deterministic perturbation and no seed ensemble",
            "child order fraction is not a physical grain count or DRX classification",
            "periodic antiplane scalar mechanics is not full crystallographic elasticity",
            "DDD source campaign rate is 4.5 s^-1; higher rates are analytical extrapolations",
        ],
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
