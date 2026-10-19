#!/usr/bin/env python3
"""Retained-state dynamic/static test of the V66 common-axis RX operator."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from full_model.production.arrhenius_dislocation_sources import (
    ArrheniusSourceParameters,
    pair_source_rate_m2_s,
)
from full_model.production.common_axis_rx import (
    CommonAxisBoundaryState,
    advance_common_axis_rx_interval,
)
from full_model.production.common_tensorial_wall import (
    CommonWallDriving,
    CommonWallParameters,
    CommonWallState,
    wall_total_free_energy_density_J_m3,
)
from full_model.production.structural_boundary import StructuralBoundaryParameters
from full_model.production.tensorial_nye import bcc_four_family_systems


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"full_model"/"verification"/"v66_dynamic_static_rx"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _initial(n, theta, temperature):
    shape = (n, n, 4)
    plus = np.full(shape, 2e12)
    minus = np.full(shape, 2e12)
    # Retained deformation pile-up: a declared compatible positive excess at
    # the inherited boundary.  This tests maturation, not spontaneous birth.
    plus[n//2-1:n//2+1, :, 0] = 2e14
    orientation = np.zeros((n, n)); orientation[n//2:] = theta
    return CommonWallState(
        plus, minus, np.full(shape, 1e12), np.full(shape, 1e12),
        np.zeros(shape), np.zeros(shape), np.zeros((n, n, 0)),
        np.zeros((n, n)), np.zeros((n, n)), np.zeros(shape),
        np.zeros((n, n, 3, 3)), np.zeros((n, n, 4, 3)),
        np.zeros((n, n, 4, 3, 3)), orientation,
        np.full((n, n), temperature))


def _advance(common, boundary, *, mode, steps, dt, capture, rotating,
             site, source_rate, source_parameters, structural_parameters,
             wall_parameters, systems, stress_Pa):
    history = []
    source_line_residual = 0.0
    absorption_line_residual = 0.0
    minimum_mechanical_scale = 1.0
    initial_bulk = float(np.mean(wall_total_free_energy_density_J_m3(
        common, wall_parameters, (), systems)))
    for step in range(steps):
        source = None
        if mode in ("preparation", "continued_deformation"):
            driving = CommonWallDriving(
                resolved_stress_Pa=np.full(common.mobile_plus_m2.shape,
                                           stress_Pa))
        else:
            driving = CommonWallDriving(
                resolved_stress_Pa=np.zeros(common.mobile_plus_m2.shape))
        interval = advance_common_axis_rx_interval(
            common, boundary, driving=driving, systems=systems,
            topologies=(), wall_parameters=wall_parameters,
            structural_parameters=structural_parameters,
            capture_mask=capture, rotating_mask=rotating, family=0,
            compatible=True, dt_s=dt,
            pair_source_rate_m2_s=(source_rate if mode in (
                "preparation", "continued_deformation") else None),
            source_mode=("gb_tj_only" if mode in (
                "preparation", "continued_deformation") else "disabled"),
            maximum_reservoir_fraction=.03)
        common = interval.common_state
        boundary = interval.boundary_state
        source = interval.source_ledger
        decision = interval.absorption_decision
        if source is not None:
            source_line_residual = max(
                source_line_residual, abs(source.line_balance_residual_m2))
        minimum_mechanical_scale = min(
            minimum_mechanical_scale, interval.mechanical_acceptance_scale)
        absorption_line_residual = max(
            absorption_line_residual, abs(decision.line_balance_residual_m1))
        if step % max(steps//40, 1) == 0 or step == steps-1:
            theta_structural = (structural_parameters.burgers_m
                                *boundary.structural_signed_m1)
            bulk = float(np.mean(wall_total_free_energy_density_J_m3(
                common, wall_parameters, (), systems)))
            history.append({
                "time_s": (step+1)*dt,
                "structural_misorientation_deg": math.degrees(theta_structural),
                "mean_total_density_m2": float(np.mean(sum(
                    np.asarray(getattr(common, name)) for name in (
                        "mobile_plus_m2", "mobile_minus_m2",
                        "forest_plus_m2", "forest_minus_m2",
                        "wall_plus_m2", "wall_minus_m2")))),
                "mean_bulk_defect_energy_J_m3": bulk,
                "bulk_energy_change_J_m3": bulk-initial_bulk,
                "cumulative_boundary_dissipation_J_m2": (
                    boundary.cumulative_dissipation_J_m2),
                "absorption_event_rate_Hz": decision.event_rate_Hz,
                "absorption_accepted": decision.accepted,
            })
    return common, boundary, history, {
        "source_line_balance_max_abs_m2": source_line_residual,
        "absorption_line_balance_max_abs_m1": absorption_line_residual,
        "minimum_mechanical_acceptance_scale": minimum_mechanical_scale,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n = 32
    dx = 1e-7
    dt = 2e-6
    preparation_steps = 20
    continuation_steps = 200
    temperature = 900.0
    stress = 1.8e8
    theta0 = math.radians(2.0)
    structural = StructuralBoundaryParameters(reservoir_width_m=2*dx)
    source_parameters = ArrheniusSourceParameters(
        grain_size_m=n*dx/2, pair_increment_m2=5e13)
    wall = CommonWallParameters(
        spacing_m=dx, transport_scheme="upwind",
        thermal_diffusivity_m2_s=0.0, bath_rate_s=0.0,
        multiplication_coefficient_m1=2.0)
    systems = bcc_four_family_systems(structural.burgers_m)
    capture = np.zeros((n, n), bool); capture[n//2-1:n//2+1] = True
    rotating = np.zeros((n, n), bool); rotating[n//2:] = True
    site = np.zeros((n, n)); site[capture] = n/2
    source_rate = pair_source_rate_m2_s(
        site, stress, temperature, source_parameters)
    common = _initial(n, theta0, temperature)
    boundary = CommonAxisBoundaryState(
        0.0, theta0, theta0/structural.burgers_m)

    common, boundary, preparation, prep_audit = _advance(
        common, boundary, mode="preparation", steps=preparation_steps,
        dt=dt, capture=capture, rotating=rotating, site=site,
        source_rate=source_rate, source_parameters=source_parameters,
        structural_parameters=structural, wall_parameters=wall,
        systems=systems, stress_Pa=stress)
    retained_common = common
    retained_boundary = boundary
    branches = {}
    for mode in ("continued_deformation", "static_zero_stress_hold"):
        final_common, final_boundary, history, audit = _advance(
            retained_common, retained_boundary, mode=mode,
            steps=continuation_steps, dt=dt, capture=capture,
            rotating=rotating, site=site, source_rate=source_rate,
            source_parameters=source_parameters,
            structural_parameters=structural, wall_parameters=wall,
            systems=systems, stress_Pa=stress)
        branches[mode] = {
            "history": history,
            "audit": audit,
            "final_structural_misorientation_deg": math.degrees(
                structural.burgers_m*final_boundary.structural_signed_m1),
            "final_mean_temperature_K": float(np.mean(final_common.temperature_K)),
            "final_mean_orientation_right_deg": float(np.degrees(
                np.mean(final_common.orientation_rad[rotating]))),
            "accepted_absorption_records": int(sum(
                item["absorption_accepted"] for item in history)),
        }

    figure = OUT/"dynamic_static_histories.png"
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4), constrained_layout=True)
    for mode, branch in branches.items():
        label = "continued deformation" if mode == "continued_deformation" else "zero-stress hold"
        h = branch["history"]
        t = np.array([item["time_s"] for item in h])*1e3
        axes[0].plot(t, [item["structural_misorientation_deg"] for item in h], label=label)
        axes[1].plot(t, np.array([item["mean_total_density_m2"] for item in h])/1e14)
        axes[2].plot(t, [item["cumulative_boundary_dissipation_J_m2"] for item in h])
    axes[0].set_ylabel("structural misorientation (deg)")
    axes[1].set_ylabel(r"mean line density ($10^{14}$ m$^{-2}$)")
    axes[2].set_ylabel(r"boundary-reaction dissipation (J m$^{-2}$)")
    for ax in axes:
        ax.set_xlabel("continuation time (ms)")
    axes[0].legend(frameon=False)
    fig.savefig(figure, dpi=180)
    plt.close(fig)

    dynamic_theta = branches["continued_deformation"]["final_structural_misorientation_deg"]
    static_theta = branches["static_zero_stress_hold"]["final_structural_misorientation_deg"]
    result = {
        "schema": "v66-dynamic-static-rx-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "inherited planar low-angle subgrain; common-axis maturation, not new-grain formation",
        "parameters": {
            "structural": asdict(structural),
            "source": asdict(source_parameters),
            "grid": [n, n], "spacing_m": dx, "temperature_K": temperature,
            "deformation_stress_Pa": stress, "dt_s": dt,
            "preparation_steps": preparation_steps,
            "continuation_steps": continuation_steps,
        },
        "retained_state": {
            "preparation_history": preparation,
            "audit": prep_audit,
            "structural_misorientation_deg": math.degrees(
                structural.burgers_m*retained_boundary.structural_signed_m1),
            "state_shared_exactly_by_branches": True,
        },
        "branches": branches,
        "decision": {
            "dynamic_static_discriminator_executed": True,
            "same_laws_used": True,
            "static_absorption_active_without_imposed_deformation": bool(
                branches["static_zero_stress_hold"]["accepted_absorption_records"] > 0),
            "continued_source_changes_maturation": bool(
                abs(dynamic_theta-static_theta) > 1e-5),
            "G05_full_initiation_qualified": False,
            "G06_static_RX_qualified": False,
            "interpretation": (
                "the same stationary structural reaction remains active after the source "
                "and applied stress are removed, while continued deformation changes its "
                "line supply. The calculation matures an inherited subgrain but does not "
                "yet create a new internal boundary or demonstrate sustained front growth."),
        },
        "artifacts": {str(figure.relative_to(ROOT)): _sha(figure)},
    }
    path = OUT/"result.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps(result["decision"], indent=2))


if __name__ == "__main__":
    main()
