#!/usr/bin/env python3
"""Test whether complete Arrhenius reactions preserve a formed signed band."""

from __future__ import annotations

from dataclasses import fields, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from full_model.analysis.run_v66_production_groma_tangent import candidate_parameters
from full_model.production.common_tensorial_wall import (
    CommonWallDriving, CommonWallState, accepted_euler_step,
    resolved_driving_components, wall_total_free_energy_density_J_m3,
)
from full_model.production.nonlocal_elasticity import elastic_energy_density
from full_model.production.tensorial_nye import (
    bcc_four_family_systems, make_junction_topology,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT/"full_model"/"verification"/"v66_production_groma_loading"
          /"final_state.npz")
OUT = ROOT/"full_model"/"verification"/"v66_groma_reaction_survival"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_state(path):
    with np.load(path) as data:
        state = CommonWallState(**{
            item.name: np.asarray(data[item.name]).copy()
            for item in fields(CommonWallState)})
        shear = float(data["engineering_shear"])
    return state, shear


def _homogenize(state):
    arrays = {}
    for item in fields(state):
        value = np.asarray(getattr(state, item.name))
        mean = np.mean(value, axis=(0, 1), keepdims=True)
        arrays[item.name] = np.broadcast_to(mean, value.shape).copy()
    arrays["family_nye_m1"][:] = 0.0
    return CommonWallState(**arrays)


def _polarization(plus, minus):
    total = plus+minus
    return np.divide(plus-minus, total, out=np.zeros_like(total), where=total > 0.0)


def _observables(state):
    reservoirs = {}
    for name in ("mobile", "forest", "wall"):
        plus = getattr(state, f"{name}_plus_m2")
        minus = getattr(state, f"{name}_minus_m2")
        total = np.sum(plus+minus, axis=2)
        polarization = _polarization(plus, minus)
        reservoirs[name] = {
            "mean_density_m2": float(np.mean(total)),
            "density_contrast": float(np.std(total)/max(np.mean(total), 1.0)),
            "polarization_rms": float(np.sqrt(np.mean(polarization**2))),
            "polarization_max_abs": float(np.max(np.abs(polarization))),
        }
    nye = np.sum(state.family_nye_m1, axis=2)
    return {
        "temperature_mean_K": float(np.mean(state.temperature_K)),
        "temperature_range_K": float(np.ptp(state.temperature_K)),
        "orientation_span_deg": float(np.ptp(state.orientation_rad)*180/np.pi),
        "nye_rms_m1": float(np.sqrt(np.mean(nye*nye))),
        "wall_order_mean": float(np.mean(state.wall_order)),
        "wall_order_max": float(np.max(state.wall_order)),
        "reservoirs": reservoirs,
    }


def _energies(state, driving, systems, topologies, parameters):
    resolved = resolved_driving_components(
        state, driving, systems, topologies, parameters)
    elastic = float(np.mean(elastic_energy_density(
        resolved["stress_tensor_Pa"], resolved["compatible_strain"],
        resolved["eigenstrain"])))
    defect = float(np.mean(wall_total_free_energy_density_J_m3(
        state, parameters, topologies, systems)))
    thermal = float(parameters.volumetric_heat_capacity_J_m3_K
                    *np.mean(state.temperature_K))
    return elastic, defect, thermal


def _run(initial, driving, systems, topologies, parameters, maximum_dt, *,
         hold_time=1.0e-3, record_interval=2.0e-5):
    state = initial
    initial_content = {
        name: float(np.sum(getattr(state, name))) for name in (
            "mobile_plus_m2", "mobile_minus_m2", "forest_plus_m2",
            "forest_minus_m2", "wall_plus_m2", "wall_minus_m2",
            "junction_m2")}
    e0 = _energies(state, driving, systems, topologies, parameters)
    records = [{"time_s": 0.0, **_observables(state)}]
    time_s = 0.0
    next_record = record_interval
    accepted_steps = 0
    limited_steps = 0
    while time_s < hold_time-1e-15:
        requested = min(maximum_dt, hold_time-time_s, next_record-time_s)
        updated, _, scale = accepted_euler_step(
            state, driving, systems, topologies, parameters, requested)
        accepted = requested*scale
        if accepted <= 0.0:
            raise RuntimeError("complete reaction step made no progress")
        state = updated
        time_s += accepted
        accepted_steps += 1
        limited_steps += int(scale < 1.0)
        if next_record-time_s < 2e-14:
            time_s = next_record
            records.append({"time_s": time_s, **_observables(state)})
            next_record += record_interval
    e1 = _energies(state, driving, systems, topologies, parameters)
    # Fixed total strain and adiabatic temperature: no external work crosses
    # the boundary during this survival hold.
    changes = [right-left for left, right in zip(e0, e1)]
    content_final = {
        name: float(np.sum(getattr(state, name))) for name in initial_content}
    line_initial = sum(initial_content[name] for name in initial_content)
    line_final = sum(content_final[name] for name in content_final)
    return state, {
        "maximum_dt_s": maximum_dt,
        "accepted_steps": accepted_steps,
        "limited_steps": limited_steps,
        "records": records,
        "final": records[-1],
        "energy": {
            "elastic_change_J_m3": changes[0],
            "defect_change_J_m3": changes[1],
            "thermal_change_J_m3": changes[2],
            "fixed_strain_adiabatic_residual_J_m3": sum(changes),
            "relative_residual": abs(sum(changes))/max(
                *(abs(item) for item in changes), 1.0),
        },
        # Junction multiplicity and annihilation make raw scalar line content
        # nonconservative; report change rather than mislabel it a residual.
        "raw_reservoir_content_change_fraction": (
            (line_final-line_initial)/line_initial),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    patterned, shear = _load_state(SOURCE)
    homogeneous = _homogenize(patterned)
    systems = bcc_four_family_systems()
    topologies = (make_junction_topology(
        systems, 0, 1, 1, -1, line_tension_J_m=0.0),)
    parameters, gb2 = candidate_parameters(2.0e-7)
    driving = CommonWallDriving(mean_strain=np.array(
        [[0.0, .5*shear], [.5*shear, 0.0]]))
    _, patterned_coarse = _run(
        patterned, driving, systems, topologies, parameters, 2.0e-6)
    _, patterned_fine = _run(
        patterned, driving, systems, topologies, parameters, 1.0e-6)
    _, homogeneous_coarse = _run(
        homogeneous, driving, systems, topologies, parameters, 2.0e-6)

    keys = (
        ("orientation_span_deg",), ("nye_rms_m1",),
        ("wall_order_max",),
        ("reservoirs", "mobile", "density_contrast"),
        ("reservoirs", "mobile", "polarization_rms"),
        ("reservoirs", "wall", "density_contrast"),
        ("reservoirs", "wall", "polarization_rms"),
    )
    convergence = {}
    for key in keys:
        a = patterned_coarse["final"]
        b = patterned_fine["final"]
        for item in key:
            a = a[item]; b = b[item]
        convergence["/".join(key)] = abs(a-b)/max(abs(b), 1e-300)

    figure = OUT/"reaction_survival.png"
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.2), constrained_layout=True)
    for label, case in (("patterned", patterned_fine),
                        ("homogeneous", homogeneous_coarse)):
        time = [row["time_s"] for row in case["records"]]
        axes[0].plot(time, [row["reservoirs"]["mobile"]["polarization_rms"]
                            for row in case["records"]], label=label)
        axes[1].plot(time, [row["reservoirs"]["wall"]["polarization_rms"]
                            for row in case["records"]], label=label)
        axes[2].plot(time, [row["wall_order_max"] for row in case["records"]],
                     label=label)
    axes[0].set(xlabel="hold time (s)", ylabel="mobile polarization RMS")
    axes[1].set(xlabel="hold time (s)", ylabel="wall polarization RMS")
    axes[2].set(xlabel="hold time (s)", ylabel="maximum wall order")
    axes[0].legend(frameon=False)
    fig.savefig(figure, dpi=180)
    plt.close(fig)

    result = {
        "schema": "v66-groma-reaction-survival-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(SOURCE.relative_to(ROOT)),
        "source_sha256": _sha(SOURCE),
        "hold_time_s": 1.0e-3, "fixed_engineering_shear": shear,
        "parameters": {"A": 1.25, "D": .5,
                       "reduced_modulus_b2_J_m": gb2},
        "cases": {"patterned_coarse": patterned_coarse,
                  "patterned_fine": patterned_fine,
                  "homogeneous_control": homogeneous_coarse},
        "coarse_fine_relative_differences": convergence,
        "decision": {
            "five_percent_timestep_convergence": bool(max(
                convergence.values()) < .05),
            "signed_pattern_survives_complete_reactions": bool(
                patterned_fine["final"]["reservoirs"]["wall"]["polarization_rms"]
                > homogeneous_coarse["final"]["reservoirs"]["wall"]["polarization_rms"]),
            "physical_wall_qualified": False,
            "recrystallization_qualified": False,
        },
        "claim_boundary": (
            "post-formation survival test; reactions were not active during formation"),
        "artifacts": {str(figure.relative_to(ROOT)): _sha(figure)},
    }
    path = OUT/"result.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"finals": {
        "patterned": patterned_fine["final"],
        "homogeneous": homogeneous_coarse["final"]},
        "convergence": convergence, "decision": result["decision"],
        "energy": patterned_fine["energy"]}, indent=2))


if __name__ == "__main__":
    main()
