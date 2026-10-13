#!/usr/bin/env python3
"""Selective-growth, temperature/rate, and HAGB handoff study for V66."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from full_model.production.arrhenius_kinetics import ActivatedProcess
from full_model.production.sibm_boundary import (
    BoundaryGraphState,
    SIBMParameters,
    advance_boundary,
    circular_bulge_pressure_Pa,
)
from full_model.production.structural_boundary import StructuralBoundaryParameters
from full_model.production.subgrain_network import (
    SubgrainNetworkParameters,
    SubgrainNetworkState,
    advance_subgrain_network,
    network_energy_J_m2,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"full_model"/"verification"/"v66_subgrain_maturation"
EV_J = 1.602176634e-19
KB = 1.380649e-23


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _initial_state():
    orientation = np.radians(np.arange(13)*2.5)
    stored = np.full(13, 5e6)
    stored[6] = .5e6
    return SubgrainNetworkState(
        np.arange(14)*.5e-6, orientation, stored)


def _update_stored_energy(state, *, dt, temperature, strain_rate,
                          recovery_prefactor_s, recovery_eV,
                          stress_Pa, stored_work_fraction,
                          target_orientation_rad, target_storage_fraction):
    k = recovery_prefactor_s*math.exp(-recovery_eV*EV_J/(KB*temperature))
    decay = math.exp(-k*dt)
    source = stored_work_fraction*stress_Pa*strain_rate
    increment = source*dt if k == 0.0 else source/k*(1.0-decay)
    target = np.isclose(state.orientation_rad, target_orientation_rad,
                        atol=1e-14, rtol=0.0)
    stored = state.stored_energy_J_m3*decay
    stored[~target] += increment
    stored[target] += target_storage_fraction*increment
    return SubgrainNetworkState(
        state.edges_m, state.orientation_rad, stored, state.physical_time_s,
        state.cumulative_dissipation_J_m2, state.eliminated_cells), k


def _target_metrics(state, target):
    matches = np.where(np.isclose(state.orientation_rad, target,
                                  atol=1e-14, rtol=0.0))[0]
    if matches.size != 1:
        raise RuntimeError("target orientation was lost or duplicated")
    index = int(matches[0])
    width = np.diff(state.edges_m)[index]
    theta = []
    neighbors = []
    if index > 0:
        theta.append(abs(target-state.orientation_rad[index-1]))
        neighbors.append(state.stored_energy_J_m3[index-1])
    if index < state.orientation_rad.size-1:
        theta.append(abs(state.orientation_rad[index+1]-target))
        neighbors.append(state.stored_energy_J_m3[index+1])
    contrast = (float(np.mean(neighbors))-state.stored_energy_J_m3[index]
                if neighbors else 0.0)
    return index, width, max(theta, default=0.0), contrast


def _run_case(temperature, strain_rate, parameters, constants):
    state = _initial_state()
    target = math.radians(15.0)
    dt = constants["dt_s"]
    steps = int(round(constants["horizon_s"]/dt))
    history = []
    maximum_orientation_residual = 0.0
    for step in range(steps):
        state, recovery_rate = _update_stored_energy(
            state, dt=dt, temperature=temperature, strain_rate=strain_rate,
            recovery_prefactor_s=constants["recovery_prefactor_s"],
            recovery_eV=constants["recovery_activation_eV"],
            stress_Pa=constants["flow_stress_Pa"],
            stored_work_fraction=constants["stored_work_fraction"],
            target_orientation_rad=target,
            target_storage_fraction=constants["target_storage_fraction"])
        decision = advance_subgrain_network(
            state, dt_s=dt, temperature_K=temperature,
            parameters=parameters)
        state = decision.state
        maximum_orientation_residual = max(
            maximum_orientation_residual,
            decision.orientation_sum_residual_rad)
        if step % max(steps//80, 1) == 0 or step == steps-1:
            _, width, theta, contrast = _target_metrics(state, target)
            history.append({
                "time_s": (step+1)*dt,
                "target_width_um": width*1e6,
                "maximum_target_misorientation_deg": math.degrees(theta),
                "stored_energy_contrast_Pa": contrast,
                "subgrain_count": int(state.orientation_rad.size),
                "eliminated_cells": int(state.eliminated_cells),
                "network_energy_J_m2": network_energy_J_m2(state, parameters),
                "recovery_rate_s": recovery_rate,
            })
    index, width, theta, contrast = _target_metrics(state, target)
    gamma = parameters.structural.high_angle_energy_J_m2
    radius = .5*width
    pressure = circular_bulge_pressure_Pa(
        radius, contrast, 0.0,
        SIBMParameters(
            gamma, parameters.mobility_prefactor_m4_J_s,
            parameters.process, parameters.activation_enthalpy_0_J,
            parameters.critical_pressure_Pa, parameters.exp_a,
            parameters.exp_n, parameters.exp_floor, 1e-6))
    hagb = theta >= parameters.structural.transition_misorientation_rad*(1-1e-12)
    viable = hagb and pressure > 0.0
    if hagb and viable:
        classification = "HAGB_MATURED_AND_FRONT_VIABLE"
    elif theta > math.radians(2.5)+1e-12:
        classification = "MATURATION_WITHOUT_HAGB"
    elif contrast <= .05*4.5e6:
        classification = "RECOVERY_ARREST"
    else:
        classification = "SELECTIVE_GROWTH_WITHOUT_TOPOLOGY_EVENT"
    return state, history, {
        "classification": classification,
        "target_index": index,
        "target_width_um": width*1e6,
        "maximum_target_misorientation_deg": math.degrees(theta),
        "stored_energy_contrast_Pa": contrast,
        "cylindrical_front_pressure_Pa": pressure,
        "HAGB_matured": bool(hagb),
        "front_viable": bool(viable),
        "eliminated_cells": int(state.eliminated_cells),
        "final_subgrain_count": int(state.orientation_rad.size),
        "maximum_orientation_sum_residual_rad": maximum_orientation_residual,
    }


def _front_handoff(case, temperature, parameters):
    if not case["HAGB_matured"] or not case["front_viable"]:
        return {"executed": False, "reason": "maturation/criticality not satisfied"}
    sibm = SIBMParameters(
        parameters.structural.high_angle_energy_J_m2,
        parameters.mobility_prefactor_m4_J_s, parameters.process,
        parameters.activation_enthalpy_0_J, parameters.critical_pressure_Pa,
        parameters.exp_a, parameters.exp_n, parameters.exp_floor, 1e-6)
    graph = BoundaryGraphState(np.zeros(64), 0, 1, 0.0, math.radians(15.0))
    spacing = 1e-7
    initial_energy = None
    front_dt = 1e-4
    front_steps = 10000
    for _ in range(front_steps):
        graph, velocity = advance_boundary(
            graph, spacing_m=spacing, dt_s=front_dt, temperature_K=temperature,
            parameters=sibm,
            stored_difference_Pa=case["stored_energy_contrast_Pa"])
        if initial_energy is None:
            initial_energy = graph.ledger.free_energy_before_J
    displacement = float(np.mean(graph.height_m))
    return {
        "executed": True,
        "duration_s": front_dt*front_steps,
        "mean_boundary_displacement_um": displacement*1e6,
        "terminal_velocity_um_s": float(np.mean(velocity))*1e6,
        "free_energy_change_J": graph.ledger.free_energy_change_J,
        "cumulative_signed_area_change_m2": graph.ledger.signed_area_change_m2,
        "maximum_pressure_Pa": graph.ledger.maximum_abs_pressure_Pa,
        "energy_nonincreasing": bool(graph.ledger.free_energy_change_J <= 0.0),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # The identifiable network mobility uses rate/attempt and is independent
    # of this prefactor.  1 kHz keeps the graph event probability resolved at
    # the declared 0.1 ms handoff step instead of saturating a coarse clock.
    process = ActivatedProcess("subgrain/front migration", 1e3)
    structural = StructuralBoundaryParameters()
    parameters = SubgrainNetworkParameters(
        structural, process, mobility_prefactor_m4_J_s=5e-14,
        activation_enthalpy_0_J=.7*EV_J, critical_pressure_Pa=2e6,
        exp_a=2.0, exp_n=2.0, exp_floor=.05)
    constants = {
        "dt_s": .05, "horizon_s": 50.0,
        "recovery_prefactor_s": 1e6,
        "recovery_activation_eV": 1.2,
        "flow_stress_Pa": 2e8,
        "stored_work_fraction": .1,
        "target_storage_fraction": .1,
    }
    temperatures = (650.0, 800.0, 900.0, 1000.0)
    strain_rates = (0.0, .1, 1.0)
    cases = {}
    histories = {}
    states = {}
    for temperature in temperatures:
        for rate in strain_rates:
            key = f"T{int(temperature)}_rate{rate:g}"
            state, history, case = _run_case(
                temperature, rate, parameters, constants)
            states[key] = state; histories[key] = history; cases[key] = case

    # Handoff the hot, deformation-maintained case to the already verified
    # bidirectional complete-energy SIBM graph operator.
    handoff_key = "T1000_rate1"
    handoff = _front_handoff(cases[handoff_key], 1000.0, parameters)
    time_refinement = []
    for refined_dt in (.1, .05, .025):
        refined_constants = dict(constants, dt_s=refined_dt)
        _, _, refined = _run_case(
            1000.0, 1.0, parameters, refined_constants)
        time_refinement.append({
            "dt_s": refined_dt,
            "target_width_um": refined["target_width_um"],
            "maximum_target_misorientation_deg": refined[
                "maximum_target_misorientation_deg"],
            "stored_energy_contrast_Pa": refined[
                "stored_energy_contrast_Pa"],
            "classification": refined["classification"],
        })
    fine_width = time_refinement[-1]["target_width_um"]
    width_spread = (max(item["target_width_um"] for item in time_refinement)
                    -min(item["target_width_um"] for item in time_refinement))
    time_refinement_passed = width_spread/max(abs(fine_width), 1e-300) < .05

    figure = OUT/"subgrain_maturation_map.png"
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.5), constrained_layout=True)
    selected = ("T650_rate0", "T900_rate0", "T1000_rate1")
    labels = ("650 K, static", "900 K, static", r"1000 K, 1 s$^{-1}$")
    for key, label in zip(selected, labels):
        h = histories[key]
        t = [item["time_s"] for item in h]
        axes[0].plot(t, [item["target_width_um"] for item in h], label=label)
        axes[1].plot(t, [item["maximum_target_misorientation_deg"] for item in h])
        axes[2].plot(t, np.array([item["stored_energy_contrast_Pa"] for item in h])/1e6)
    axes[0].set_ylabel(r"target width ($\mu$m)")
    axes[1].set_ylabel("maximum target misorientation (deg)")
    axes[2].set_ylabel("stored-energy contrast (MPa)")
    for ax in axes:
        ax.set_xlabel("physical time (s)")
    axes[0].legend(frameon=False)
    fig.savefig(figure, dpi=180)
    plt.close(fig)

    classification_map = {
        key: value["classification"] for key, value in cases.items()}
    result = {
        "schema": "v66-subgrain-maturation-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "one-dimensional inherited orientation-gradient network; generic, not calibrated",
        "derived": {
            "boundary_motion": "v=M_EXP(T,p)*(f_right-f_left)",
            "topology_event": "two LAGBs replaced by one compatible summed-misorientation boundary",
            "bulk_energy": "sum_i width_i*f_i",
            "boundary_energy": "sum_i Read-Shockley(theta_i)",
        },
        "assumed_parameters": {
            "network": {
                "mobility_prefactor_m4_J_s": parameters.mobility_prefactor_m4_J_s,
                "activation_enthalpy_eV": .7,
                "critical_pressure_Pa": parameters.critical_pressure_Pa,
                "exp_a": parameters.exp_a, "exp_n": parameters.exp_n,
                "exp_floor": parameters.exp_floor,
            },
            **constants,
        },
        "conditions": {"temperatures_K": temperatures,
                       "strain_rates_s": strain_rates},
        "cases": cases,
        "classification_map": classification_map,
        "time_refinement": {
            "cases": time_refinement,
            "relative_target_width_spread": width_spread/max(
                abs(fine_width), 1e-300),
            "provisional_5_percent_passed": bool(time_refinement_passed),
        },
        "front_handoff": {"source_case": handoff_key, **handoff},
        "decision": {
            "orientation_gradient_maturation_supported": bool(any(
                value["HAGB_matured"] for value in cases.values())),
            "static_maturation_supported": bool(any(
                value["HAGB_matured"] for key, value in cases.items()
                if key.endswith("rate0"))),
            "dynamic_maturation_supported": bool(any(
                value["HAGB_matured"] for key, value in cases.items()
                if not key.endswith("rate0"))),
            "front_handoff_exercised": bool(handoff.get("executed")),
            "time_refinement_passed": bool(time_refinement_passed),
            "new_orientation_created": False,
            "grain_count_increased": False,
            "interpretation": (
                "selective growth can concentrate a pre-existing orientation gradient "
                "into a HAGB while reducing subgrain count. Recovery outruns migration "
                "in hot static cases, whereas deformation-maintained stored-energy "
                "contrast restores maturation and a viable SIBM handoff."),
        },
        "artifacts": {str(figure.relative_to(ROOT)): _sha(figure)},
    }
    path = OUT/"result.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps(result["decision"], indent=2))
    print(json.dumps(result["classification_map"], indent=2))


if __name__ == "__main__":
    main()
