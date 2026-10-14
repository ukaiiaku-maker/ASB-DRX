#!/usr/bin/env python3
"""Strain-supplied nonlinear test of the V66 production Groma branch."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from full_model.analysis.run_v66_interaction_stability import _state
from full_model.analysis.run_v66_production_groma_nonlinear import (
    _combine, _elastic_energy, _map, _observables, _rhs,
)
from full_model.analysis.run_v66_production_groma_tangent import candidate_parameters
from full_model.production.common_tensorial_wall import (
    CommonWallDriving, resolved_driving_components,
    wall_free_energy_density_J_m3,
)
from full_model.production.tensorial_nye import (
    bcc_four_family_systems, make_junction_topology,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"full_model"/"verification"/"v66_production_groma_loading"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n = 24
    spacing = 2.0e-7
    initial_shear = 1.65e-3
    shear_rate = 3.1259408471629676e-7
    systems = bcc_four_family_systems()
    topologies = (make_junction_topology(
        systems, 0, 1, 1, -1, line_tension_J_m=0.0),)
    parameters, gb2 = candidate_parameters(spacing)
    state = replace(_state(n), temperature_K=np.full((n, n), 900.0))
    rng = np.random.default_rng(66024)
    plus_noise = rng.normal(size=state.mobile_plus_m2.shape)
    minus_noise = rng.normal(size=state.mobile_minus_m2.shape)
    plus_noise -= np.mean(plus_noise, axis=(0, 1), keepdims=True)
    minus_noise -= np.mean(minus_noise, axis=(0, 1), keepdims=True)
    state = replace(
        state, mobile_plus_m2=state.mobile_plus_m2*(1.0+1e-3*plus_noise),
        mobile_minus_m2=state.mobile_minus_m2*(1.0+1e-3*minus_noise))
    strain_rate = np.array([[0.0, .5*shear_rate], [.5*shear_rate, 0.0]])

    def driving(time_s):
        shear = initial_shear+shear_rate*time_s
        return CommonWallDriving(mean_strain=np.array(
            [[0.0, .5*shear], [.5*shear, 0.0]]))

    def stage(current, time_s):
        drive = driving(time_s)
        rate, plastic_power, heat_rate = _rhs(
            current, drive, systems, topologies, parameters)
        resolved = resolved_driving_components(
            current, drive, systems, topologies, parameters)
        boundary_power = float(np.mean(np.sum(
            resolved["stress_tensor_Pa"]*strain_rate, axis=(-2, -1))))
        mean_shear_stress = float(np.mean(resolved["stress_tensor_Pa"][..., 0, 1]))
        return rate, plastic_power, heat_rate, boundary_power, mean_shear_stress

    def rk4(current, time_s, dt):
        k1, p1, h1, w1, _ = stage(current, time_s)
        k2, p2, h2, w2, _ = stage(
            _combine(current, k1, .5*dt), time_s+.5*dt)
        k3, p3, h3, w3, _ = stage(
            _combine(current, k2, .5*dt), time_s+.5*dt)
        k4, p4, h4, w4, _ = stage(
            _combine(current, k3, dt), time_s+dt)
        increment = _map(k1, lambda name, value: (
            value+2.0*getattr(k2, name)+2.0*getattr(k3, name)
            +getattr(k4, name))/6.0)
        quadrature = lambda a, b, c, d: dt*(a+2*b+2*c+d)/6.0
        return (_combine(current, increment, dt),
                quadrature(p1, p2, p3, p4),
                quadrature(h1, h2, h3, h4),
                quadrature(w1, w2, w3, w4))

    def record(current, time_s):
        _, _, _, _, shear_stress = stage(current, time_s)
        return {"time_s": time_s,
                "engineering_shear": initial_shear+shear_rate*time_s,
                "mean_shear_stress_Pa": shear_stress,
                **_observables(current, spacing)}

    drive0 = driving(0.0)
    initial_content = np.sum(
        state.mobile_plus_m2+state.mobile_minus_m2, axis=(0, 1))
    initial_defect = float(np.mean(wall_free_energy_density_J_m3(
        state, parameters, topologies, systems)))
    initial_elastic = _elastic_energy(
        state, drive0, systems, topologies, parameters)
    records = [record(state, 0.0)]
    time_s = 0.0
    dt = 10.0
    next_record = 500.0
    plastic_work = heat = boundary_work = 0.0
    rejected = 0
    termination_reason = "target_time_reached"
    while time_s < 12000.0-1e-12:
        trial_dt = min(dt, 12000.0-time_s, next_record-time_s)
        try:
            candidate, p_inc, h_inc, w_inc = rk4(state, time_s, trial_dt)
            relative_increment = max(
                float(np.max(np.abs(candidate.mobile_plus_m2-state.mobile_plus_m2)
                             /state.mobile_plus_m2)),
                float(np.max(np.abs(candidate.mobile_minus_m2-state.mobile_minus_m2)
                             /state.mobile_minus_m2)))
            if (np.min(candidate.mobile_plus_m2) <= 0.0
                    or np.min(candidate.mobile_minus_m2) <= 0.0
                    or relative_increment > .03):
                raise FloatingPointError("inadmissible transport trial")
            candidate.validate(systems, topologies)
        except (FloatingPointError, ValueError):
            dt *= .5
            rejected += 1
            if dt < 1e-3:
                termination_reason = "loss_of_unprojected_state_below_1_ms_step"
                break
            continue
        state = candidate
        time_s += trial_dt
        plastic_work += p_inc
        heat += h_inc
        boundary_work += w_inc
        if relative_increment < .005:
            dt = min(10.0, 1.25*dt)
        if abs(time_s-next_record) < 1e-9:
            records.append(record(state, time_s))
            print(json.dumps(records[-1]), flush=True)
            next_record += 500.0
    if records[-1]["time_s"] != time_s:
        records.append(record(state, time_s))

    final_content = np.sum(
        state.mobile_plus_m2+state.mobile_minus_m2, axis=(0, 1))
    final_defect = float(np.mean(wall_free_energy_density_J_m3(
        state, parameters, topologies, systems)))
    final_elastic = _elastic_energy(
        state, driving(time_s), systems, topologies, parameters)
    defect_change = final_defect-initial_defect
    elastic_change = final_elastic-initial_elastic
    first_law = boundary_work-elastic_change-defect_change-heat
    mechanical_residual = boundary_work-elastic_change-plastic_work

    figure = OUT/"loading_history.png"
    t = np.array([row["time_s"] for row in records])
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.2), constrained_layout=True)
    axes[0].plot(t, 1e-6*np.array([row["mean_shear_stress_Pa"] for row in records]))
    axes[0].set(xlabel="time (s)", ylabel="mean shear stress (MPa)")
    axes[1].semilogy(t, [row["density_contrast"] for row in records])
    axes[1].set(xlabel="time (s)", ylabel="density contrast")
    axes[2].semilogy(t, [row["family_polarization_rms"] for row in records])
    axes[2].set(xlabel="time (s)", ylabel="polarization RMS")
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    checkpoint = OUT/"final_state.npz"
    np.savez_compressed(
        checkpoint, **state.__dict__, physical_time_s=np.asarray(time_s),
        engineering_shear=np.asarray(initial_shear+shear_rate*time_s))

    scale = max(abs(boundary_work), abs(elastic_change), abs(defect_change),
                abs(heat), 1.0)
    result = {
        "schema": "v66-production-groma-loading-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "reaction-disabled fixed-rate production transport and mechanics",
        "grid": [n, n], "spacing_m": spacing, "temperature_K": 900.0,
        "initial_engineering_shear": initial_shear,
        "engineering_shear_rate_s": shear_rate,
        "seed": 66024, "parameters": {"A": 1.25, "D": .5,
                                        "reduced_modulus_b2_J_m": gb2},
        "records": records,
        "ledger": {
            "completed_time_s": time_s, "termination_reason": termination_reason,
            "maximum_relative_family_content_error": float(np.max(np.abs(
                final_content-initial_content))/np.max(initial_content)),
            "boundary_work_J_m3": boundary_work,
            "recoverable_elastic_change_J_m3": elastic_change,
            "plastic_work_J_m3": plastic_work,
            "defect_free_energy_change_J_m3": defect_change,
            "irreversible_heat_J_m3": heat,
            "first_law_residual_J_m3": first_law,
            "relative_first_law_residual": abs(first_law)/scale,
            "mechanical_power_residual_J_m3": mechanical_residual,
            "relative_mechanical_power_residual": abs(mechanical_residual)/scale,
            "rejected_steps": rejected,
        },
        "decision": {
            "finite_wavelength_nonlinear_amplification": bool(
                records[-1]["density_contrast"] > 2.0*records[0]["density_contrast"]),
            "gnd_bearing_wall_qualified": False,
            "recrystallization_qualified": False,
            "reason": "reaction-disabled loading discriminator cannot qualify a wall or RX",
        },
        "claim_boundary": "no reactions, front motion, grain allocation, or calibration",
        "artifacts": {
            str(figure.relative_to(ROOT)): _sha(figure),
            str(checkpoint.relative_to(ROOT)): _sha(checkpoint),
        },
    }
    path = OUT/"result.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"initial": records[0], "final": records[-1],
                      "ledger": result["ledger"],
                      "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
