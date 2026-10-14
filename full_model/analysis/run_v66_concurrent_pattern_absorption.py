#!/usr/bin/env python3
"""Nonlinear concurrent H3 formation and H1 compatible absorption."""

from __future__ import annotations

from dataclasses import fields, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from full_model.analysis.run_v66_groma_reaction_survival import (
    _energies, _homogenize, _observables,
)
from full_model.analysis.run_v66_interaction_stability import _state
from full_model.analysis.run_v66_polarized_absorption_handoff import _spatial
from full_model.analysis.run_v66_production_groma_nonlinear import _combine, _map
from full_model.analysis.run_v66_production_groma_tangent import candidate_parameters
from full_model.production.common_tensorial_wall import (
    CommonWallDriving, wall_residual,
)
from full_model.production.tensorial_nye import (
    bcc_four_family_systems, make_junction_topology,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = (ROOT/"full_model"/"verification"/
       "v66_concurrent_pattern_absorption_frozen_order")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_rk4(initial, driving, systems, topologies, parameters, maximum_dt, *,
             hold_time, record_interval):
    """Fourth-order same-residual integration with admissibility rejection."""
    state = initial
    e0 = _energies(state, driving, systems, topologies, parameters)
    records = [{"time_s": 0.0, **_observables(state)}]
    time_s = 0.0
    dt = maximum_dt
    next_record = record_interval
    accepted_steps = rejected_steps = 0

    def rate(value):
        return wall_residual(
            value, driving, systems, topologies, parameters).state_rate

    while time_s < hold_time-1e-15:
        requested = min(dt, hold_time-time_s, next_record-time_s)
        try:
            k1 = rate(state)
            k2 = rate(_combine(state, k1, .5*requested))
            k3 = rate(_combine(state, k2, .5*requested))
            k4 = rate(_combine(state, k3, requested))
            increment = _map(k1, lambda name, value: (
                value+2.0*getattr(k2, name)+2.0*getattr(k3, name)
                +getattr(k4, name))/6.0)
            candidate = _combine(state, increment, requested)
            candidate.validate(systems, topologies)
        except (ValueError, FloatingPointError):
            dt *= .5
            rejected_steps += 1
            if dt < 1.0e-10:
                raise RuntimeError("RK4 admissibility step fell below 0.1 ns")
            continue
        state = candidate
        time_s += requested
        accepted_steps += 1
        if dt < maximum_dt:
            dt = min(maximum_dt, 1.25*dt)
        if next_record-time_s < 2e-14:
            time_s = next_record
            records.append({"time_s": time_s, **_observables(state)})
            next_record += record_interval
    e1 = _energies(state, driving, systems, topologies, parameters)
    changes = [right-left for left, right in zip(e0, e1)]
    return state, {
        "integrator": "same_residual_RK4",
        "maximum_dt_s": maximum_dt,
        "accepted_steps": accepted_steps,
        "rejected_steps": rejected_steps,
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
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n = 24
    spacing = 2.0e-7
    temperature = 1500.0
    engineering_shear = 1.6495e-3
    # The prospective full mobile tangent predicts 1.43 e-folds in 2 ms.
    # This prefix is therefore sufficient to discriminate amplification from
    # damping while remaining below the observed forest-depletion stiffness.
    duration = 2.0e-3
    systems = bcc_four_family_systems()
    topologies = (make_junction_topology(
        systems, 0, 1, 1, -1, line_tension_J_m=0.0),)
    parameters, gb2 = candidate_parameters(spacing)
    parameters = replace(
        parameters, polarized_wall_absorption_enabled=True,
        wall_absorption_polarization_power=1.0,
        wall_order_enabled=False)
    base = replace(_state(n), temperature_K=np.full((n, n), temperature))
    base = replace(
        base, mobile_minus_m2=base.mobile_plus_m2.copy(),
        forest_minus_m2=base.forest_plus_m2.copy(),
        wall_minus_m2=base.wall_plus_m2.copy())
    rng = np.random.default_rng(661500)
    arrays = {}
    for name in ("mobile_plus_m2", "mobile_minus_m2"):
        noise = rng.normal(size=getattr(base, name).shape)
        noise -= np.mean(noise, axis=(0, 1), keepdims=True)
        arrays[name] = getattr(base, name)*(1.0+1.0e-4*noise)
    patterned = replace(base, **arrays)
    homogeneous = _homogenize(base)
    driving = CommonWallDriving(mean_strain=np.array(
        [[0.0, .5*engineering_shear], [.5*engineering_shear, 0.0]]))
    def save_branch(name, state, ledger):
        checkpoint = OUT/f"{name}_state.npz"
        np.savez_compressed(checkpoint, **{
            item.name: getattr(state, item.name) for item in fields(state)},
            physical_time_s=np.asarray(duration),
            engineering_shear=np.asarray(engineering_shear))
        (OUT/f"{name}.json").write_text(json.dumps(
            {"schema": "v66-concurrent-pattern-branch-v1", "name": name,
             "checkpoint_sha256": _sha(checkpoint), "ledger": ledger},
            indent=2, sort_keys=True)+"\n")

    coarse_state, coarse = _run_rk4(
        patterned, driving, systems, topologies, parameters, 2.0e-6,
        hold_time=duration, record_interval=4.0e-5)
    save_branch("coarse", coarse_state, coarse)
    fine_state, fine = _run_rk4(
        patterned, driving, systems, topologies, parameters, 1.0e-6,
        hold_time=duration, record_interval=4.0e-5)
    save_branch("fine", fine_state, fine)
    control_state, control = _run_rk4(
        homogeneous, driving, systems, topologies, parameters, 2.0e-6,
        hold_time=duration, record_interval=4.0e-5)
    save_branch("homogeneous_control", control_state, control)
    spatial = {
        "initial": _spatial(patterned, spacing),
        "coarse": _spatial(coarse_state, spacing),
        "fine": _spatial(fine_state, spacing),
        "homogeneous_control": _spatial(control_state, spacing),
    }
    convergence_keys = (
        "wall_density_mean_m2", "wall_density_contrast",
        "wall_vector_polarization_mean", "orientation_gradient_rms_m1")
    convergence = {key: abs(spatial["coarse"][key]-spatial["fine"][key])
                   /max(abs(spatial["fine"][key]), 1.0)
                   for key in convergence_keys}
    initial_observables = _observables(patterned)
    final_observables = _observables(fine_state)

    checkpoint = OUT/"final_state.npz"
    np.savez_compressed(checkpoint, **{
        item.name: getattr(fine_state, item.name) for item in fields(fine_state)},
        physical_time_s=np.asarray(duration),
        engineering_shear=np.asarray(engineering_shear))
    figure = OUT/"concurrent_history.png"
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), constrained_layout=True)
    for label, case in (("patterned", fine), ("homogeneous", control)):
        time = [row["time_s"] for row in case["records"]]
        axes[0].semilogy(time, [row["reservoirs"]["mobile"]["polarization_rms"]
                               for row in case["records"]], label=label)
        axes[1].semilogy(time, [max(row["reservoirs"]["wall"]["polarization_rms"],
                                      1e-16) for row in case["records"]], label=label)
        axes[2].plot(time, [row["wall_order_max"] for row in case["records"]],
                     label=label)
    axes[0].set(xlabel="time (s)", ylabel="mobile polarization RMS")
    axes[1].set(xlabel="time (s)", ylabel="wall polarization RMS")
    axes[2].set(xlabel="time (s)", ylabel="maximum wall order")
    axes[0].legend(frameon=False)
    fig.savefig(figure, dpi=180)
    plt.close(fig)

    energy_ok = fine["energy"]["relative_residual"] < 5.0e-4
    result = {
        "schema": "v66-concurrent-pattern-absorption-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "grid": [n, n], "spacing_m": spacing, "seed": 661500,
        "temperature_K": temperature,
        "fixed_engineering_shear": engineering_shear,
        "hold_time_s": duration,
        "initial_noise_relative_amplitude": 1.0e-4,
        "fixed_parameters": {
            "A": 1.25, "D": .5, "reduced_modulus_b2_J_m": gb2,
            "glide_barrier_eV": parameters.glide_barrier_eV,
            "reaction_barriers_eV": {
                "lock": parameters.lock_barrier_eV,
                "wall": parameters.wall_barrier_eV,
                "annihilation": parameters.annihilation_barrier_eV,
                "junction": parameters.junction_barrier_eV,
                "order": parameters.order_barrier_eV,
            },
            "exp_floor": parameters.exp_floor,
            "taylor_alpha": parameters.taylor_alpha,
            "wall_order_enabled": parameters.wall_order_enabled,
        },
        "initial_observables": initial_observables,
        "final_observables": final_observables,
        "spatial": spatial,
        "cases": {"coarse": coarse, "fine": fine,
                  "homogeneous_control": control},
        "coarse_fine_relative_differences": convergence,
        "decision": {
            "five_percent_timestep_convergence": bool(max(convergence.values()) < .05),
            "energy_gate_passed": energy_ok,
            "nonlinear_mobile_polarization_amplified": bool(
                final_observables["reservoirs"]["mobile"]["polarization_rms"]
                > 2.0*initial_observables["reservoirs"]["mobile"]["polarization_rms"]),
            "compatible_wall_polarization_above_control": bool(
                final_observables["reservoirs"]["wall"]["polarization_rms"]
                > 10.0*control["final"]["reservoirs"]["wall"]["polarization_rms"]),
            "physical_wall_qualified": False,
            "recrystallization_qualified": False,
        },
        "claim_boundary": (
            "wall-order-frozen mechanism ablation of concurrent signed "
            "transport and absorption; cannot qualify a physical wall, grain "
            "allocation, material calibration, or RX"),
        "artifacts": {
            str(figure.relative_to(ROOT)): _sha(figure),
            str(checkpoint.relative_to(ROOT)): _sha(checkpoint),
        },
    }
    (OUT/"result.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"initial": initial_observables,
                      "final": final_observables, "spatial": spatial,
                      "convergence": convergence, "energy": fine["energy"],
                      "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
