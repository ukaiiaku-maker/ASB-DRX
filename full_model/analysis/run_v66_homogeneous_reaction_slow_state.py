#!/usr/bin/env python3
"""Resolve the homogeneous complete-reaction slow state with a stiff solver."""

from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from full_model.analysis.run_v66_groma_reaction_survival import (
    _energies, _homogenize, _load_state, _observables,
)
from full_model.analysis.run_v66_production_groma_tangent import candidate_parameters
from full_model.production.common_tensorial_wall import (
    CommonWallDriving, CommonWallState, wall_residual,
)
from full_model.production.tensorial_nye import (
    bcc_four_family_systems, make_junction_topology,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT/"full_model"/"verification"/"v66_production_groma_loading"
          /"final_state.npz")
OUT = ROOT/"full_model"/"verification"/"v66_homogeneous_reaction_slow_state"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _single_cell(state):
    return CommonWallState(**{
        item.name: np.asarray(getattr(state, item.name))[:1, :1].copy()
        for item in fields(state)})


def _layout(state, parameters):
    result = []
    offset = 0
    positive_names = {
        "mobile_plus_m2", "mobile_minus_m2", "forest_plus_m2",
        "forest_minus_m2", "wall_plus_m2", "wall_minus_m2",
        "junction_m2", "temperature_K"}
    for item in fields(state):
        value = np.asarray(getattr(state, item.name))
        positive = item.name in positive_names
        scale = (parameters.rho_reference_m2 if item.name in positive_names
                 and item.name != "temperature_K" else
                 parameters.bath_temperature_K if item.name == "temperature_K"
                 else 1.0)
        result.append((item.name, value.shape, slice(offset, offset+value.size),
                       positive, scale))
        offset += value.size
    return tuple(result)


def _pack(state, layout):
    arrays = []
    for name, _, _, positive, scale in layout:
        value = np.asarray(getattr(state, name)).reshape(-1)
        arrays.append(np.log(value/scale) if positive else value/scale)
    return np.concatenate(arrays)


def _unpack(vector, layout):
    arrays = {}
    for name, shape, section, positive, scale in layout:
        value = np.asarray(vector[section]).reshape(shape)
        arrays[name] = scale*np.exp(value) if positive else scale*value
    return CommonWallState(**arrays)


def _pack_transformed_rate(rate, state, layout):
    arrays = []
    for name, _, _, positive, scale in layout:
        value = np.asarray(getattr(rate, name)).reshape(-1)
        if positive:
            base = np.asarray(getattr(state, name)).reshape(-1)
            arrays.append(value/base)
        else:
            arrays.append(value/scale)
    return np.concatenate(arrays)


def _integrate(initial, driving, systems, topologies, parameters, rtol):
    layout = _layout(initial, parameters)
    y0 = _pack(initial, layout)
    atol = np.full_like(y0, 1e-11)

    def rhs(_, vector):
        state = _unpack(vector, layout)
        rate = wall_residual(
            state, driving, systems, topologies, parameters,
            differentiation_state=True).state_rate
        return _pack_transformed_rate(rate, state, layout)

    horizon = .1
    sample_times = np.unique(np.concatenate((
        np.asarray([0.0]), np.geomspace(1e-7, horizon, 120))))
    solution = solve_ivp(
        rhs, (0.0, horizon), y0, method="BDF", t_eval=sample_times,
        rtol=rtol, atol=atol, max_step=.02)
    if not solution.success:
        raise RuntimeError(solution.message)
    states = [_unpack(solution.y[:, index], layout)
              for index in range(solution.y.shape[1])]
    for state in states:
        state.validate(systems, topologies)
    return solution, states


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source, shear = _load_state(SOURCE)
    initial = _single_cell(_homogenize(source))
    systems = bcc_four_family_systems()
    topologies = (make_junction_topology(
        systems, 0, 1, 1, -1, line_tension_J_m=0.0),)
    parameters, gb2 = candidate_parameters(2.0e-7)
    driving = CommonWallDriving(mean_strain=np.array(
        [[0.0, .5*shear], [.5*shear, 0.0]]))
    coarse_solution, coarse_states = _integrate(
        initial, driving, systems, topologies, parameters, 2e-7)
    fine_solution, fine_states = _integrate(
        initial, driving, systems, topologies, parameters, 5e-8)
    final = fine_states[-1]
    final.validate(systems, topologies)

    initial_energy = _energies(
        initial, driving, systems, topologies, parameters)
    final_energy = _energies(
        final, driving, systems, topologies, parameters)
    changes = [right-left for left, right in zip(initial_energy, final_energy)]
    final_residual = wall_residual(
        final, driving, systems, topologies, parameters)
    rate_norms = {}
    for item in fields(final):
        value = np.asarray(getattr(final, item.name))
        rate = np.asarray(getattr(final_residual.state_rate, item.name))
        scale = max(float(np.max(np.abs(value))), 1.0)
        rate_norms[item.name] = float(np.max(np.abs(rate))/scale)

    comparison_layout = _layout(final, parameters)
    coarse = _pack(coarse_states[-1], comparison_layout)
    fine = _pack(final, comparison_layout)
    # Positive variables are logarithmic, so this is a relative state error;
    # signed variables use their declared dimensionless scales.
    convergence = float(np.max(np.abs(coarse-fine)))
    checkpoint = OUT/"slow_state.npz"
    np.savez_compressed(
        checkpoint, **final.__dict__, physical_time_s=np.asarray(.1),
        engineering_shear=np.asarray(shear))

    figure = OUT/"slow_state_history.png"
    time = fine_solution.t
    observed = [_observables(state) for state in fine_states]
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.2), constrained_layout=True)
    axes[0].semilogx(time[1:], [row["reservoirs"]["mobile"]["mean_density_m2"]
                               for row in observed[1:]])
    axes[0].set(xlabel="hold time (s)", ylabel="mobile density (1/m2)")
    axes[1].semilogx(time[1:], [row["reservoirs"]["forest"]["mean_density_m2"]
                               for row in observed[1:]])
    axes[1].set(xlabel="hold time (s)", ylabel="forest density (1/m2)")
    axes[2].semilogx(time[1:], [row["temperature_mean_K"] for row in observed[1:]])
    axes[2].set(xlabel="hold time (s)", ylabel="temperature (K)")
    fig.savefig(figure, dpi=180)
    plt.close(fig)

    energy_scale = max(*(abs(item) for item in changes), 1.0)
    result = {
        "schema": "v66-homogeneous-reaction-slow-state-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(SOURCE.relative_to(ROOT)),
        "source_sha256": _sha(SOURCE),
        "fixed_engineering_shear": shear,
        "parameters": {"A": 1.25, "D": .5,
                       "reduced_modulus_b2_J_m": gb2},
        "integration": {
            "method": "scipy_BDF_analysis_only",
            "coarse_rtol": 2e-7, "fine_rtol": 5e-8,
            "coarse_nfev": coarse_solution.nfev,
            "fine_nfev": fine_solution.nfev,
            "maximum_component_relative_difference": convergence,
        },
        "initial": _observables(initial),
        "final": _observables(final),
        "final_relative_rate_norms_s": rate_norms,
        "energy": {
            "elastic_change_J_m3": changes[0],
            "defect_change_J_m3": changes[1],
            "thermal_change_J_m3": changes[2],
            "fixed_strain_adiabatic_residual_J_m3": sum(changes),
            "relative_residual": abs(sum(changes))/energy_scale,
        },
        "decision": {
            "five_percent_solver_convergence": convergence < .05,
            "energy_gate_passed": abs(sum(changes))/energy_scale < .05,
            "slow_state_reached": max(rate_norms.values()) < 1e-3,
            "production_integrator_qualified": False,
        },
        "claim_boundary": (
            "analysis-only homogeneous stiff integration; not a spatial "
            "production trajectory or material calibration"),
        "artifacts": {
            str(checkpoint.relative_to(ROOT)): _sha(checkpoint),
            str(figure.relative_to(ROOT)): _sha(figure),
        },
    }
    path = OUT/"result.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"integration": result["integration"],
                      "final": result["final"],
                      "largest_rate_norms": dict(sorted(
                          rate_norms.items(), key=lambda item: item[1],
                          reverse=True)[:8]),
                      "energy": result["energy"],
                      "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
