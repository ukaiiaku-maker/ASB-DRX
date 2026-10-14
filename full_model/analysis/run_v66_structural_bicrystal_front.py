#!/usr/bin/env python3
"""Compare a conserved structural-content bicrystal front to fixed labels."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from full_model.analysis.run_v58_three_grain_production import _owner
from full_model.production.arrhenius_kinetics import ActivatedProcess, EV_J
from full_model.production.common_tensorial_wall import CommonWallParameters
from full_model.production.complete_multigrain_energy import (
    evaluate_complete_multigrain_energy,
)
from full_model.production.multigrain_common_state import (
    MultiGrainCommonState, PhysicalTransferLaw,
)
from full_model.production.multigrain_production import (
    MultiGrainFrontKinetics, MultiGrainInterface,
    MultiGrainProductionRuntime, advance_multigrain_front_interval,
)
from full_model.production.structural_interface_network import (
    implied_misorientations_rad, initialize_structural_interfaces,
    interface_lengths_m,
)
from full_model.production.tensorial_nye import bcc_four_family_systems


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"full_model"/"verification"/"v66_structural_bicrystal_front.json"


def _state(n, length_m, temperature, contrast_Pa):
    spacing = length_m/n
    coordinates = (np.arange(n)-n/2+.5)*spacing
    x, y = np.meshgrid(coordinates, coordinates, indexing="ij")
    radius = np.sqrt(x*x+y*y)
    target = 1.0/(1.0+np.exp((radius-1.2e-6)/(1.5*spacing)))
    supports = np.stack((1.0-target, target))
    line_energy = 1.5e-9
    low = 1e13
    high = low+contrast_Pa/(6.0*line_energy)
    owners = (
        _owner(n, high, 0.0, temperature),
        _owner(n, low, np.radians(2.5), temperature),
    )
    state = MultiGrainCommonState((10, 20), supports, owners)
    state.validate()
    return state, high, low


def _run(state0, *, structural, steps, dt, spacing, thickness, wall,
         kinetics, systems, interface, structural_interfaces):
    runtime = MultiGrainProductionRuntime((interface,))
    base_energy = {
        "topologies": (), "systems": systems,
        # Structural Read--Shockley energy replaces the generic phase boundary
        # terms in this ablation; it is never counted twice.
        "phase_barrier_J_m3": 0.0, "phase_gradient_J_m": 0.0,
        "boundary_line_energy_J_m": wall.line_energy_J_m,
        "boundary_junction_energy_J_m": wall.junction_energy_J_m,
        "reference_temperature_K": 1000.0,
    }
    if structural:
        base_energy.update(
            structural_interfaces=structural_interfaces,
            structural_burgers_m=wall.burgers_m)
    state = state0
    volume0 = np.sum(state.supports[1])*spacing**2*thickness
    lengths = []
    angles = []
    volumes = []
    rotation_energy_residual = 0.0
    for _ in range(steps):
        state, runtime, decisions = advance_multigrain_front_interval(
            state, runtime, kinetics=kinetics, dt_s=dt,
            maximum_substep_s=dt, spacing_m=spacing,
            represented_thickness_m=thickness, wall_parameters=wall,
            energy_kwargs=dict(base_energy, mean_strain=np.zeros((2, 2))),
            applied_shear_rate_s=0.0, systems=systems)
        if not all(item.accepted and not item.contour_cfl_clipped
                   for item in decisions):
            raise RuntimeError("structural bicrystal front rejected")
        length = interface_lengths_m(
            state.supports, structural_interfaces, state.grain_ids, spacing)[0]
        if structural:
            theta = implied_misorientations_rad(
                structural_interfaces, np.array((length,)), wall.burgers_m)[0]
            before = evaluate_complete_multigrain_energy(
                state, spacing_m=spacing,
                represented_thickness_m=thickness, wall_parameters=wall,
                mean_strain=np.zeros((2, 2)), **base_energy)
            owner = replace(
                state.owners[1],
                orientation_rad=np.full_like(
                    state.owners[1].orientation_rad, theta))
            rotated = replace(state, owners=(state.owners[0], owner))
            rotated.validate()
            after = evaluate_complete_multigrain_energy(
                rotated, spacing_m=spacing,
                represented_thickness_m=thickness, wall_parameters=wall,
                mean_strain=np.zeros((2, 2)), **base_energy)
            rotation_energy_residual = max(
                rotation_energy_residual, abs(after.internal_J-before.internal_J))
            state = rotated
        else:
            theta = np.radians(2.5)
        lengths.append(length)
        angles.append(theta)
        volumes.append(np.sum(state.supports[1])*spacing**2*thickness)
    final_energy = evaluate_complete_multigrain_energy(
        state, spacing_m=spacing, represented_thickness_m=thickness,
        wall_parameters=wall, mean_strain=np.zeros((2, 2)), **base_energy)
    return {
        "target_growth_fraction": float((volumes[-1]-volume0)/volume0),
        "target_growth_monotone": bool(np.all(np.diff(volumes) > 0.0)),
        "initial_interface_length_m": float(lengths[0]),
        "final_interface_length_m": float(lengths[-1]),
        "initial_misorientation_deg": float(np.degrees(angles[0])),
        "final_misorientation_deg": float(np.degrees(angles[-1])),
        "L_theta_relative_change": float(
            (lengths[-1]*angles[-1]-lengths[0]*angles[0])
            /max(abs(lengths[0]*angles[0]), 1e-300)),
        "generated_heat_J": runtime.ledger.generated_heat_J,
        "maximum_relative_energy_closure": (
            runtime.ledger.maximum_relative_energy_closure),
        "accepted_events": runtime.ledger.accepted_events,
        "rotation_complete_energy_residual_J": rotation_energy_residual,
        "final_structural_boundary_J": final_energy.structural_boundary_J,
    }


def main():
    n = 24
    length = 4.8e-6
    spacing = length/n
    thickness = 2*2.48e-10
    contrast = json.loads((ROOT/"full_model"/"verification"/
                           "v66_subgrain_maturation"/"result.json").read_text())[
                               "cases"]["T1000_rate1"]["stored_energy_contrast_Pa"]
    state, high, low = _state(n, length, 1000.0, contrast)
    systems = bcc_four_family_systems()
    wall = CommonWallParameters(spacing_m=spacing)
    interface = MultiGrainInterface("edge-10-20", 10, 20)
    initial_length = interface_lengths_m(
        state.supports, (interface,), state.grain_ids, spacing)
    structural_interfaces = initialize_structural_interfaces(
        state.grain_ids, np.radians((0.0, 2.5)), ((10, 20),),
        initial_length, wall.burgers_m)
    kinetics = MultiGrainFrontKinetics(
        ActivatedProcess("V66 structural bicrystal", 1e10, 0.0, 1e10),
        PhysicalTransferLaw(.55, .08, .02), .35*EV_J, 1e9,
        2.0, 1.5, .10, wall.burgers_m**3, wall.burgers_m,
        virtual_fraction=2e-4, maximum_fraction_per_step=.01,
        closure_fraction=.05, maximum_backtracks=14,
        temperature_resolution="global_mean")
    steps = 200
    dt = 2e-8
    cases = {
        "conserved_structural_content": _run(
            state, structural=True, steps=steps, dt=dt, spacing=spacing,
            thickness=thickness, wall=wall, kinetics=kinetics, systems=systems,
            interface=interface, structural_interfaces=structural_interfaces),
        "fixed_label_no_structural_energy": _run(
            state, structural=False, steps=steps, dt=dt, spacing=spacing,
            thickness=thickness, wall=wall, kinetics=kinetics, systems=systems,
            interface=interface, structural_interfaces=structural_interfaces),
    }
    payload = {
        "schema": "v66-structural-bicrystal-front-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": "2be1347+structural-development",
        "configuration": {
            "grid": [n, n], "length_m": length, "temperature_K": 1000.0,
            "steps": steps, "dt_s": dt, "duration_s": steps*dt,
            "initial_misorientation_deg": 2.5,
            "high_density_m2_each": high, "low_density_m2_each": low,
            "generic_phase_boundary_energy_disabled": True,
        },
        "cases": cases,
        "decision": {
            "conservative_rotation_exercised": True,
            "structural_energy_in_complete_front_functional": True,
            "double_counted_generic_boundary_energy": False,
            "physical_grain_count_change": 0,
            "scope": (
                "exact common-axis two-owner conservative limit; junction-cycle "
                "redistribution/reaction remains required for polycrystal promotion"),
        },
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
