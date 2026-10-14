#!/usr/bin/env python3
"""Exercise a reduced-network HAGB handoff in the full multigrain front path."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from full_model.analysis.run_v58_three_grain_production import (
    _owner,
    initialize_state,
    network_interfaces,
)
from full_model.production.arrhenius_kinetics import ActivatedProcess, EV_J
from full_model.production.common_tensorial_wall import CommonWallParameters
from full_model.production.complete_multigrain_energy import (
    evaluate_complete_multigrain_energy,
)
from full_model.production.multigrain_common_state import (
    MultiGrainCommonState,
    PhysicalTransferLaw,
    multigrain_checkpoint_arrays,
    multigrain_checkpoint_metadata,
    multigrain_from_checkpoint,
    multigrain_state_digest,
    reconstruct_multigrain_common,
)
from full_model.production.multigrain_production import (
    MultiGrainFrontKinetics,
    MultiGrainProductionRuntime,
    advance_multigrain_front_interval,
)
from full_model.production.tensorial_nye import bcc_four_family_systems


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"full_model"/"verification"/"v66_multigrain_handoff"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _initial(n, length_m, temperature, contrast_Pa):
    spacing = length_m/n
    template = initialize_state(
        n, temperature, spacing_m=spacing, interface_width_m=2*spacing)
    line_energy = 1.5e-9
    low = 1e13
    high = low+contrast_Pa/(6.0*line_energy)
    owners = (
        _owner(n, high, np.radians(0.0), temperature),
        _owner(n, low, np.radians(15.0), temperature),
        _owner(n, high, np.radians(30.0), temperature),
    )
    state = MultiGrainCommonState((10, 20, 30), template.supports, owners)
    state.validate()
    return state, high, low


def _advance(state, runtime, steps, dt, *, spacing, thickness, wall,
             kinetics, energy, systems):
    history = []
    decisions = []
    initial_volume = np.sum(state.supports, axis=(1, 2))*spacing**2*thickness
    for step in range(steps):
        state, runtime, interval = advance_multigrain_front_interval(
            state, runtime, kinetics=kinetics, dt_s=dt,
            maximum_substep_s=dt, spacing_m=spacing,
            represented_thickness_m=thickness, wall_parameters=wall,
            energy_kwargs=energy, applied_shear_rate_s=0.0,
            systems=systems)
        if not all(item.accepted for item in interval):
            raise RuntimeError("full multigrain handoff rejected an interval")
        decisions.extend(interval)
        volume = np.sum(state.supports, axis=(1, 2))*spacing**2*thickness
        history.append({
            "time_s": runtime.ledger.physical_time_s,
            "target_volume_change_fraction": float(
                (volume[1]-initial_volume[1])/initial_volume[1]),
            "volume_change_fraction_by_owner": (
                (volume-initial_volume)/initial_volume).tolist(),
            "accepted_events": runtime.ledger.accepted_events,
            "generated_heat_J": runtime.ledger.generated_heat_J,
            "maximum_relative_energy_closure": (
                runtime.ledger.maximum_relative_energy_closure),
            "directions": interval[-1].direction_by_interface,
        })
    return state, runtime, decisions, history


def _plot(initial, final, spacing, wall, path):
    initial_common, _ = reconstruct_multigrain_common(initial, spacing)
    final_common, _ = reconstruct_multigrain_common(final, spacing)
    def density(common):
        return np.sum(
            common.mobile_plus_m2+common.mobile_minus_m2
            +common.forest_plus_m2+common.forest_minus_m2
            +common.wall_plus_m2+common.wall_minus_m2, axis=2)
    panels = (
        (initial.supports[1], "initial target support", "viridis"),
        (final.supports[1], "final target support", "viridis"),
        (density(initial_common)/1e16,
         r"initial line density ($10^{16}$ m$^{-2}$)", "magma"),
        (density(final_common)/1e16,
         r"final line density ($10^{16}$ m$^{-2}$)", "magma"),
    )
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.0), constrained_layout=True)
    for ax, (value, label, cmap) in zip(axes, panels):
        image = ax.imshow(value.T, origin="lower", cmap=cmap,
                          interpolation="nearest")
        ax.set_xlabel("x cell"); ax.set_ylabel("y cell")
        cbar = fig.colorbar(image, ax=ax, shrink=.8); cbar.set_label(label)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    network_result = json.loads((
        ROOT/"full_model"/"verification"/"v66_subgrain_maturation"/"result.json"
    ).read_text())
    source_case = network_result["cases"]["T1000_rate1"]
    contrast = source_case["stored_energy_contrast_Pa"]
    n = 16
    length_m = 4.8e-6
    spacing = length_m/n
    thickness = 2*2.48e-10
    temperature = 1000.0
    dt = 2e-8
    steps = 20
    systems = bcc_four_family_systems()
    wall = CommonWallParameters(spacing_m=spacing)
    state0, high_density, low_density = _initial(
        n, length_m, temperature, contrast)
    runtime0 = MultiGrainProductionRuntime(network_interfaces(state0))
    kinetics = MultiGrainFrontKinetics(
        ActivatedProcess("V66 matured HAGB", 1e10, 0.0, 1e10),
        PhysicalTransferLaw(.55, .08, .02), .35*EV_J, 1e9,
        2.0, 1.5, .10, wall.burgers_m**3, wall.burgers_m,
        virtual_fraction=2e-4, maximum_fraction_per_step=.01,
        closure_fraction=.05, maximum_backtracks=14,
        temperature_resolution="global_mean")
    energy = {
        "mean_strain": np.zeros((2, 2)),
        "topologies": (), "systems": systems,
        "phase_barrier_J_m3": 5e6, "phase_gradient_J_m": 5e-7,
        "boundary_line_energy_J_m": wall.line_energy_J_m,
        "boundary_junction_energy_J_m": wall.junction_energy_J_m,
        "reference_temperature_K": temperature,
    }
    energy0 = evaluate_complete_multigrain_energy(
        state0, spacing_m=spacing, represented_thickness_m=thickness,
        wall_parameters=wall, **energy)

    # Continuous reference.
    continuous, runtime_continuous, decisions, history = _advance(
        state0, runtime0, steps, dt, spacing=spacing, thickness=thickness,
        wall=wall, kinetics=kinetics, energy=energy, systems=systems)

    # Exact checkpoint/restart fork at half horizon.
    half, runtime_half, _, _ = _advance(
        state0, runtime0, steps//2, dt, spacing=spacing,
        thickness=thickness, wall=wall, kinetics=kinetics,
        energy=energy, systems=systems)
    arrays = multigrain_checkpoint_arrays(half)
    restored = multigrain_from_checkpoint(
        multigrain_checkpoint_metadata(half), arrays)
    restarted, runtime_restarted, _, _ = _advance(
        restored, runtime_half, steps-steps//2, dt, spacing=spacing,
        thickness=thickness, wall=wall, kinetics=kinetics,
        energy=energy, systems=systems)
    restart_equal = multigrain_state_digest(restarted) == multigrain_state_digest(
        continuous)
    if not restart_equal:
        raise RuntimeError("multigrain handoff failed exact restart agreement")

    energy_final = evaluate_complete_multigrain_energy(
        continuous, spacing_m=spacing, represented_thickness_m=thickness,
        wall_parameters=wall, **energy)
    initial_volume = np.sum(state0.supports, axis=(1, 2))*spacing**2*thickness
    final_volume = np.sum(continuous.supports, axis=(1, 2))*spacing**2*thickness
    target_growth = float((final_volume[1]-initial_volume[1])/initial_volume[1])
    target_history = np.array(
        [item["target_volume_change_fraction"] for item in history])
    incident_directions = [
        item.direction_by_interface for item in decisions]
    figure = OUT/"multigrain_handoff_fields.png"
    _plot(state0, continuous, spacing, wall, figure)

    result = {
        "schema": "v66-full-multigrain-handoff-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_reduced_case": "T1000_rate1",
        "handoff_mapping": {
            "orientations_deg": [0.0, 15.0, 30.0],
            "target_owner": 20,
            "stored_energy_contrast_Pa": contrast,
            "low_reservoir_density_m2_each": low_density,
            "high_reservoir_density_m2_each": high_density,
            "support_geometry": "resolved three-owner smooth Voronoi",
            "new_owner_allocated": False,
        },
        "numerics": {"grid": [n, n], "length_m": length_m,
                     "dt_s": dt, "steps": steps,
                     "duration_s": dt*steps},
        "result": {
            "all_intervals_accepted": bool(all(item.accepted for item in decisions)),
            "accepted_front_events": runtime_continuous.ledger.accepted_events,
            "target_growth_fraction": target_growth,
            "target_growth_monotone": bool(np.all(np.diff(target_history) > 0.0)),
            "initial_target_volume_m3": float(initial_volume[1]),
            "final_target_volume_m3": float(final_volume[1]),
            "complete_helmholtz_change_J": (
                energy_final.helmholtz_J-energy0.helmholtz_J),
            "generated_heat_J": runtime_continuous.ledger.generated_heat_J,
            "maximum_relative_energy_closure": (
                runtime_continuous.ledger.maximum_relative_energy_closure),
            "continuous_restart_exact": restart_equal,
            "initial_state_digest": multigrain_state_digest(state0),
            "final_state_digest": multigrain_state_digest(continuous),
            "restarted_state_digest": multigrain_state_digest(restarted),
            "terminal_incident_directions": incident_directions[-1],
        },
        "decision": {
            "full_multigrain_handoff_exercised": True,
            "target_consumes_both_high_defect_neighbors": bool(
                incident_directions[-1].get("edge-10-20") == "a_to_b"
                and incident_directions[-1].get("edge-20-30") == "b_to_a"),
            "sustained_over_tested_horizon": bool(
                target_growth > 0.0 and np.all(np.diff(target_history) > 0.0)),
            "G02_qualified": False,
            "G06_qualified": False,
            "reason": (
                "the mature inherited orientation survives an exact restart and grows "
                "monotonically through the full complete-energy owner/front transaction, "
                "but the 0.4 microsecond converted handoff is not yet a retained-history "
                "long-horizon static RX qualification"),
        },
        "artifacts": {str(figure.relative_to(ROOT)): _sha(figure)},
    }
    path = OUT/"result.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps(result["decision"], indent=2))
    print(json.dumps(result["result"], indent=2))


if __name__ == "__main__":
    main()
