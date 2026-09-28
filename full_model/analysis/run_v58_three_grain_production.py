#!/usr/bin/env python3
"""Run the recurrent V58 three-grain common-owner production trajectory."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from full_model.production.arrhenius_kinetics import ActivatedProcess, EV_J
from full_model.production.common_tensorial_wall import (
    CommonWallDriving, CommonWallParameters, CommonWallState,
)
from full_model.production.complete_multigrain_energy import (
    evaluate_complete_multigrain_energy,
)
from full_model.production.multigrain_common_state import (
    MultiGrainCommonState, PhysicalTransferLaw, audit_multigrain_nye,
    multigrain_checkpoint_arrays, multigrain_checkpoint_metadata,
    multigrain_from_checkpoint,
)
from full_model.production.multigrain_production import (
    MultiGrainFrontKinetics, MultiGrainInterface,
    MultiGrainProductionLedger, MultiGrainProductionRuntime,
    advance_multigrain_front, advance_multigrain_mechanics,
)
from full_model.production.tensorial_nye import bcc_four_family_systems


def _owner(n, density, orientation, temperature):
    grid = (n, n); family = grid+(4,)
    value = np.full(family, density)
    return CommonWallState(
        .08*value, .08*value, .18*value, .18*value,
        .12*value, .12*value, np.zeros(grid+(0,)),
        np.full(grid, .25), np.zeros(grid), np.zeros(family),
        np.zeros(grid+(3, 3)), np.zeros(family+(3,)),
        np.zeros(family+(3, 3)), np.full(grid, orientation),
        np.full(grid, temperature))


def _periodic_distance(coordinate, center, n):
    delta = np.abs(coordinate-center)
    return np.minimum(delta, n-delta)


def initialize_state(n, temperature, equal_density=False):
    x, y = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    centers = ((.22*n, .50*n), (.72*n, .27*n), (.72*n, .73*n))
    distance = np.stack([
        _periodic_distance(x, cx, n)**2+_periodic_distance(y, cy, n)**2
        for cx, cy in centers])
    # Smooth Voronoi ownership produces resolved pure cores, three distinct
    # orientations, and two incident arms meeting at a real triple junction.
    width_cells = 2.0
    logits = -distance/(2.0*width_cells**2)
    logits -= np.max(logits, axis=0, keepdims=True)
    supports = np.exp(logits)
    supports /= np.sum(supports, axis=0, keepdims=True)
    densities = ((2.2e14, 2.2e14, 2.2e14) if equal_density
                 else (2.2e14, 5.0e13, 1.8e14))
    owners = tuple(_owner(n, rho, angle, temperature) for rho, angle in zip(
        densities, (0.0, np.deg2rad(18.0), np.deg2rad(-14.0))))
    state = MultiGrainCommonState((10, 20, 30), supports, owners)
    state.validate()
    return state


def _save_checkpoint(path, state, runtime, step, gamma, initial_volume):
    arrays = multigrain_checkpoint_arrays(state)
    arrays["metadata_json"] = np.asarray(multigrain_checkpoint_metadata(state))
    arrays["runtime_json"] = np.asarray(json.dumps({
        "step": int(step), "gamma": float(gamma),
        "ledger": asdict(runtime.ledger),
        "interfaces": [asdict(item) for item in runtime.interfaces],
    }, sort_keys=True))
    arrays["initial_grain_volume_m3"] = np.asarray(initial_volume)
    np.savez_compressed(path, **arrays)


def _load_checkpoint(path):
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files
                  if key not in ("metadata_json", "runtime_json",
                                 "initial_grain_volume_m3")}
        state = multigrain_from_checkpoint(str(data["metadata_json"]), arrays)
        meta = json.loads(str(data["runtime_json"]))
        initial_volume = np.asarray(data["initial_grain_volume_m3"]).copy()
    runtime = MultiGrainProductionRuntime(
        tuple(MultiGrainInterface(**item) for item in meta["interfaces"]),
        MultiGrainProductionLedger(**meta["ledger"]))
    return (state, runtime, int(meta["step"]), float(meta["gamma"]),
            initial_volume)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--dt", type=float, default=2e-10)
    parser.add_argument("--shear-rate", type=float, default=2e4)
    parser.add_argument("--temperature", type=float, default=900.0)
    parser.add_argument("--case", choices=("baseline", "equal_density", "no_front"),
                        default="baseline")
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--resume")
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    spacing = 2e-8; thickness = 5e-10
    systems = bcc_four_family_systems()
    wall = CommonWallParameters(
        spacing_m=spacing, elastic_iterations=2,
        mobile_correlation_diffusivity_m2_s=0.0,
        thermal_diffusivity_m2_s=1e-7, bath_rate_s=0.0)
    kinetics = MultiGrainFrontKinetics(
        ActivatedProcess("V58 multi-grain HAGB", 1e8, 0.0, 1e8),
        PhysicalTransferLaw(.55, .08, .02), .35*EV_J, 1e9,
        2.0, 1.5, .10, wall.burgers_m**3, wall.burgers_m,
        virtual_fraction=2e-4, maximum_fraction_per_step=.015,
        closure_fraction=.02, maximum_backtracks=14)
    interfaces = tuple(MultiGrainInterface(name, a, b) for name, a, b in (
        ("arm-10-20", 10, 20), ("arm-10-30", 10, 30),
        ("arm-20-30", 20, 30)))
    if args.resume:
        state, runtime, start, gamma, initial_volume = _load_checkpoint(args.resume)
    else:
        state = initialize_state(
            args.n, args.temperature, args.case == "equal_density")
        runtime = MultiGrainProductionRuntime(interfaces)
        start = 0; gamma = .012
        initial_volume = np.sum(
            state.supports, axis=(1, 2))*spacing**2*thickness
    history_path = out/"history.json"
    history = (json.loads(history_path.read_text())
               if args.resume and history_path.exists() else [])
    energy_options = dict(
        mean_strain=np.array([[0.0, .5*gamma], [.5*gamma, 0.0]]),
        topologies=(), systems=systems, phase_barrier_J_m3=8e7,
        phase_gradient_J_m=3.2e-8,
        boundary_line_energy_J_m=wall.line_energy_J_m,
        boundary_junction_energy_J_m=wall.junction_energy_J_m,
        reference_temperature_K=args.temperature)
    for step in range(start, args.steps):
        gamma += args.shear_rate*args.dt
        mean_slip = float(np.mean(sum(
            state.supports[index][..., None]*owner.slip
            for index, owner in enumerate(state.owners))))
        shear_stress = np.clip(
            wall.c44_Pa*(gamma-mean_slip), -2.5e9, 2.5e9)
        resolved = np.empty((args.n, args.n, 4))
        resolved[:] = shear_stress*np.asarray((1.0, .82, .71, .63))
        state, mechanical = advance_multigrain_mechanics(
            state, driving=CommonWallDriving(resolved_stress_Pa=resolved),
            systems=systems, topologies=(), wall_parameters=wall,
            dt_s=args.dt, represented_thickness_m=thickness)
        energy_options["mean_strain"] = np.array(
            [[0.0, .5*gamma], [.5*gamma, 0.0]])
        if args.case == "no_front":
            front = None
            runtime = MultiGrainProductionRuntime(
                runtime.interfaces, replace_ledger(runtime.ledger, args.dt,
                                                   args.shear_rate))
        else:
            state, runtime, front = advance_multigrain_front(
                state, runtime, kinetics=kinetics, dt_s=args.dt,
                spacing_m=spacing, represented_thickness_m=thickness,
                wall_parameters=wall, energy_kwargs=energy_options,
                applied_shear_rate_s=args.shear_rate, systems=systems)
        if step == start or (step+1) % max(args.checkpoint_every, 1) == 0:
            energy = evaluate_complete_multigrain_energy(
                state, spacing_m=spacing,
                represented_thickness_m=thickness,
                wall_parameters=wall, **energy_options)
            audit = audit_multigrain_nye(state, spacing)
            volumes = np.sum(state.supports, axis=(1, 2))*spacing**2*thickness
            temperatures = sum(state.supports[index]*owner.temperature_K
                               for index, owner in enumerate(state.owners))
            history.append({
                "step": step+1, "time_s": runtime.ledger.physical_time_s,
                "applied_shear_strain": gamma,
                "shear_stress_Pa": float(shear_stress),
                "grain_volume_m3": volumes.tolist(),
                "grain_volume_change_m3": (volumes-initial_volume).tolist(),
                "temperature_mean_K": float(np.mean(temperatures)),
                "temperature_contrast_K": float(np.max(temperatures)
                                                -np.min(temperatures)),
                "helmholtz_J": energy.helmholtz_J,
                "thermal_internal_J": energy.thermal_internal_J,
                "support_gradient_nye_norm_m1": float(np.linalg.norm(
                    audit.support_gradient_m1)),
                "owner_nye_mismatch_norm_m1": float(np.linalg.norm(
                    audit.owner_reservoir_mismatch_m1)),
                "mechanical": asdict(mechanical),
                "front": None if front is None else {
                    "accepted": front.accepted,
                    "classification": front.classification,
                    "directions": front.direction_by_interface,
                    "pressures_Pa": front.pressure_by_interface_Pa,
                    "backtracks": front.backtracks,
                },
                "runtime": asdict(runtime.ledger),
            })
            checkpoint = out/f"checkpoint_{step+1:06d}.npz"
            _save_checkpoint(
                checkpoint, state, runtime, step+1, gamma, initial_volume)
            (out/"history.json").write_text(json.dumps(history, indent=2))
            print(json.dumps(history[-1], sort_keys=True), flush=True)
    latest = sorted(out.glob("checkpoint_*.npz"))[-1]
    result = {
        "schema": "asb-drx-v58-three-grain-production-v1",
        "case": args.case, "n": args.n, "steps": args.steps,
        "dt_s": args.dt, "temperature_K": args.temperature,
        "shear_rate_s": args.shear_rate,
        "source_commit": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), text=True).strip(),
        "initial_grain_volume_m3": initial_volume.tolist(),
        "final": history[-1], "runtime": asdict(runtime.ledger),
        "checkpoint": str(latest),
        "checkpoint_sha256": hashlib.sha256(latest.read_bytes()).hexdigest(),
    }
    (out/"result.json").write_text(json.dumps(result, indent=2))


def replace_ledger(ledger, dt, shear_rate):
    from dataclasses import replace
    return replace(
        ledger, intervals=ledger.intervals+1,
        stationary_intervals=ledger.stationary_intervals+1,
        physical_time_s=ledger.physical_time_s+dt,
        applied_shear_strain=ledger.applied_shear_strain+shear_rate*dt)


if __name__ == "__main__":
    main()
