#!/usr/bin/env python3
"""Restartable V30 Tier-B1 production Mura/Nye case runner.

This runner advances the accepted V24 mechanical state only.  It never
allocates phase or grain labels.  Physical progress and wall-clock checkpoints
are independent so a scheduler termination preserves every completed window.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from full_model.analysis.run_v24_mechanical_supply import build_case
from full_model.production.common_tensorial_wall import CommonWallDriving
from full_model.production.density_state_map import (
    SIGNED_RESERVOIRS, derived_density_fields,
)
from full_model.production.tensorial_nye import (
    divergence_of_nye, nye_from_plastic_distortion,
)
from full_model.production.v24_mechanical_wall import (
    MuraWorkBudgetError, V24MechanicalWallState, accepted_v24_mechanical_step,
    mechanical_checkpoint_arrays, mechanical_from_checkpoint_arrays,
    synchronize_common,
)
from full_model.production.wall_topology_supply import reservoir_nye_m1


STOP_REQUESTED = False


def _request_stop(signum, frame):
    del signum, frame
    global STOP_REQUESTED
    STOP_REQUESTED = True


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    temporary.replace(path)


def broadband_field(n, seed):
    rng = np.random.default_rng(seed)
    spectrum = np.fft.fftn(rng.normal(size=(n, n)))
    kx = np.fft.fftfreq(n)[:, None]
    ky = np.fft.fftfreq(n)[None, :]
    radius = np.sqrt(kx*kx+ky*ky)
    spectrum *= (radius >= 2/n)&(radius <= .22)
    field = np.real(np.fft.ifftn(spectrum))
    field -= np.mean(field)
    return field/max(float(np.std(field)), 1e-300)


def apply_unsigned_noise(state, topologies, seed, fraction=.01):
    noise = broadband_field(state.common.orientation_rad.shape[0], seed)
    factor = np.maximum(1.0+fraction*noise[..., None], .5)
    density_changes = {}
    alignment_changes = {}
    for name in SIGNED_RESERVOIRS:
        density_changes[name] = np.asarray(getattr(state.density, name))*factor
        alignment_changes[name] = (
            np.asarray(getattr(state.reservoir_alignment, name))
            *factor[..., None])
    density = replace(state.density, **density_changes)
    alignment = replace(state.reservoir_alignment, **alignment_changes)
    noisy = synchronize_common(V24MechanicalWallState(
        state.common, density, alignment), topologies)
    return noisy


def create_case(grid, condition, seed, length_m):
    data = build_case(grid, length_m=length_m,
                      periodic_nye_consistent=True)
    state, _, _, support, systems, topologies, common, extensive, kinetics, dx = data
    fixed = np.zeros((grid, grid, 2, 2))
    active_support = np.zeros((grid, grid), dtype=bool)
    if condition == "broadband_noise":
        state = apply_unsigned_noise(state, topologies, seed)
    elif condition == "mechanical_heterogeneity":
        x = (np.arange(grid)-grid/2)*dx
        y = (np.arange(grid)-grid/2)*dx
        X, Y = np.meshgrid(x, y, indexing="ij")
        particle = np.exp(-((X/(.55e-6))**2+(Y/(.35e-6))**2)**2)
        fixed[..., 0, 1] = .025*particle
        fixed[..., 1, 0] = fixed[..., 0, 1]
        active_support = support
    elif condition != "homogeneous":
        raise ValueError(f"unknown Tier-B1 condition: {condition}")
    state.validate(systems, topologies)
    return (state, fixed, active_support, systems, topologies, common,
            extensive, kinetics, dx)


def latest_checkpoint(case_dir):
    checkpoints = sorted(Path(case_dir).glob("checkpoint_step_*.npz"))
    return checkpoints[-1] if checkpoints else None


def checkpoint_payload(state, metadata):
    payload = mechanical_checkpoint_arrays(state)
    payload["v30_metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    return payload


def write_checkpoint(case_dir, state, metadata, systems, topologies):
    path = Path(case_dir)/(f"checkpoint_step_{metadata['step']:09d}_"
                           f"strain_{metadata['applied_strain']:.8f}.npz")
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **checkpoint_payload(state, metadata))
    temporary.replace(path)
    # Validate the exact representation immediately. Continuous-versus-
    # segmented equality is checked by the test suite and local hard gate.
    with np.load(path, allow_pickle=False) as archive:
        restored = mechanical_from_checkpoint_arrays(
            archive, systems, topologies)
    exact = all(np.array_equal(getattr(getattr(state, group), name),
                               getattr(getattr(restored, group), name))
                for group in ("common", "density", "reservoir_alignment")
                for name in getattr(state, group).__dict__)
    if not exact:
        raise RuntimeError("written mechanical checkpoint failed exact reload")
    return path


def load_checkpoint(path, systems, topologies):
    with np.load(path, allow_pickle=False) as archive:
        state = mechanical_from_checkpoint_arrays(archive, systems, topologies)
        metadata = json.loads(str(archive["v30_metadata_json"].item()))
    return state, metadata


def compact_metrics(state, ledger, systems, topologies, spacing):
    reservoir = reservoir_nye_m1(
        state.reservoir_alignment, systems,
        state.common.orientation_rad, topologies)["total"]
    alpha = np.sum(state.common.family_nye_m1, axis=2)
    curl_beta = nye_from_plastic_distortion(state.common.beta_p, spacing)
    source = alpha-curl_beta
    scale = max(float(np.sqrt(np.mean(alpha*alpha))), 1.0)
    div_scale = max(scale/spacing, 1.0)
    signed = sum(
        np.asarray(getattr(state.density, f"{stem}_plus_m2"))
        -np.asarray(getattr(state.density, f"{stem}_minus_m2"))
        for stem in ("mobile", "forest", "wall_tangle", "wall_ordered"))
    signed_norm = np.linalg.norm(signed, axis=2)
    centered = signed_norm-np.mean(signed_norm)
    power = np.abs(np.fft.fftn(centered))**2
    power[0, 0] = 0.0
    index = np.unravel_index(np.argmax(power), power.shape)
    frequency = np.hypot(np.fft.fftfreq(power.shape[0], d=spacing)[index[0]],
                         np.fft.fftfreq(power.shape[1], d=spacing)[index[1]])
    balance = ledger["mura_balance_ledger"]
    budget = ledger["mura_work_budget"]
    density_fields = derived_density_fields(state.density, topologies)
    wall = density_fields["rho_wall_m2"]
    ordered = density_fields["rho_wall_ordered_m2"]
    wall_threshold = max(float(np.quantile(wall, .9)),
                         .1*float(np.max(wall)), np.finfo(float).tiny)
    wall_local = wall >= wall_threshold
    # ``reservoir`` above is the total tensor for legacy metric compatibility;
    # obtain the component dictionary once for V37 organization diagnostics.
    reservoir_components = reservoir_nye_m1(
        state.reservoir_alignment, systems,
        state.common.orientation_rad, topologies)
    ordered_alpha = reservoir_components["wall_ordered"]
    bmean = float(np.mean([item.burgers_m for item in systems]))
    polarization = np.linalg.norm(ordered_alpha, axis=(-2, -1))/np.maximum(
        bmean*ordered, 1e-300)
    return {
        "dual_nye_relative_rms": float(
            np.sqrt(np.mean((reservoir-alpha)**2))/scale),
        "authoritative_source_offset_relative_rms": float(
            np.sqrt(np.mean(source**2))/scale),
        "normalized_line_continuity_residual": float(
            np.sqrt(np.mean(divergence_of_nye(alpha, spacing)**2))/div_scale),
        "orientation_span_deg": float(
            np.ptp(state.common.orientation_rad)*180/np.pi),
        "maximum_signed_density_m2": float(np.max(signed_norm)),
        "dominant_signed_wavelength_m": (
            None if frequency == 0.0 else float(1.0/frequency)),
        "structure_factor_peak_fraction": float(
            np.max(power)/max(float(np.sum(power)), 1e-300)),
        "ordered_line_m2_cells": float(np.sum(ordered)),
        "total_wall_line_m2_cells": float(np.sum(wall)),
        "ordered_fraction_global": float(
            np.sum(ordered)/max(float(np.sum(wall)), 1e-300)),
        "ordered_fraction_wall_local": float(
            np.sum(ordered[wall_local])/max(
                float(np.sum(wall[wall_local])), 1e-300)),
        "ordered_polarization_maximum": float(np.max(polarization)),
        "ordered_polarization_wall_local_mean": float(
            np.mean(polarization[wall_local]) if np.any(wall_local) else 0.0),
        "junction_line_m2_cells": float(np.sum(state.density.junction_m2)),
        "maximum_scalar_line_balance_residual_m": float(
            balance["maximum_scalar_line_balance_residual_m"]),
        "maximum_alignment_balance_residual_m": float(
            balance["maximum_alignment_balance_residual_m"]),
        "energy_balance_relative": float(abs(
            balance["global_work_minus_heat_storage_residual_J_m3_cells"])
            /max(float(np.sum(np.abs(
                balance["plastic_work_increment_J_m3"]))), 1.0)),
        "minimum_heat_increment_J_m3": float(np.min(
            balance["deposited_heat_increment_J_m3"])),
        "mura_event_scale": float(ledger["mura_event_scale"]),
        "mura_work_budget_trial_count": int(budget["trial_count"]),
        "mura_physical_stall": bool(budget["physical_stall"]),
        "accepted_step_hard_invariant_passed": bool(
            ledger["nye_suboperator_audit"][
                "accepted_step_hard_invariant_passed"]),
        "post_step_projection_used": bool(
            ledger["nye_suboperator_audit"]["post_step_projection_used"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--grid", type=int, required=True)
    parser.add_argument("--condition", choices=(
        "homogeneous", "broadband_noise", "mechanical_heterogeneity"),
        required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--length-m", type=float, default=1e-5)
    parser.add_argument("--temperature-K", type=float, default=1100.0)
    parser.add_argument("--strain-rate-s", type=float, default=1e4)
    parser.add_argument("--target-strain", type=float, default=.2)
    parser.add_argument("--initial-strain", type=float, default=.01)
    parser.add_argument("--trial-dt-s", type=float, default=2e-9)
    parser.add_argument("--progress-checkpoint-strain", type=float, default=.01)
    parser.add_argument("--wall-checkpoint-s", type=float, default=840.0)
    parser.add_argument("--history-interval", type=int, default=25)
    parser.add_argument("--max-steps", type=int, default=2000000)
    parser.add_argument("--max-wall-s", type=float, default=54000.0)
    parser.add_argument("--mura-work-budget-mode", choices=(
        "energy_limited", "energy_limited_feasible_extents",
        "legacy_reject"), default="energy_limited")
    parser.add_argument("--topology-route-enabled", action="store_true",
                        help="use explicit reorientation/junction comparator")
    args = parser.parse_args()
    args.case_dir.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _request_stop)

    (initial, fixed, support, systems, topologies, common, extensive,
     kinetics, spacing) = create_case(
        args.grid, args.condition, args.seed, args.length_m)
    common = replace(common, bath_temperature_K=args.temperature_K)
    initial = replace(initial, common=replace(
        initial.common, temperature_K=np.full((args.grid, args.grid),
                                               args.temperature_K)))
    configuration = {
        "grid": args.grid, "condition": args.condition, "seed": args.seed,
        "length_m": args.length_m, "temperature_K": args.temperature_K,
        "strain_rate_s": args.strain_rate_s,
        "initial_strain": args.initial_strain,
        "target_strain": args.target_strain, "trial_dt_s": args.trial_dt_s,
        "mura_work_budget_mode": args.mura_work_budget_mode,
        "topology_route_enabled": bool(args.topology_route_enabled),
        "source_sha": os.environ.get(
            "V37_SOURCE_SHA", os.environ.get(
            "V35_SOURCE_SHA", os.environ.get(
                "V32_SOURCE_SHA", os.environ.get(
                    "V30_SOURCE_SHA", "UNRECORDED")))),
        "rng_state": {"ongoing_stochastic_evolution": False,
                      "initialization_seed": args.seed},
    }
    atomic_json(args.case_dir/"case_config.json", configuration)
    checkpoint = latest_checkpoint(args.case_dir)
    if checkpoint is None:
        state = initial; step = 0
        physical_time = args.initial_strain/args.strain_rate_s
        cumulative = {"plastic_work_J_m3_cells": 0.0,
                      "heat_J_m3_cells": 0.0,
                      "stored_line_energy_J_m3_cells": 0.0,
                      "capture_line_m2_cells": 0.0,
                      "mura_line_stretching_m2_cells": 0.0,
                      "locking_unlocking_abs_line_m2_cells": 0.0}
        resumed_from = None
    else:
        state, metadata = load_checkpoint(checkpoint, systems, topologies)
        step = int(metadata["step"])
        physical_time = float(metadata["physical_time_s"])
        cumulative = dict(metadata["cumulative_ledger"])
        for key in ("capture_line_m2_cells",
                    "mura_line_stretching_m2_cells",
                    "locking_unlocking_abs_line_m2_cells"):
            cumulative.setdefault(key, 0.0)
        resumed_from = checkpoint.name
    start_wall = time.monotonic(); last_checkpoint_wall = start_wall
    applied_strain = args.strain_rate_s*physical_time
    next_progress = (np.floor(applied_strain/args.progress_checkpoint_strain)+1
                     )*args.progress_checkpoint_strain
    history_path = args.case_dir/"history.jsonl"
    terminal = "RUNNING"
    last_ledger = None
    try:
        target_tolerance = 64*np.finfo(float).eps*max(
            abs(args.target_strain), 1.0)
        while (applied_strain < args.target_strain-target_tolerance
               and step < args.max_steps and not STOP_REQUESTED):
            mean = np.array([[0.0, .5*applied_strain],
                             [.5*applied_strain, 0.0]])
            driving = CommonWallDriving(mean_strain=mean,
                                        fixed_eigenstrain=fixed)
            remaining_time = max(
                (args.target_strain-applied_strain)/args.strain_rate_s, 0.0)
            requested_dt = min(args.trial_dt_s, remaining_time)
            state, ledger = accepted_v24_mechanical_step(
                state, driving, support, systems, topologies, common,
                extensive, kinetics, requested_dt,
                topology_route_enabled=args.topology_route_enabled,
                mura_work_budget_mode=args.mura_work_budget_mode)
            last_ledger = ledger
            physical_time += ledger["accepted_dt_s"]
            applied_strain = args.strain_rate_s*physical_time
            if abs(applied_strain-args.target_strain) <= target_tolerance:
                applied_strain = float(args.target_strain)
                physical_time = applied_strain/args.strain_rate_s
            step += 1
            balance = ledger["mura_balance_ledger"]
            cumulative["plastic_work_J_m3_cells"] += float(np.sum(
                balance["plastic_work_increment_J_m3"]))
            cumulative["heat_J_m3_cells"] += float(np.sum(
                balance["deposited_heat_increment_J_m3"]))
            cumulative["stored_line_energy_J_m3_cells"] += float(np.sum(
                balance["stored_line_energy_increment_J_m3"]))
            for sign in ("plus", "minus"):
                cumulative["capture_line_m2_cells"] += float(np.sum(
                    ledger["transport_capture"]["sign"][sign][
                        "captured_line_m2"]))
                cumulative["mura_line_stretching_m2_cells"] += float(np.sum(
                    ledger["transport_capture"]["sign"][sign][
                        "mura_line_stretching_m2"]))
                cumulative["locking_unlocking_abs_line_m2_cells"] += float(
                    np.sum(np.abs(ledger["locking_unlocking"]["sign"][sign][
                        "accepted_line_m2"])))
            now = time.monotonic()
            if step % args.history_interval == 0:
                record = {"step": step, "physical_time_s": physical_time,
                          "applied_strain": applied_strain,
                          **compact_metrics(state, ledger, systems, topologies,
                                            spacing)}
                with history_path.open("a") as stream:
                    stream.write(json.dumps(record, sort_keys=True)+"\n")
            progress_due = applied_strain+1e-15 >= next_progress
            wall_due = now-last_checkpoint_wall >= args.wall_checkpoint_s
            deadline_due = now-start_wall >= args.max_wall_s
            if progress_due or wall_due or deadline_due:
                serial = {
                    **configuration, "step": step,
                    "physical_time_s": physical_time,
                    "applied_strain": applied_strain,
                    "cumulative_ledger": cumulative,
                    "systems": [item.name for item in systems],
                    "topologies": [item.character for item in topologies],
                }
                path = write_checkpoint(
                    args.case_dir, state, serial, systems, topologies)
                last_checkpoint_wall = now
                while next_progress <= applied_strain+1e-15:
                    next_progress += args.progress_checkpoint_strain
                if deadline_due:
                    terminal = "PARTIAL_WALLCLOCK_LIMIT"
                    break
        if applied_strain >= args.target_strain:
            terminal = "COMPLETED"
        elif STOP_REQUESTED:
            terminal = "PARTIAL_SIGNALLED"
        elif step >= args.max_steps:
            terminal = "PARTIAL_STEP_LIMIT"
    except Exception as error:
        terminal = "FAILED_SCIENTIFIC_OR_APPLICATION"
        failure = {
            "type": type(error).__name__, "message": str(error),
            "step": step, "physical_time_s": physical_time,
            "applied_strain": applied_strain}
        if isinstance(error, MuraWorkBudgetError):
            failure["mura_work_budget_audit"] = error.audit
        atomic_json(args.case_dir/"failure.json", failure)
        raise
    finally:
        if last_ledger is not None:
            metadata = {**configuration, "step": step,
                        "physical_time_s": physical_time,
                        "applied_strain": applied_strain,
                        "cumulative_ledger": cumulative,
                        "systems": systems, "topologies": topologies}
            # Final checkpoint uses a direct serializable validation path.
            serial = dict(metadata)
            serial["systems"] = [item.name for item in systems]
            serial["topologies"] = [item.character for item in topologies]
            path = Path(args.case_dir)/(f"checkpoint_step_{step:09d}_"
                                        f"strain_{applied_strain:.8f}.npz")
            temporary = path.with_suffix(".npz.tmp")
            payload = mechanical_checkpoint_arrays(state)
            payload["v30_metadata_json"] = np.asarray(
                json.dumps(serial, sort_keys=True))
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **payload)
            temporary.replace(path)
        status = {
            **configuration, "status": terminal, "step": step,
            "physical_time_s": physical_time,
            "applied_strain": applied_strain,
            "resumed_from": resumed_from,
            "wall_seconds_this_invocation": time.monotonic()-start_wall,
            "latest_checkpoint": (latest_checkpoint(args.case_dir).name
                                  if latest_checkpoint(args.case_dir) else None),
            "cumulative_ledger": cumulative,
        }
        if last_ledger is not None:
            status["latest_metrics"] = compact_metrics(
                state, last_ledger, systems, topologies, spacing)
        atomic_json(args.case_dir/"status.json", status)
    print(json.dumps(status, sort_keys=True))


if __name__ == "__main__":
    main()
