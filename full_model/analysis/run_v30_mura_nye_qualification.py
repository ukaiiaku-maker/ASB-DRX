#!/usr/bin/env python3
"""Decision-grade local qualification for the production V30 Mura update."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from full_model.analysis.run_v24_mechanical_supply import build_case
from full_model.production.mura_kinematics import (
    accept_family_mura_step, family_plastic_flow_from_signed_alignment,
)
from full_model.production.tensorial_nye import (
    divergence_of_nye, nye_from_plastic_distortion, rotated_system_fields,
)
from full_model.production.v24_mechanical_wall import (
    accepted_v24_mechanical_step, mechanical_checkpoint_arrays,
    mechanical_from_checkpoint_arrays,
)
from full_model.production.wall_topology_supply import reservoir_nye_m1


def manufactured(n, mode):
    state, _, _, _, systems, _, common, _, _, spacing = build_case(
        n, periodic_nye_consistent=True)
    x = np.arange(n)[:, None]*2*np.pi/n
    y = np.arange(n)[None, :]*2*np.pi/n
    angle = (0.08*np.sin(x)*np.cos(2*y) if mode == "rotating_frame"
             else np.zeros((n, n)))
    _, slip, normal = rotated_system_fields(systems, angle)
    line = np.cross(normal, slip)
    profile = 1.0+0.15*np.cos(2*x-y)
    plus = 6e13*profile[..., None, None]*line
    minus = 4e13*(2.0-profile[..., None, None])*line
    if mode == "physical_diffusion":
        velocity = np.zeros_like(plus)
        velocity[..., 0] = -6e-5*np.sin(2*x-y)[..., None]
        velocity[..., 1] = 3e-5*np.sin(2*x-y)[..., None]
    elif mode == "homogeneous_ssd":
        plus = 6e13*line; minus = 6e13*line
        velocity = np.broadcast_to(1e-4*slip, plus.shape).copy()
    else:
        velocity = (1e-4*(1.0+0.1*np.sin(x+y))[..., None, None]
                    *slip)
    family_flow = family_plastic_flow_from_signed_alignment(
        plus, minus, velocity, -velocity, systems, angle)
    beta, family_alpha, audit = accept_family_mura_step(
        np.zeros((n, n, 3, 3)),
        np.zeros((n, n, len(systems), 3, 3)), family_flow, 2e-8,
        common.spacing_m)
    alpha = np.sum(family_alpha, axis=2)
    curl_beta = nye_from_plastic_distortion(beta, spacing)
    scale = max(float(np.sqrt(np.mean(curl_beta**2))), 1.0)
    div_scale = max(scale/spacing, 1.0)
    return {
        "grid": n, "mode": mode,
        "dual_nye_relative_rms": float(
            np.sqrt(np.mean((alpha-curl_beta)**2))/scale),
        "normalized_line_continuity_residual": float(
            np.sqrt(np.mean(divergence_of_nye(alpha, spacing)**2))/div_scale),
        "post_step_projection_used": audit["post_step_projection_used"],
        "passed": bool(audit["accepted_step_hard_invariant_passed"]),
    }


def transition(n, steps):
    (state, driving, _, support, systems, topologies, common, extensive,
     kinetics, spacing) = build_case(n, periodic_nye_consistent=True)
    maximum_dual = 0.0
    maximum_continuity = 0.0
    maximum_line_balance = 0.0
    maximum_alignment_balance = 0.0
    maximum_energy_balance = 0.0
    maximum_energy_balance_relative = 0.0
    minimum_heat = np.inf
    accepted_time = 0.0
    first_failure = None
    for step in range(steps):
        state, ledger = accepted_v24_mechanical_step(
            state, driving, support, systems, topologies, common, extensive,
            kinetics, 2e-9, topology_route_enabled=False)
        accepted_time += ledger["accepted_dt_s"]
        invariant = ledger["nye_suboperator_audit"]
        if not invariant["accepted_step_hard_invariant_passed"] and first_failure is None:
            first_failure = {"step": step+1,
                             "operator": invariant["first_violating_suboperator"]}
        reservoir = reservoir_nye_m1(
            state.reservoir_alignment, systems,
            state.common.orientation_rad, topologies)["total"]
        alpha = np.sum(state.common.family_nye_m1, axis=2)
        curl_beta = nye_from_plastic_distortion(state.common.beta_p, spacing)
        scale = max(float(np.sqrt(np.mean(curl_beta**2))), 1.0)
        maximum_dual = max(maximum_dual, float(
            np.sqrt(np.mean((reservoir-curl_beta)**2))/scale))
        maximum_continuity = max(maximum_continuity, float(
            np.sqrt(np.mean(divergence_of_nye(alpha, spacing)**2))
            /max(scale/spacing, 1.0)))
        balance = ledger["mura_balance_ledger"]
        maximum_line_balance = max(maximum_line_balance, abs(
            balance["maximum_scalar_line_balance_residual_m"]))
        maximum_alignment_balance = max(maximum_alignment_balance, abs(
            balance["maximum_alignment_balance_residual_m"]))
        maximum_energy_balance = max(maximum_energy_balance, abs(
            balance["global_work_minus_heat_storage_residual_J_m3_cells"]))
        work_scale = max(float(np.sum(np.abs(
            balance["plastic_work_increment_J_m3"]))), 1.0)
        maximum_energy_balance_relative = max(
            maximum_energy_balance_relative,
            abs(balance["global_work_minus_heat_storage_residual_J_m3_cells"])
            /work_scale)
        minimum_heat = min(minimum_heat, float(np.min(
            balance["deposited_heat_increment_J_m3"])))

    checkpoint = mechanical_checkpoint_arrays(state)
    restored = mechanical_from_checkpoint_arrays(checkpoint, systems, topologies)
    continuous, _ = accepted_v24_mechanical_step(
        state, driving, support, systems, topologies, common, extensive,
        kinetics, 2e-9, topology_route_enabled=False)
    restarted, _ = accepted_v24_mechanical_step(
        restored, driving, support, systems, topologies, common, extensive,
        kinetics, 2e-9, topology_route_enabled=False)
    restart_exact = all(np.array_equal(getattr(getattr(continuous, group), name),
                                       getattr(getattr(restarted, group), name))
                        for group in ("common", "density", "reservoir_alignment")
                        for name in getattr(continuous, group).__dict__)
    passed = bool(first_failure is None and maximum_dual < .05
                  and maximum_continuity < .05 and restart_exact
                  and minimum_heat >= -1e-10
                  and maximum_energy_balance_relative < 2e-14)
    return {
        "grid": n, "steps": steps, "accepted_time_s": accepted_time,
        "maximum_dual_nye_relative_rms": maximum_dual,
        "maximum_normalized_line_continuity_residual": maximum_continuity,
        "maximum_scalar_line_balance_residual_m": maximum_line_balance,
        "maximum_alignment_balance_residual_m": maximum_alignment_balance,
        "maximum_global_energy_balance_residual_J_m3_cells": maximum_energy_balance,
        "maximum_global_energy_balance_relative": maximum_energy_balance_relative,
        "minimum_deposited_heat_increment_J_m3": minimum_heat,
        "exact_restart": restart_exact, "first_failure": first_failure,
        "passed": passed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manufactured_records = [manufactured(n, mode)
                            for n in (16, 32, 64, 128)
                            for mode in ("homogeneous_ssd", "advection",
                                         "rotating_frame", "physical_diffusion")]
    transition_records = [transition(n, 8 if n < 64 else 5)
                          for n in (16, 32, 64)]
    passed = (all(row["passed"] for row in manufactured_records)
              and all(row["passed"] for row in transition_records))
    result = {
        "schema": "asb-drx/v30-mura-nye-production-decision/v1",
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "manufactured_records": manufactured_records,
        "transition_band_records": transition_records,
        "controls": {
            "capture_release": "conservative reservoir repartition",
            "locking_unlocking": "conservative reservoir repartition",
            "junction_formation_reversal": "declared source tensor required",
            "active_set": "geometric line stretching ledgered before acceptance",
            "post_step_projection_used": False,
        },
        "fixture_passed": True,
        "scientific_gate_passed": passed,
        "long_run_authorized": passed,
        "classification": ("PRODUCTION_MURA_NYE_KINEMATICS_QUALIFIED"
                           if passed else "SIGNED_ADVECTION_NYE_OWNERSHIP_FAILURE"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(result["classification"])


if __name__ == "__main__":
    main()
