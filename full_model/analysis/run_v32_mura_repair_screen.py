#!/usr/bin/env python3
"""Compact bounded screen of the V32 Mura work-budget repair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from full_model.analysis.run_v30_mura_tier_b1_case import create_case, load_checkpoint
from full_model.production.common_tensorial_wall import CommonWallDriving
from full_model.production.v24_mechanical_wall import accepted_v24_mechanical_step


def state_metrics(state):
    signed = sum(
        np.asarray(getattr(state.density, f"{stem}_plus_m2"))
        - np.asarray(getattr(state.density, f"{stem}_minus_m2"))
        for stem in ("mobile", "forest", "wall_tangle", "wall_ordered"))
    total = sum(
        np.asarray(getattr(state.density, f"{stem}_plus_m2"))
        + np.asarray(getattr(state.density, f"{stem}_minus_m2"))
        for stem in ("mobile", "forest", "wall_tangle", "wall_ordered"))
    alpha = np.sum(state.common.family_nye_m1, axis=2)
    return {
        "alpha_rms_m1": float(np.sqrt(np.mean(alpha**2))),
        "maximum_signed_density_m2": float(np.max(np.linalg.norm(signed, axis=2))),
        "mean_total_density_m2": float(np.mean(np.sum(total, axis=2))),
        "orientation_span_deg": float(np.ptp(state.common.orientation_rad)*180/np.pi),
        "slip_rms": float(np.sqrt(np.mean(state.common.slip**2))),
    }


def advance(checkpoint, grid, requested_dt, physical_horizon_s):
    data = create_case(grid, "mechanical_heterogeneity", 42, 1e-5)
    _, fixed, support, systems, topologies, common, extensive, kinetics, _ = data
    state, metadata = load_checkpoint(checkpoint, systems, topologies)
    elapsed = 0.0; steps = 0; ledgers = []
    while elapsed < physical_horizon_s-1e-18:
        strain = 1e4*(float(metadata["physical_time_s"])+elapsed)
        driving = CommonWallDriving(
            mean_strain=np.array([[0.0, .5*strain], [.5*strain, 0.0]]),
            fixed_eigenstrain=fixed)
        request = min(requested_dt, physical_horizon_s-elapsed)
        state, ledger = accepted_v24_mechanical_step(
            state, driving, support, systems, topologies, common, extensive,
            kinetics, request, mura_work_budget_mode="energy_limited")
        elapsed += ledger["accepted_dt_s"]; steps += 1; ledgers.append(ledger)
        if steps > 100:
            raise RuntimeError("bounded V32 screen exceeded 100 steps")
    state.validate(systems, topologies)
    maximum_first_law = max(abs(item["mura_balance_ledger"][
        "global_work_minus_heat_storage_residual_J_m3_cells"])
        for item in ledgers)
    maximum_source = max(abs(item["mura_balance_ledger"][
        "recoverable_elastic_energy_release_J_m3_cells"]) for item in ledgers)
    return state, {
        "requested_dt_s": requested_dt,
        "physical_horizon_s": elapsed,
        "steps": steps,
        "minimum_event_scale": min(item["mura_event_scale"] for item in ledgers),
        "stalled_families_seen": sorted(set(index for item in ledgers
            for index in item["mura_work_budget"]["stalled_families"])),
        "full_physical_stall_count": sum(item["mura_work_budget"]["physical_stall"]
                                         for item in ledgers),
        "maximum_first_law_relative": maximum_first_law/max(maximum_source, 1.0),
        "all_hard_invariants_passed": all(item["nye_suboperator_audit"][
            "accepted_step_hard_invariant_passed"] for item in ledgers),
        "post_step_projection_used": any(item["nye_suboperator_audit"][
            "post_step_projection_used"] for item in ledgers),
        "metrics": state_metrics(state),
    }


def relative(a, b):
    return abs(a-b)/max(abs(a), abs(b), 1e-30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v31-cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    accepted_cfl = {64: 2.0918202586737654e-10,
                    128: 9.540876989319022e-11}
    refined_dt = {64: 1.0e-10, 128: 4.5e-11}
    cases = []
    for grid in (64, 128):
        case_dir = args.v31_cases/f"mechanical_heterogeneity_{grid}"
        status = json.loads((case_dir/"status.json").read_text())
        checkpoint = case_dir/status["latest_checkpoint"]
        horizon = 4.0*accepted_cfl[grid]
        coarse_state, coarse = advance(checkpoint, grid, 2e-9, horizon)
        fine_state, fine = advance(checkpoint, grid, refined_dt[grid], horizon)
        del coarse_state, fine_state
        differences = {name: relative(coarse["metrics"][name],
                                      fine["metrics"][name])
                       for name in coarse["metrics"]}
        passed = (coarse["all_hard_invariants_passed"]
                  and fine["all_hard_invariants_passed"]
                  and not coarse["post_step_projection_used"]
                  and not fine["post_step_projection_used"]
                  and coarse["maximum_first_law_relative"] < 2e-11
                  and fine["maximum_first_law_relative"] < 2e-11
                  and max(differences.values()) < .05)
        cases.append({"grid": grid, "checkpoint": checkpoint.name,
                      "coarse": coarse, "refined": fine,
                      "timestep_relative_differences": differences,
                      "passed": bool(passed)})
    local_gate = all(case["passed"] for case in cases)
    result = {
        "schema": "asb-drx/v32-mura-repair-screen/v1",
        "cases": cases,
        "local_repair_gate_passed": local_gate,
        "b2_temperature_rate_seed_matrix_authorized": False,
        "authorization_reason": (
            "The local repair is admissible and timestep stable, but V31 "
            "matched-horizon spatial convergence remains false. The only "
            "justified continuation is a repaired 64/128 matched-horizon "
            "mechanical pair before temperature/rate/seed expansion."
            if local_gate else
            "The bounded local repair gate failed; no continuation is authorized."),
        "justified_next_cases": ([{
            "condition": "mechanical_heterogeneity", "grids": [64, 128],
            "start_strain": .03, "target_strain": .05,
            "temperature_K": 1100.0, "strain_rate_s": 1e4, "seed": 42,
            "purpose": "matched-horizon convergence of repaired family-wise stall"
        }] if local_gate else []),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"local_gate": local_gate,
                      "matrix_authorized": False}, sort_keys=True))


if __name__ == "__main__":
    main()
