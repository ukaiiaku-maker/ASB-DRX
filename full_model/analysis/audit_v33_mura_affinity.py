#!/usr/bin/env python3
"""Checkpoint-fork audit of the V32 Mura work-budget complementarity.

The input checkpoint is immutable.  Every scenario advances an in-memory copy
by one accepted production step and reports both the raw grid sum and its
volume-average/unit-thickness integral.  No output is written beside the
checkpoint.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from full_model.analysis.run_v30_mura_tier_b1_case import create_case, load_checkpoint
from full_model.production.common_tensorial_wall import (
    CommonWallDriving, resolved_driving_components, wall_residual,
)
from full_model.production.mura_kinematics import (
    family_plastic_flow_from_signed_alignment,
)
from full_model.production.tensorial_nye import rotated_system_fields
from full_model.production.v24_mechanical_wall import (
    _elastic_energy_sum_J_m3_cells, _mura_budget_audit,
    _resolved_elastic_energy_sum_J_m3_cells, accepted_v24_mechanical_step,
)
from full_model.production.wall_topology_supply import (
    accepted_mura_transport_capture_step,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_sha() -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def energy_forms(raw_sum, cells, spacing_m, dt_s):
    if raw_sum is None:
        return None
    raw = float(raw_sum)
    average = raw/float(cells)
    return {
        "raw_sum_J_m3_cells": raw,
        "volume_average_J_m3": average,
        "unit_thickness_total_J": raw*spacing_m**2,
        "raw_sum_rate_W_m3_cells": raw/dt_s,
        "volume_average_rate_W_m3": average/dt_s,
        "unit_thickness_total_rate_W": raw*spacing_m**2/dt_s,
    }


def compact_step(ledger, cells, spacing_m):
    budget = ledger["mura_work_budget"]
    accepted = budget["accepted"]
    dt_s = float(ledger["accepted_dt_s"])
    energy = {}
    for name in (
        "plastic_work_J_m3_cells",
        "recoverable_elastic_energy_release_J_m3_cells",
        "existing_defect_energy_release_J_m3_cells",
        "proposed_defect_energy_change_J_m3_cells",
        "proposed_line_creation_energy_J_m3_cells",
        "dissipative_drag_and_heat_J_m3_cells",
    ):
        energy[name.removesuffix("_J_m3_cells")] = energy_forms(
            accepted[name], cells, spacing_m, dt_s)
    locking = ledger["locking_unlocking"]
    return {
        "accepted_dt_s": dt_s,
        "event_scale": float(ledger["mura_event_scale"]),
        "family_event_scales": [float(x) for x in
                                ledger["mura_family_event_scales"]],
        "stalled_families": budget["stalled_families"],
        "trial_count": int(budget["trial_count"]),
        "physical_stall": bool(budget["physical_stall"]),
        "event_extent_changes_constitutive_amplitude": True,
        "event_extent_changes_accepted_dt_or_loading_clock": False,
        "energy": energy,
        "proposed_family_mechanical_work_J_m3_cells": accepted[
            "proposed_family_work_before_stall_J_m3_cells"],
        "family_plastic_work_J_m3_cells": accepted[
            "family_plastic_work_J_m3_cells"],
        "family_line_creation_energy_J_m3_cells": accepted[
            "family_line_creation_energy_J_m3_cells"],
        "reaction_extents": accepted["reaction_extents"],
        "locking_unlocking_abs_line_m2_cells": float(sum(
            np.sum(np.abs(locking["sign"][sign]["accepted_line_m2"]))
            for sign in ("plus", "minus"))),
        "minimum_heat_increment_J_m3": float(np.min(
            ledger["mura_balance_ledger"]["deposited_heat_increment_J_m3"])),
        "first_law_residual_J_m3_cells": float(ledger[
            "mura_balance_ledger"][
                "global_work_minus_heat_storage_residual_J_m3_cells"]),
        "hard_invariant_passed": bool(ledger["nye_suboperator_audit"][
            "accepted_step_hard_invariant_passed"]),
        "post_step_projection_used": bool(ledger["nye_suboperator_audit"][
            "post_step_projection_used"]),
    }


def discrete_affinity_by_family(state, driving, support, systems, topologies,
                                common, extensive, dt_s):
    del extensive
    mechanics = replace(
        common, wall_order_enabled=False, extensive_wall_partition_enabled=True,
        transport_scheme="upwind", mobile_correlation_diffusivity_m2_s=0.0)
    drive = resolved_driving_components(
        state.common, driving, systems, topologies, mechanics)
    resolved = CommonWallDriving(glide_speed_m_s=drive["speed_m_s"],
                                 resolved_stress_Pa=drive["raw_stress_Pa"])
    residual = wall_residual(
        state.common, resolved, systems, topologies, mechanics)
    _, slip_directions, plane_normals = rotated_system_fields(
        systems, state.common.orientation_rad)
    velocity_plus = drive["speed_m_s"][..., None]*slip_directions
    velocity_minus = -velocity_plus
    courant_rate = np.max(np.sum(np.abs(velocity_plus[..., :2]), axis=-1)
                              /common.spacing_m)
    orientation_rate = np.max(np.abs(residual.state_rate.orientation_rad))
    accepted_dt = min(float(dt_s), 0.8/max(float(courant_rate), 1e-300),
                      .02/max(float(orientation_rate), 1e-300))
    flow = family_plastic_flow_from_signed_alignment(
        state.reservoir_alignment.mobile_plus_m2,
        state.reservoir_alignment.mobile_minus_m2,
        velocity_plus, velocity_minus, systems, state.common.orientation_rad)
    schmid = np.einsum("...ai,...aj->...aij", slip_directions, plane_normals)
    schmid_norm2 = np.sum(schmid*schmid, axis=(-2, -1))
    slip_rate = np.divide(
        np.sum(flow*schmid, axis=(-2, -1)), schmid_norm2,
        out=np.zeros_like(schmid_norm2), where=schmid_norm2 > 0.0)
    mechanical = accepted_dt*np.sum(
        drive["raw_stress_Pa"]*slip_rate, axis=(0, 1))
    tolerance = 1e-14*max(float(np.max(np.abs(mechanical))), 1.0)
    complementarity = mechanical > tolerance
    active_planar_velocity = velocity_plus[..., :2]*complementarity[
        None, None, :, None]
    active_courant_rate = np.max(
        np.sum(np.abs(active_planar_velocity), axis=-1)/common.spacing_m)
    active_flow = flow*complementarity[None, None, :, None, None]
    active_total_flow = np.sum(active_flow, axis=2)
    active_orientation_rate = common.orientation_spin_weight*.5*(
        active_total_flow[..., 1, 0]-active_total_flow[..., 0, 1])
    accepted_dt = min(
        float(dt_s), 0.8/max(float(active_courant_rate), 1e-300),
        .02/max(float(np.max(np.abs(active_orientation_rate))), 1e-300))
    mechanical = accepted_dt*np.sum(
        drive["raw_stress_Pa"]*slip_rate, axis=(0, 1))

    elastic_before = _resolved_elastic_energy_sum_J_m3_cells(drive)

    def evaluate(extent, label):
        extent = np.asarray(extent, dtype=float)
        extent4 = extent[None, None, :, None]
        masked_velocity_plus = velocity_plus*extent4
        masked_velocity_minus = velocity_minus*extent4
        masked_flow = flow*extent[None, None, :, None, None]
        try:
            density, alignment, capture = accepted_mura_transport_capture_step(
                state.density, state.reservoir_alignment,
                masked_velocity_plus, masked_velocity_minus, support, systems,
                state.common.orientation_rad, common.spacing_m, accepted_dt,
                topologies)
            del alignment
            beta_after = state.common.beta_p+accepted_dt*np.sum(masked_flow, axis=2)
            elastic_after = _elastic_energy_sum_J_m3_cells(
                state.common, beta_after, driving, common)
            release = (None if elastic_before is None
                       else elastic_before-elastic_after)
            audit, _, _, _, _ = _mura_budget_audit(
                state, density, capture, masked_flow, drive["raw_stress_Pa"],
                schmid, schmid_norm2, 1.0, accepted_dt, systems, topologies,
                common, release)
            return {"label": label,
                    "family_extent": [float(x) for x in extent],
                    "result": "EVALUATED", **audit}
        except (ValueError, RuntimeError) as error:
            return {"label": label,
                    "family_extent": [float(x) for x in extent],
                    "result": "INADMISSIBLE_KINEMATICS",
                    "error": f"{type(error).__name__}: {error}"}

    families = []
    for family in range(len(systems)):
        extent = np.zeros(len(systems)); extent[family] = 1.0
        full = evaluate(extent, f"family_{family}_full_event")
        extent[family] = 2.0**-20
        tangent = evaluate(extent, f"family_{family}_near_zero_event")
        families.append({
            "family": family,
            "frozen_v32_mechanical_work_J_m3_cells": float(mechanical[family]),
            "frozen_v32_complementarity_active": bool(complementarity[family]),
            "full_event": full,
            "near_zero_event": tangent,
        })
    all_families = evaluate(np.ones(len(systems), dtype=bool), "all_families")
    gated = evaluate(complementarity, "v32_complementarity_mask")
    return {
        "accepted_dt_s": float(accepted_dt),
        "work_tolerance_J_m3_cells": float(tolerance),
        "frozen_v32_family_mechanical_work_J_m3_cells": [float(x) for x in mechanical],
        "frozen_v32_complementarity_mask": [bool(x) for x in complementarity],
        "discrete_family_trials": families,
        "all_family_full_event": all_families,
        "complementarity_masked_full_event": gated,
        "affinity_definition": (
            "recoverable elastic energy release minus exact defect free-energy "
            "change for the complete isolated-family finite-volume event"),
        "nonadditivity_warning": (
            "isolated family affinities are directional finite-event tests; "
            "elastic quadratic and defect-energy coupling prevent summing them"),
    }


def run_checkpoint(checkpoint, grid, trial_dt_s, strain_rate_s,
                   condition="mechanical_heterogeneity"):
    (initial, fixed, support, systems, topologies, common, extensive,
     kinetics, spacing) = create_case(grid, condition, 42, 1e-5)
    del initial
    state, metadata = load_checkpoint(checkpoint, systems, topologies)
    strain = float(metadata["applied_strain"])
    base_mean = np.array([[0.0, .5*strain], [.5*strain, 0.0]])
    base = CommonWallDriving(mean_strain=base_mean, fixed_eigenstrain=fixed)
    cells = grid*grid
    scenarios = []

    def execute(name, state_arg=state, driving=base, parameters=common,
                dt=trial_dt_s):
        _, ledger = accepted_v24_mechanical_step(
            state_arg, driving, support, systems, topologies, parameters,
            extensive, kinetics, dt, topology_route_enabled=False,
            mura_work_budget_mode="energy_limited")
        scenarios.append({"name": name, "requested_dt_s": float(dt),
                          **compact_step(ledger, cells, spacing)})

    for factor in (1.0, .5, .25, .125, .0625):
        execute(f"dt_refinement_{factor:g}", dt=trial_dt_s*factor)
    execute("fixed_total_strain_hold")
    load_increment = strain_rate_s*trial_dt_s
    continued_mean = np.array([
        [0.0, .5*(strain+load_increment)],
        [.5*(strain+load_increment), 0.0]])
    execute("continued_load_plus_requested_increment",
            driving=CommonWallDriving(mean_strain=continued_mean,
                                      fixed_eigenstrain=fixed))
    execute("mechanical_constraint_release",
            driving=CommonWallDriving(mean_strain=base_mean,
                                      fixed_eigenstrain=np.zeros_like(fixed)))
    reaction_off = replace(
        common, lock_barrier_eV=100.0, wall_barrier_eV=100.0,
        annihilation_barrier_eV=100.0, junction_barrier_eV=100.0)
    execute("recovery_exchange_reactions_off", parameters=reaction_off)
    warm_state = replace(state, common=replace(
        state.common, temperature_K=state.common.temperature_K+25.0))
    execute("temperature_plus_25K", state_arg=warm_state,
            parameters=replace(common, bath_temperature_K=
                               common.bath_temperature_K+25.0))
    perturb = 2.5e-5
    perturbed_mean = np.array([
        [0.0, .5*(strain+perturb)], [.5*(strain+perturb), 0.0]])
    execute("admissible_load_perturbation_plus_2p5e_5",
            driving=CommonWallDriving(mean_strain=perturbed_mean,
                                      fixed_eigenstrain=fixed))
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_source_sha": metadata.get("source_sha", "UNRECORDED"),
        "grid": grid, "step": int(metadata["step"]),
        "applied_strain": strain, "spacing_m": spacing,
        "cell_count": cells, "unit_thickness_m": 1.0,
        "trial_dt_s": trial_dt_s, "strain_rate_s": strain_rate_s,
        "event_scale_semantics": {
            "scaled": ["signed velocities", "family plastic-flow rate",
                       "slip increment", "beta_p/Nye increment",
                       "orientation increment"],
            "not_scaled": ["accepted_dt_s", "runner physical_time increment",
                           "runner imposed-strain clock increment",
                           "post-Mura reaction interval"],
        },
        "full_discrete_affinity": discrete_affinity_by_family(
            state, base, support, systems, topologies, common, extensive,
            trial_dt_s),
        "one_step_checkpoint_forks": scenarios,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--trial-dt-s", type=float, default=2e-9)
    parser.add_argument("--strain-rate-s", type=float, default=1e4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [run_checkpoint(path, args.grid, args.trial_dt_s,
                            args.strain_rate_s) for path in args.checkpoint]
    result = {
        "schema": "asb-drx/v33-mura-affinity-audit/v1",
        "diagnostic_worktree_head_before_evidence_commit": source_sha(),
        "diagnostic_script_sha256": sha256(Path(__file__)),
        "input_checkpoints_are_read_only": True,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"cases": len(cases), "output": str(args.output)},
                     sort_keys=True))


if __name__ == "__main__":
    main()
