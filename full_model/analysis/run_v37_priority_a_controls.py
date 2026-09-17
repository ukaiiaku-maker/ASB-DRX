#!/usr/bin/env python3
"""V37 Priority-A current-source controls and interval audit.

The controls distinguish exact complete-state equality, scalar-density equality
with a tensorial contrast, full material exchange, and proposal reversal.  All
cases use the production complete-energy/common-state recurrent cycle at zero
applied front work.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from full_model.analysis.run_v34_finite_coupled_response import (
    I3State, I3Controls, resolved_bicrystal, run_i3_cycle,
)
from full_model.analysis.run_v36_recurrent_physical_response import (
    geometric_envelope,
)
from full_model.production.common_front_state import (
    reconstruct_common, reconstruct_mechanical_state,
)
from full_model.production.coupled_front_production import (
    initialize_coupled_front_runtime,
)
from full_model.production.tensorial_nye import (
    plastic_distortion_from_slip,
)


SCHEMA = "asb-drx/v37/current-source-controls/v1"


def _utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _source_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip()


def _array_groups(state):
    result = {}
    for owner_name in ("parent", "child", "wake"):
        owner = getattr(state.common_front, owner_name)
        for item in fields(owner):
            result[f"common.{owner_name}.{item.name}"] = np.asarray(
                getattr(owner, item.name))
        density = getattr(state.common_front, owner_name+"_density")
        alignment = getattr(state.common_front, owner_name+"_alignment")
        if density is not None:
            for item in fields(density):
                result[f"density.{owner_name}.{item.name}"] = np.asarray(
                    getattr(density, item.name))
        if alignment is not None:
            for item in fields(alignment):
                result[f"alignment.{owner_name}.{item.name}"] = np.asarray(
                    getattr(alignment, item.name))
    return result


def compare_complete_owners(state, left="parent", right="child"):
    groups = _array_groups(state)
    rows = []
    for prefix in ("common", "density", "alignment"):
        names = sorted(name for name in groups
                       if name.startswith(prefix+"."+left+"."))
        for name in names:
            suffix = name.split(".", 2)[2]
            other = f"{prefix}.{right}.{suffix}"
            a = groups[name]; b = groups[other]
            scale = max(float(np.max(np.abs(a))),
                        float(np.max(np.abs(b))), 1.0)
            rows.append({
                "field": f"{prefix}.{suffix}",
                "exact": bool(np.array_equal(a, b)),
                "maximum_abs_difference": float(np.max(np.abs(a-b))),
                "maximum_relative_difference": float(
                    np.max(np.abs(a-b))/scale),
            })
    return {
        "exact_complete_state_equal": all(row["exact"] for row in rows),
        "maximum_abs_difference": max(
            (row["maximum_abs_difference"] for row in rows), default=0.0),
        "maximum_relative_difference": max(
            (row["maximum_relative_difference"] for row in rows), default=0.0),
        "fields": rows,
    }


def compare_complete_recurrent_state(state):
    parent_child = compare_complete_owners(state, "parent", "child")
    parent_wake = compare_complete_owners(state, "parent", "wake")
    return {
        "parent_child": parent_child,
        "parent_wake": parent_wake,
        "exact_complete_recurrent_state_equal": bool(
            parent_child["exact_complete_state_equal"]
            and parent_wake["exact_complete_state_equal"]),
    }


def equalize_complete_recurrent_state(context, state):
    """Populate inactive wake history without changing its zero support."""
    common = state.common_front
    parent = common.parent
    front = replace(
        common.front, child=common.front.parent,
        recovered_wake=common.front.parent)
    common = replace(
        common, front=front, child=parent, wake=parent,
        child_density=common.parent_density,
        wake_density=common.parent_density,
        child_alignment=common.parent_alignment,
        wake_alignment=common.parent_alignment)
    mechanical = reconstruct_mechanical_state(
        common, context["spacing_m"], context["systems"],
        context["topologies"])
    return replace(state, common_front=common, mechanical=mechanical)


def scalar_equal_tensor_contrast(context, state, slip_contrast=2.0e-3):
    """Keep every line scalar fixed but add a compatible uniform child slip."""
    child = state.common_front.child
    slip = np.asarray(child.slip).copy()
    slip[..., 0] += float(slip_contrast)
    beta = plastic_distortion_from_slip(
        slip, context["systems"], child.orientation_rad)
    child = replace(child, slip=slip, beta_p=beta)
    common_front = replace(state.common_front, child=child)
    mechanical = reconstruct_mechanical_state(
        common_front, context["spacing_m"], context["systems"],
        context["topologies"])
    return replace(state, common_front=common_front, mechanical=mechanical)


def exchange_initial_materials(context, state):
    """Exchange complete parent/child owners and phase labels at fixed geometry."""
    common = state.common_front
    front = common.front
    if (common.ledger.attempted_commits != 0
            or np.any(common.boundary_plus_m2)
            or np.any(common.boundary_minus_m2)
            or np.any(common.boundary_junction_m2)):
        raise ValueError("material exchange control requires pristine initial state")
    chi = 1.0-np.asarray(front.chi)
    exchanged_front = replace(
        front, parent=front.child, child=front.parent,
        recovered_wake=front.child, chi=chi.copy(),
        processed_max=chi.copy(), cleanup_max=chi.copy(),
        boundary_signed_density_m2=-front.boundary_signed_density_m2)
    exchanged_common = replace(
        common, front=exchanged_front,
        parent=common.child, child=common.parent, wake=common.child,
        parent_density=common.child_density,
        child_density=common.parent_density,
        wake_density=common.child_density,
        parent_alignment=common.child_alignment,
        child_alignment=common.parent_alignment,
        wake_alignment=common.child_alignment,
        interface_nye_m1=-common.interface_nye_m1)
    eta = np.asarray(state.eta)[..., ::-1].copy()
    runtime = initialize_coupled_front_runtime(
        exchanged_front, eta[..., 1]-eta[..., 0], normal_axis=0,
        periodic=True)
    mechanical = reconstruct_mechanical_state(
        exchanged_common, context["spacing_m"], context["systems"],
        context["topologies"])
    return I3State(mechanical, exchanged_common, runtime, eta)


def _ledger_delta(after, before):
    return {item.name: float(getattr(after, item.name)-getattr(before, item.name))
            for item in fields(after)}


def _owner_line_means(state):
    result = {}
    for name in ("parent", "child", "wake"):
        owner = getattr(state.common_front, name)
        total = (owner.mobile_plus_m2+owner.mobile_minus_m2
                 +owner.forest_plus_m2+owner.forest_minus_m2
                 +owner.wall_plus_m2+owner.wall_minus_m2)
        total = np.sum(total, axis=2)+np.sum(owner.junction_m2, axis=2)
        result[name] = float(np.mean(total))
    return result


def run_interval(context, state, *, direction, dt_s=5.0e-6,
                 proposal_fraction=.0625, mura_enabled=True,
                 front_enabled=True, mean_shear=.01):
    trial = geometric_envelope(
        state.eta, proposal_fraction, direction=direction)
    controls = I3Controls(
        mura_enabled=mura_enabled, front_enabled=front_enabled,
        driving_pressure_a_to_b_Pa=0.0,
        applied_pressure_a_to_b_Pa=0.0,
        geometric_probe_pressure_Pa=1.0e8,
        trial_dt_s=dt_s, front_dt_s=dt_s)
    from full_model.production.common_tensorial_wall import CommonWallDriving
    grid = state.eta.shape[0]
    driving = CommonWallDriving(
        mean_strain=np.array([[0.0, mean_shear], [mean_shear, 0.0]]),
        fixed_eigenstrain=np.zeros((grid, grid, 2, 2)))
    front0 = state.front_runtime.ledger
    common0 = state.common_front.ledger
    wall0 = time.perf_counter()
    updated, audit = run_i3_cycle(context, state, trial, driving, controls)
    elapsed = time.perf_counter()-wall0
    front = audit["front_decision"]
    proposal_signed = (None if front is None else
                       float(front["proposed_signed_volume_m3"]))
    accepted_signed = float(audit["sweep"]["net_m3"])
    interface_area = (None if front is None else
                      float(front["interface_area_m2"]))
    accepted_displacement = (0.0 if not interface_area else
                             accepted_signed/interface_area)
    component_displacements = [float(row["normal_displacement_m"])
                               for row in audit["sweep"]["components"]]
    front_delta = _ledger_delta(updated.front_runtime.ledger, front0)
    common_delta = _ledger_delta(updated.common_front.ledger, common0)
    return updated, {
        "wall_seconds": elapsed,
        "direction": int(direction),
        "proposal_fraction": float(proposal_fraction),
        "front_published": bool(audit["candidate_sweep_published"]),
        "classification": None if front is None else front["classification"],
        "gross_directional_activity_s": None if front is None else float(
            front["gross_channel_activity_s"]),
        "raw_constitutive_net_velocity_m_s": None if front is None else float(
            front["net_velocity_a_to_b_m_s"]),
        "proposed_signed_volume_m3": proposal_signed,
        "accepted_signed_volume_m3": accepted_signed,
        "accepted_absolute_volume_m3": float(audit["sweep"]["absolute_m3"]),
        "accepted_positive_volume_m3": float(audit["sweep"]["positive_m3"]),
        "accepted_negative_volume_m3": float(audit["sweep"]["negative_m3"]),
        "accepted_contour_displacement_m": accepted_displacement,
        "accepted_contour_displacement_cells": (
            accepted_displacement/context["spacing_m"]),
        "accepted_contour_displacement_interface_widths": (
            accepted_displacement/context["interface_width_m"]),
        "component_contour_displacements_m": component_displacements,
        "interface_area_m2": interface_area,
        "represented_thickness_m": context["represented_thickness_m"],
        "complete_candidate_endpoints": audit[
            "complete_directional_kinetics"],
        "complete_energy_decision": audit["complete_energy"]["front_decision"],
        "front_ledger_increment": front_delta,
        "common_ledger_increment": common_delta,
        "first_passage_volume_m3": max(
            front_delta["a_to_b_swept_volume_m3"]
            +front_delta["b_to_a_swept_volume_m3"]
            -front_delta["revisit_volume_m3"], 0.0),
        "revisit_volume_m3": front_delta["revisit_volume_m3"],
        "processed_line_m": common_delta["processed_line_m"],
        "boundary_storage_line_m": common_delta["boundary_line_m"],
        "annihilated_line_m": common_delta["annihilated_line_m"],
        "sink_line_m": common_delta["sink_line_m"],
        "owner_total_line_density_mean_m2_before": _owner_line_means(state),
        "owner_total_line_density_mean_m2_after": _owner_line_means(updated),
        "mura_clock": audit["mura"],
        "front_clock_s": float(dt_s),
        "actual_inventory_change": audit["actual_inventory_change"],
        "temperature_range_K": audit["temperature_range_K"],
    }


def _file_hash_at(commit, path):
    try:
        content = subprocess.check_output(["git", "show", f"{commit}:{path}"])
    except subprocess.CalledProcessError:
        return None
    return hashlib.sha256(content).hexdigest()


def source_equivalence_audit():
    commits = {
        "local_n128_scientific": "31faf90ec82d19f8c034bebad6a02990d4f4e09e",
        "hpc_n192_scientific": "890cb8906a9772d8bd5c5eb43164ecd44ad2720f",
        "v37_current": _source_commit(),
    }
    paths = (
        "full_model/analysis/run_v36_recurrent_physical_response.py",
        "full_model/analysis/run_v34_finite_coupled_response.py",
        "full_model/production/coupled_front_event.py",
        "full_model/production/coupled_front_production.py",
        "full_model/production/complete_front_energy.py",
        "full_model/production/common_front_state.py",
        "full_model/production/v24_mechanical_wall.py",
    )
    hashes = {role: {path: _file_hash_at(commit, path) for path in paths}
              for role, commit in commits.items()}
    return {"commits": commits, "module_sha256": hashes}


def build_controls(grid=128, dt_s=5.0e-6, proposal_fraction=.0625):
    base_context = resolved_bicrystal(
        grid=grid, length_m=3.2e-6, interface_width_m=4.0e-7,
        child_line_fraction=.35, temperature_K=1100.0)
    equal_context = resolved_bicrystal(
        grid=grid, length_m=3.2e-6, interface_width_m=4.0e-7,
        child_line_fraction=1.0, temperature_K=1100.0)
    scalar_named_equal0 = equal_context["state"]
    scalar_named_recurrent = compare_complete_recurrent_state(
        scalar_named_equal0)
    equal0 = equalize_complete_recurrent_state(
        equal_context, scalar_named_equal0)
    equal_before = compare_complete_recurrent_state(equal0)
    equal_mura, equal_mura_audit = run_interval(
        equal_context, equal0, direction=1, dt_s=dt_s,
        proposal_fraction=proposal_fraction, front_enabled=False)
    equal_after_mura = compare_complete_recurrent_state(equal_mura)
    _, exact_audit = run_interval(
        equal_context, equal0, direction=1, dt_s=dt_s,
        proposal_fraction=proposal_fraction)

    scalar_state = scalar_equal_tensor_contrast(equal_context, equal0)
    scalar_compare = compare_complete_recurrent_state(scalar_state)
    _, scalar_audit = run_interval(
        equal_context, scalar_state, direction=1, dt_s=dt_s,
        proposal_fraction=proposal_fraction)

    base = base_context["state"]
    _, base_audit = run_interval(
        base_context, base, direction=1, dt_s=dt_s,
        proposal_fraction=proposal_fraction)
    exchanged = exchange_initial_materials(base_context, base)
    _, exchange_audit = run_interval(
        base_context, exchanged, direction=-1, dt_s=dt_s,
        proposal_fraction=proposal_fraction)
    _, reversal_audit = run_interval(
        base_context, base, direction=-1, dt_s=dt_s,
        proposal_fraction=proposal_fraction)

    return {
        "schema": SCHEMA,
        "created_utc": _utc_now(),
        "source_commit": _source_commit(),
        "configuration": {
            "grid": int(grid), "dt_s": float(dt_s),
            "proposal_fraction": float(proposal_fraction),
            "length_m": 3.2e-6, "interface_width_m": 4.0e-7,
            "temperature_K": 1100.0, "applied_front_pressure_Pa": 0.0,
        },
        "source_equivalence": source_equivalence_audit(),
        "exact_complete_state_control": {
            "unreconciled_scalar_named_initial": scalar_named_recurrent,
            "before_mura_owner_comparison": equal_before,
            "after_mura_owner_comparison": equal_after_mura,
            "mura_only_interval": equal_mura_audit,
            "coupled_interval": exact_audit,
        },
        "equal_scalar_tensor_contrast_control": {
            "owner_comparison": scalar_compare,
            "declared_slip_contrast": 2.0e-3,
            "interval": scalar_audit,
        },
        "material_exchange_control": {
            "original": base_audit,
            "complete_exchange": exchange_audit,
            "velocity_exchange_residual_m_s": (
                (base_audit["raw_constitutive_net_velocity_m_s"] or 0.0)
                +(exchange_audit["raw_constitutive_net_velocity_m_s"] or 0.0)),
        },
        "proposal_reversal_only_control": reversal_audit,
        "historical_case_classification": {
            "547ea643_equal_state": (
                "SUPERSEDED_SCALAR_NAME_NOT_CURRENT_COMPLETE_STATE_FIXTURE"),
            "v36_stationary_endpoint": "VALID_STANDALONE_COMPLETE_ENDPOINT_FIXTURE",
            "v37_exact_complete_state": "CURRENT_SOURCE_RECURRENT_CONTROL",
            "v37_complete_exchange": "CURRENT_SOURCE_MATERIAL_EXCHANGE_CONTROL",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--dt-s", type=float, default=5.0e-6)
    parser.add_argument("--proposal-fraction", type=float, default=.0625)
    args = parser.parse_args()
    result = build_controls(args.grid, args.dt_s, args.proposal_fraction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({
        "output": str(args.output), "source_commit": result["source_commit"],
        "grid": args.grid,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
