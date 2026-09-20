#!/usr/bin/env python3
"""Transactional V39 recurrent Mura/front/thermal physical-time driver."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from full_model.analysis.run_v34_finite_coupled_response import (
    I3Controls, _energy_options, checkpoint_payload, resolved_bicrystal, run_i3_cycle,
    state_from_payload,
)
from full_model.analysis.run_v36_recurrent_physical_response import (
    driving_at_time, geometric_envelope,
)
from full_model.production.density_state_map import derived_density_fields
from full_model.production.complete_front_energy import evaluate_complete_front_energy
from full_model.production.tensorial_nye import nye_from_plastic_distortion
from full_model.production.v24_mechanical_wall import resolved_driving_components
from full_model.production.wall_topology_supply import reservoir_nye_m1


SCHEMA = "asb-drx/v39/transactional-common-horizon/v1"


def source_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        staged = os.environ.get("V40_SOURCE_SHA")
        if not staged:
            raise RuntimeError(
                "non-Git staged source requires V40_SOURCE_SHA") from None
        return staged


def atomic_npz(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
    temporary.replace(path)


def save_stage(path, state, context, metadata):
    payload = checkpoint_payload(state, context)
    payload["v39_stage_metadata_json"] = np.asarray(json.dumps(
        {"schema": SCHEMA, **metadata}, sort_keys=True))
    atomic_npz(path, payload)


def load_stage(path, context):
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    metadata = json.loads(str(payload.pop("v39_stage_metadata_json").item()))
    if metadata.get("schema") != SCHEMA:
        raise ValueError("unsupported V39 stage checkpoint")
    return state_from_payload(payload, context), metadata


def mura_to_time(context, state, duration_s, driving, maximum_substep_s=None,
                 mura_transport_operator="legacy_mixed"):
    elapsed = 0.0; audits = []
    tolerance = 64*np.finfo(float).eps*max(duration_s, 1e-300)
    while elapsed < duration_s-tolerance:
        requested = duration_s-elapsed
        if maximum_substep_s is not None:
            if float(maximum_substep_s) <= 0.0:
                raise ValueError("Mura maximum substep must be positive")
            requested = min(requested, float(maximum_substep_s))
        state, audit = run_i3_cycle(
            context, state, state.eta.copy(), driving,
            I3Controls(mura_enabled=True, front_enabled=False,
                       trial_dt_s=requested, front_dt_s=requested,
                       mura_transport_operator=mura_transport_operator))
        accepted = float(audit["mura"]["accepted_dt_s"])
        if accepted <= tolerance or accepted > requested+tolerance:
            raise RuntimeError("Mura multirate subcycle made invalid clock progress")
        elapsed += accepted; audits.append(audit["mura"])
    return state, audits, elapsed


def front_stage(context, state, duration_s, driving, proposal_fraction,
                front_exp_n, front_direction):
    envelope = geometric_envelope(
        state.eta, proposal_fraction, direction=front_direction)
    controls = I3Controls(
        mura_enabled=False, front_enabled=True,
        driving_pressure_a_to_b_Pa=0.0,
        applied_pressure_a_to_b_Pa=0.0,
        geometric_probe_pressure_Pa=1e8,
        trial_dt_s=duration_s, front_dt_s=duration_s,
        front_exp_n=front_exp_n)
    return run_i3_cycle(context, state, envelope, driving, controls)


def front_metrics(audit):
    decision = audit["front_decision"]
    energy_decision = audit["complete_energy"].get("front_decision")
    directional = audit.get("complete_directional_kinetics")
    interface_area = (None if decision is None else float(
        decision["interface_area_m2"]))
    sweep = float(audit["sweep"]["net_m3"])
    return {
        "front_classification": (None if decision is None
                                 else decision["classification"]),
        "front_kinetic_accepted": bool(
            decision is not None and decision["accepted"]),
        "front_proposed_signed_volume_m3": (
            None if decision is None else float(
                decision["proposed_signed_volume_m3"])),
        "front_thermodynamic_classification": (
            None if energy_decision is None else energy_decision[
                "classification"]),
        "front_thermodynamic_accepted": bool(
            energy_decision is not None and energy_decision["accepted"]),
        "front_candidate_delta_helmholtz_J": (
            None if energy_decision is None else float(
                energy_decision["delta_helmholtz_J"])),
        "front_candidate_available_change_J": (
            None if energy_decision is None else float(
                energy_decision["available_change_J"])),
        "front_external_work_J": (
            None if energy_decision is None else float(
                energy_decision["external_work_J"])),
        "front_reverse_edge_status": (
            None if directional is None else directional[
                "reverse_edge_status"]),
        "front_a_to_b_event_J": (
            None if directional is None else float(
                directional["a_to_b_event_J"])),
        "front_b_to_a_event_J": (
            None if directional is None else float(
                directional["b_to_a_event_J"])),
        "front_published": bool(audit["candidate_sweep_published"]),
        "front_signed_sweep_m3": sweep,
        "front_absolute_sweep_m3": float(audit["sweep"]["absolute_m3"]),
        "front_contour_displacement_m": (
            0.0 if not interface_area else sweep/interface_area),
        "front_complete_energy_delta_J": float(
            audit["complete_energy"]["delta_helmholtz_J"]),
        "front_rate_m_s": (None if decision is None else float(
            decision["net_velocity_a_to_b_m_s"])),
    }


def state_metrics(state, context, driving):
    fields = derived_density_fields(state.mechanical.density,
                                    context["topologies"])
    common = state.mechanical.common
    resolved = resolved_driving_components(
        common, driving, context["systems"], context["topologies"],
        context["wall_parameters"])
    area = context["spacing_m"]**2
    return {
        "total_line_m_per_m_thickness": float(np.sum(
            fields["rho_total_m2"], dtype=np.longdouble)*area),
        "ordered_line_m_per_m_thickness": float(np.sum(
            fields["rho_wall_ordered_m2"], dtype=np.longdouble)*area),
        "maximum_abs_slip": float(np.max(np.abs(common.slip))),
        "beta_p_rms": float(np.sqrt(np.mean(common.beta_p**2))),
        "family_nye_rms_m1": float(np.sqrt(np.mean(common.family_nye_m1**2))),
        "orientation_range_rad": float(np.max(common.orientation_rad)
                                       -np.min(common.orientation_rad)),
        "effective_stress_rms_Pa": float(np.sqrt(np.mean(
            resolved["effective_stress_Pa"]**2))),
        "temperature_minimum_K": float(np.min(common.temperature_K)),
        "temperature_maximum_K": float(np.max(common.temperature_K)),
    }


def stage_diagnostics(state, context, driving):
    """Grid-aware intensive/extensive diagnostics at a split-stage boundary."""
    density = state.mechanical.density
    common = state.mechanical.common
    spacing = context["spacing_m"]
    area = spacing**2
    line = {}
    for stem in ("mobile", "forest", "wall_tangle", "wall_ordered"):
        for sign in ("plus", "minus"):
            value = np.asarray(getattr(density, f"{stem}_{sign}_m2"))
            line[f"{stem}_{sign}_m_per_m_by_family"] = [float(x) for x in
                np.sum(value, axis=(0, 1), dtype=np.longdouble)*area]
            line[f"{stem}_{sign}_active_cells_by_family"] = [int(x) for x in
                np.count_nonzero(value > 0.0, axis=(0, 1))]
    line["junction_extent_m_per_m_by_topology"] = [float(x) for x in
        np.sum(density.junction_m2, axis=(0, 1), dtype=np.longdouble)*area]
    curl_nye = nye_from_plastic_distortion(common.beta_p, spacing)
    moment_nye = reservoir_nye_m1(
        state.mechanical.reservoir_alignment, context["systems"],
        common.orientation_rad, context["topologies"])["total"]
    # The reservoir first moments are a bulk line representation.  A moving
    # material support also contributes the separately owned product-rule
    # interface term.  The old curl-minus-reservoir diagnostic incorrectly
    # classified that declared surface contribution as a transfer error.
    interface_nye = np.asarray(state.common_front.interface_nye_m1)
    bulk_only_difference = curl_nye-moment_nye
    declared_difference = curl_nye-(moment_nye+interface_nye)
    child = np.asarray(state.eta[..., 1])
    interface = (child > 0.05) & (child < 0.95)
    bulk = ~interface
    def rms(field, mask=None):
        selected = field if mask is None else field[mask]
        return float(np.sqrt(np.mean(selected*selected))) if selected.size else 0.0
    energy = evaluate_complete_front_energy(
        state.common_front, state.eta, spacing_m=spacing,
        represented_thickness_m=context["represented_thickness_m"],
        **_energy_options(context, driving))
    components = {key: float(value) for key, value in asdict(energy).items()}
    components.update(helmholtz_J=float(energy.helmholtz_J),
                      internal_J=float(energy.internal_J))
    return {
        "line_inventory": line,
        "owner_support": {
            "child_fraction_mean": float(np.mean(child)),
            "parent_pure_cells": int(np.count_nonzero(child <= 0.05)),
            "child_pure_cells": int(np.count_nonzero(child >= 0.95)),
            "interface_cells": int(np.count_nonzero(interface)),
            "interface_area_proxy_m2_per_m": float(
                np.sum(np.sqrt(sum(g*g for g in np.gradient(child, spacing))))
                *area),
        },
        "nye": {
            "curl_beta_rms_m1": rms(curl_nye),
            "reservoir_moment_rms_m1": rms(moment_nye),
            "interface_product_rule_rms_m1": rms(interface_nye),
            "bulk_only_difference_rms_m1": rms(bulk_only_difference),
            "difference_rms_m1": rms(declared_difference),
            "difference_interface_rms_m1": rms(declared_difference, interface),
            "difference_bulk_rms_m1": rms(declared_difference, bulk),
            "declared_nye_identity": (
                "curl_beta = reservoir_first_moment + interface_product_rule "
                "+ representation_residual"),
        },
        "maximum_abs_beta_p": float(np.max(np.abs(common.beta_p))),
        "maximum_abs_slip": float(np.max(np.abs(common.slip))),
        "temperature_minimum_K": float(np.min(common.temperature_K)),
        "temperature_maximum_K": float(np.max(common.temperature_K)),
        "physical_energy": components,
    }


def run_case(output_dir, *, grid=16, macro_dt_s=1e-3, intervals=1,
             proposal_fraction=.0625, front_exp_n=1.0, restart=None,
             inject_post_front_failure=False, front_direction=1,
             mura_substeps_per_half=1,
             mura_transport_operator="legacy_mixed"):
    if front_direction not in (-1, 1):
        raise ValueError("front_direction must be -1 or +1")
    if int(mura_substeps_per_half) < 1:
        raise ValueError("Mura substeps per half must be positive")
    frozen_source_sha = source_sha()
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    context = resolved_bicrystal(
        grid=grid, length_m=3.2e-6, interface_width_m=4e-7,
        temperature_K=1100.0, child_line_fraction=.35)
    # V39 is retained as a restart/split-clock regression fixture.  Preserve
    # its historically qualified equilibrium dispatch so those tests isolate
    # checkpoint semantics rather than paying for V47 finite kinetics.  New
    # physical-response drivers use resolved_bicrystal's guarded 5% default.
    context["extensive_parameters"] = replace(
        context["extensive_parameters"],
        ordering_asymptotic_maximum_endpoint_distance_relative=1.0,
        ordering_asymptotic_certificate_mode="legacy_euclidean_heuristic")
    records = []; completed = 0; physical_time = 0.0
    pending = None
    partial_path = None
    if restart is not None:
        state, metadata = load_stage(restart, context)
        if int(metadata["grid"]) != grid:
            raise ValueError("restart grid differs from requested case")
        stored_macro_dt = float(metadata["macro_dt_s"])
        if (metadata["stage"] == "POST_MURA_PENDING"
                and stored_macro_dt != macro_dt_s):
            raise ValueError(
                "pending split-stage restart requires its original macro dt")
        completed = int(metadata["completed_intervals"])
        physical_time = float(metadata["physical_time_s"])
        records = list(metadata.get("records", []))
        pending = metadata if metadata["stage"] == "POST_MURA_PENDING" else None
        partial_path = Path(restart)
    else:
        state = context["state"]
    wall_start = time.perf_counter()
    for index in range(completed, intervals):
        midpoint = physical_time+0.5*macro_dt_s
        driving = driving_at_time(grid, .01, "hold", 0.0, midpoint)
        macro_start = state
        if pending is None:
            initial_stage = stage_diagnostics(
                macro_start, context, driving)
            state_after_pre, pre, pre_elapsed = mura_to_time(
                context, macro_start, 0.5*macro_dt_s, driving,
                maximum_substep_s=.5*macro_dt_s/int(mura_substeps_per_half),
                mura_transport_operator=mura_transport_operator)
            pre_stage = stage_diagnostics(state_after_pre, context, driving)
            state_after_front, front = front_stage(
                context, state_after_pre, macro_dt_s, driving,
                proposal_fraction, front_exp_n, front_direction)
            front_stage_state = stage_diagnostics(
                state_after_front, context, driving)
            partial_metadata = {
                "source_sha": frozen_source_sha, "stage": "POST_MURA_PENDING",
                "grid": grid, "macro_dt_s": macro_dt_s,
                "interval_index": index, "completed_intervals": index,
                "physical_time_s": physical_time,
                "pre_mura_elapsed_s": pre_elapsed,
                "pre_mura_audits": pre,
                "front_exposure_s": macro_dt_s,
                "front_direction": front_direction,
                "front_metrics": front_metrics(front), "records": records,
                "initial_stage_diagnostics": initial_stage,
                "pre_mura_stage_diagnostics": pre_stage,
                "post_front_stage_diagnostics": front_stage_state,
                "front_must_not_repeat_on_restart": True,
            }
            partial_path = output_dir/f"partial_{index+1:06d}_post_front.npz"
            save_stage(partial_path, state_after_front, context, partial_metadata)
            if inject_post_front_failure:
                terminal = {
                    "schema": SCHEMA, "status": "NUMERICAL_INTEGRATOR_TERMINAL",
                    "accepted_complete_macros": index,
                    "physical_time_s": physical_time,
                    "typed_partial_checkpoint": str(partial_path.resolve()),
                    "partial_stage": "POST_MURA_PENDING",
                }
                atomic_json(output_dir/"terminal.json", terminal)
                raise RuntimeError("injected post-front failure")
        else:
            state_after_front = state
            pre = list(pending.get("pre_mura_audits", []))
            pre_elapsed = float(pending["pre_mura_elapsed_s"])
            front = None
            partial_metadata = pending
            initial_stage = pending.get("initial_stage_diagnostics")
            pre_stage = pending.get("pre_mura_stage_diagnostics")
            front_stage_state = pending.get("post_front_stage_diagnostics")
            pending = None
        try:
            state_after_post, post, post_elapsed = mura_to_time(
                context, state_after_front, 0.5*macro_dt_s, driving,
                maximum_substep_s=.5*macro_dt_s/int(mura_substeps_per_half),
                mura_transport_operator=mura_transport_operator)
        except Exception as error:
            # The completed macro remains macro_start.  The durable typed
            # partial owns the accepted pre/front history and exact next stage.
            atomic_json(output_dir/"terminal.json", {
                "schema": SCHEMA, "status": "NUMERICAL_INTEGRATOR_TERMINAL",
                "accepted_complete_macros": index,
                "physical_time_s": physical_time,
                "typed_partial_checkpoint": str(partial_path.resolve()),
                "partial_stage": "POST_MURA_PENDING",
                "error_type": type(error).__name__, "error": str(error),
            })
            state = macro_start
            raise
        state = state_after_post
        post_stage = stage_diagnostics(state, context, driving)
        physical_time += macro_dt_s
        metrics = (partial_metadata["front_metrics"] if front is None
                   else front_metrics(front))
        row = {
            "interval": index, "physical_time_end_s": physical_time,
            "macro_dt_s": macro_dt_s,
            "mura_pre_elapsed_s": pre_elapsed,
            "mura_post_elapsed_s": post_elapsed,
            "mura_operator_exposure_s": pre_elapsed+post_elapsed,
            "front_operator_exposure_s": macro_dt_s,
            "external_clock_increment_s": macro_dt_s,
            "ordering_pre": pre[-1] if pre else None,
            "ordering_post": post[-1] if post else None,
            "state_metrics": state_metrics(state, context, driving),
            "mura_plastic_work_increment_J_m3_cells": float(sum(
                item.get("plastic_work_increment_J_m3_cells", 0.0)
                for item in pre+post)),
            "mura_deposited_heat_increment_J_m3_cells": float(sum(
                item.get("deposited_heat_increment_J_m3_cells", 0.0)
                for item in pre+post)),
            "mura_stored_line_energy_increment_J_m3_cells": float(sum(
                item.get("stored_line_energy_increment_J_m3_cells", 0.0)
                for item in pre+post)),
            **metrics,
            "temperature_range_K": [float(np.min(
                state.mechanical.common.temperature_K)), float(np.max(
                state.mechanical.common.temperature_K))],
            "common_clock_closed": bool(np.isclose(
                pre_elapsed+post_elapsed, macro_dt_s, rtol=0.0,
                atol=128*np.finfo(float).eps*macro_dt_s)),
            "resumed_without_repeating_front": front is None,
            "stage_diagnostics": {
                "macro_initial": initial_stage,
                "after_first_mura_half_stage": pre_stage,
                "after_front_transaction": front_stage_state,
                "after_second_mura_half_stage": post_stage,
            },
        }
        records.append(row)
        save_stage(output_dir/f"checkpoint_{index+1:06d}.npz", state, context, {
            "source_sha": frozen_source_sha, "stage": "MACRO_COMPLETE",
            "grid": grid, "macro_dt_s": macro_dt_s,
            "completed_intervals": index+1, "physical_time_s": physical_time,
            "records": records,
        })
    result = {
        "schema": SCHEMA, "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": frozen_source_sha, "status": "HORIZON_COMPLETE",
        "grid": grid, "macro_dt_s": macro_dt_s,
        "mura_transport_operator": mura_transport_operator,
        "front_direction": front_direction,
        "macro_dt_values_s": sorted({float(
            record["macro_dt_s"]) for record in records}),
        "completed_intervals": intervals, "physical_time_s": physical_time,
        "protocol": "predeformed_hold_zero_applied_front_work",
        "composition": "A(H/2)-B(H)-A(H/2); external clock H",
        "all_common_clocks_closed": all(r["common_clock_closed"] for r in records),
        "accepted_front_intervals": sum(r["front_published"] for r in records),
        "cumulative_contour_displacement_m": float(sum(
            r["front_contour_displacement_m"] for r in records)),
        "wall_seconds": time.perf_counter()-wall_start,
        "records": records, "drx_claimed": False, "strict_asb_claimed": False,
    }
    atomic_json(output_dir/"result.json", result)
    result["result_sha256"] = hashlib.sha256(
        (output_dir/"result.json").read_bytes()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--macro-dt-s", type=float, default=1e-3)
    parser.add_argument("--intervals", type=int, default=1)
    parser.add_argument("--proposal-fraction", type=float, default=.0625)
    parser.add_argument("--front-exp-n", type=float, default=1.0)
    parser.add_argument("--front-direction", type=int, choices=(-1, 1),
                        default=1)
    parser.add_argument("--restart", type=Path)
    parser.add_argument("--inject-post-front-failure", action="store_true")
    parser.add_argument("--mura-substeps-per-half", type=int, default=1)
    parser.add_argument("--mura-transport-operator",
                        choices=("legacy_mixed", "compatible_dealiased"),
                        default="legacy_mixed")
    args = parser.parse_args()
    result = run_case(**vars(args))
    print(json.dumps({key: result[key] for key in (
        "status", "grid", "macro_dt_s", "physical_time_s",
        "all_common_clocks_closed", "accepted_front_intervals",
        "cumulative_contour_displacement_m", "wall_seconds")},
        indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
