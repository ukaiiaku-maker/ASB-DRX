#!/usr/bin/env python3
"""Transactional V39 recurrent Mura/front/thermal physical-time driver."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from full_model.analysis.run_v34_finite_coupled_response import (
    I3Controls, checkpoint_payload, resolved_bicrystal, run_i3_cycle,
    state_from_payload,
)
from full_model.analysis.run_v36_recurrent_physical_response import (
    driving_at_time, geometric_envelope,
)
from full_model.production.density_state_map import derived_density_fields
from full_model.production.v24_mechanical_wall import resolved_driving_components


SCHEMA = "asb-drx/v39/transactional-common-horizon/v1"


def source_sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


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


def mura_to_time(context, state, duration_s, driving):
    elapsed = 0.0; audits = []
    tolerance = 64*np.finfo(float).eps*max(duration_s, 1e-300)
    while elapsed < duration_s-tolerance:
        requested = duration_s-elapsed
        state, audit = run_i3_cycle(
            context, state, state.eta.copy(), driving,
            I3Controls(mura_enabled=True, front_enabled=False,
                       trial_dt_s=requested, front_dt_s=requested))
        accepted = float(audit["mura"]["accepted_dt_s"])
        if accepted <= tolerance or accepted > requested+tolerance:
            raise RuntimeError("Mura multirate subcycle made invalid clock progress")
        elapsed += accepted; audits.append(audit["mura"])
    return state, audits, elapsed


def front_stage(context, state, duration_s, driving, proposal_fraction,
                front_exp_n):
    envelope = geometric_envelope(state.eta, proposal_fraction, direction=1)
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
    interface_area = (None if decision is None else float(
        decision["interface_area_m2"]))
    sweep = float(audit["sweep"]["net_m3"])
    return {
        "front_classification": (None if decision is None
                                 else decision["classification"]),
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


def run_case(output_dir, *, grid=16, macro_dt_s=1e-3, intervals=1,
             proposal_fraction=.0625, front_exp_n=1.0, restart=None,
             inject_post_front_failure=False):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    context = resolved_bicrystal(
        grid=grid, length_m=3.2e-6, interface_width_m=4e-7,
        temperature_K=1100.0, child_line_fraction=.35)
    records = []; completed = 0; physical_time = 0.0
    pending = None
    partial_path = None
    if restart is not None:
        state, metadata = load_stage(restart, context)
        if (int(metadata["grid"]) != grid
                or float(metadata["macro_dt_s"]) != macro_dt_s):
            raise ValueError("restart configuration differs from requested case")
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
            state_after_pre, pre, pre_elapsed = mura_to_time(
                context, macro_start, 0.5*macro_dt_s, driving)
            state_after_front, front = front_stage(
                context, state_after_pre, macro_dt_s, driving,
                proposal_fraction, front_exp_n)
            partial_metadata = {
                "source_sha": source_sha(), "stage": "POST_MURA_PENDING",
                "grid": grid, "macro_dt_s": macro_dt_s,
                "interval_index": index, "completed_intervals": index,
                "physical_time_s": physical_time,
                "pre_mura_elapsed_s": pre_elapsed,
                "pre_mura_audits": pre,
                "front_exposure_s": macro_dt_s,
                "front_metrics": front_metrics(front), "records": records,
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
            pending = None
        try:
            state_after_post, post, post_elapsed = mura_to_time(
                context, state_after_front, 0.5*macro_dt_s, driving)
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
        }
        records.append(row)
        save_stage(output_dir/f"checkpoint_{index+1:06d}.npz", state, context, {
            "source_sha": source_sha(), "stage": "MACRO_COMPLETE",
            "grid": grid, "macro_dt_s": macro_dt_s,
            "completed_intervals": index+1, "physical_time_s": physical_time,
            "records": records,
        })
    result = {
        "schema": SCHEMA, "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": source_sha(), "status": "HORIZON_COMPLETE",
        "grid": grid, "macro_dt_s": macro_dt_s,
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
    parser.add_argument("--restart", type=Path)
    parser.add_argument("--inject-post-front-failure", action="store_true")
    args = parser.parse_args()
    result = run_case(**vars(args))
    print(json.dumps({key: result[key] for key in (
        "status", "grid", "macro_dt_s", "physical_time_s",
        "all_common_clocks_closed", "accepted_front_intervals",
        "cumulative_contour_displacement_m", "wall_seconds")},
        indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
