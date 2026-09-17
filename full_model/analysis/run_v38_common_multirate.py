#!/usr/bin/env python3
"""Strang-multirate recurrent front/Mura/thermal common-state trajectory."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from full_model.analysis.run_v34_finite_coupled_response import (
    I3Controls, checkpoint_payload, resolved_bicrystal, run_i3_cycle,
)
from full_model.analysis.run_v36_recurrent_physical_response import (
    driving_at_time, geometric_envelope,
)


def source_sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def mura_to_time(context, state, duration_s, driving):
    elapsed = 0.0; audits = []
    tolerance = 64*np.finfo(float).eps*max(duration_s, 1e-300)
    while elapsed < duration_s-tolerance:
        requested = duration_s-elapsed
        controls = I3Controls(
            mura_enabled=True, front_enabled=False,
            trial_dt_s=requested, front_dt_s=requested)
        state, audit = run_i3_cycle(
            context, state, state.eta.copy(), driving, controls)
        accepted = float(audit["mura"]["accepted_dt_s"])
        if accepted <= tolerance or accepted > requested+tolerance:
            raise RuntimeError("Mura multirate subcycle made invalid clock progress")
        elapsed += accepted; audits.append(audit["mura"])
    return state, audits, elapsed


def run_case(output_dir, *, grid, macro_dt_s, intervals):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    context = resolved_bicrystal(
        grid=grid, length_m=3.2e-6, interface_width_m=4e-7,
        temperature_K=1100.0, child_line_fraction=.35)
    context["extensive_parameters"] = replace(
        context["extensive_parameters"],
        ordering_internal_substep_s=5e-13,
        ordering_internal_max_substeps=8192)
    state = context["state"]
    records = []; physical_time = 0.0
    for index in range(intervals):
        midpoint = physical_time+0.5*macro_dt_s
        driving = driving_at_time(grid, .01, "hold", 0.0, midpoint)
        state, pre, pre_elapsed = mura_to_time(
            context, state, 0.5*macro_dt_s, driving)
        envelope = geometric_envelope(state.eta, .0625, direction=1)
        front_controls = I3Controls(
            mura_enabled=False, front_enabled=True,
            driving_pressure_a_to_b_Pa=0.0,
            applied_pressure_a_to_b_Pa=0.0,
            geometric_probe_pressure_Pa=1e8,
            trial_dt_s=macro_dt_s, front_dt_s=macro_dt_s,
            front_exp_n=1.0)
        state, front = run_i3_cycle(
            context, state, envelope, driving, front_controls)
        state, post, post_elapsed = mura_to_time(
            context, state, 0.5*macro_dt_s, driving)
        physical_time += macro_dt_s
        decision = front["front_decision"]
        interface_area = (None if decision is None else float(
            decision["interface_area_m2"]))
        sweep = float(front["sweep"]["net_m3"])
        records.append({
            "interval": index, "physical_time_end_s": physical_time,
            "macro_dt_s": macro_dt_s,
            "mura_pre_elapsed_s": pre_elapsed,
            "mura_post_elapsed_s": post_elapsed,
            "mura_pre_subcycles": pre, "mura_post_subcycles": post,
            "front_classification": (None if decision is None
                                     else decision["classification"]),
            "front_published": bool(front["candidate_sweep_published"]),
            "front_signed_sweep_m3": sweep,
            "front_contour_displacement_m": (
                0.0 if not interface_area else sweep/interface_area),
            "front_complete_energy_delta_J": float(
                front["complete_energy"]["delta_helmholtz_J"]),
            "front_rate_m_s": (None if decision is None else float(
                decision["net_velocity_a_to_b_m_s"])),
            "temperature_range_K": front["temperature_range_K"],
            "common_clock_closed": bool(np.isclose(
                pre_elapsed+post_elapsed, macro_dt_s, rtol=0.0,
                atol=128*np.finfo(float).eps*macro_dt_s)),
        })
        payload = checkpoint_payload(state, context)
        payload["v38_multirate_metadata_json"] = np.asarray(json.dumps({
            "schema": "asb-drx/v38/common-multirate-checkpoint/v1",
            "source_sha": source_sha(), "grid": grid,
            "macro_dt_s": macro_dt_s, "completed_intervals": index+1,
            "physical_time_s": physical_time, "records": records,
        }, sort_keys=True))
        temporary = output_dir/f"checkpoint_{index+1:06d}.tmp.npz"
        np.savez_compressed(temporary, **payload)
        temporary.replace(output_dir/f"checkpoint_{index+1:06d}.npz")
    result = {
        "schema": "asb-drx/v38/common-multirate-response/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": source_sha(), "grid": grid,
        "macro_dt_s": macro_dt_s, "completed_intervals": intervals,
        "physical_time_s": physical_time,
        "protocol": "predeformed_hold_zero_applied_front_work",
        "splitting": "Mura/thermal half-step; front full-step; Mura/thermal half-step",
        "all_common_clocks_closed": all(row["common_clock_closed"] for row in records),
        "accepted_front_intervals": sum(row["front_published"] for row in records),
        "cumulative_contour_displacement_m": float(sum(
            row["front_contour_displacement_m"] for row in records)),
        "records": records,
        "drx_claimed": False, "strict_asb_claimed": False,
    }
    path = output_dir/"result.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    result["result_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--macro-dt-s", type=float, default=1e-3)
    parser.add_argument("--intervals", type=int, default=1)
    args = parser.parse_args()
    result = run_case(**vars(args))
    print(json.dumps({key: result[key] for key in (
        "grid", "macro_dt_s", "physical_time_s", "all_common_clocks_closed",
        "accepted_front_intervals", "cumulative_contour_displacement_m")},
        indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
