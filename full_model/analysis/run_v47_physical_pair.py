#!/usr/bin/env python3
"""Bounded current-source hold/continued-loading response from one V46 state."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from full_model.analysis.run_v34_finite_coupled_response import (
    I3Controls, resolved_bicrystal, run_i3_cycle,
)
from full_model.analysis.run_v36_recurrent_physical_response import driving_at_time
from full_model.analysis.run_v39_common_horizon import load_stage
from full_model.production.v24_mechanical_wall import resolved_driving_components


DT_S = 4.8828125e-7


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observables(context, state, driving):
    common = state.mechanical.common
    resolved = resolved_driving_components(
        common, driving, context["systems"], context["topologies"],
        context["wall_parameters"])
    return {
        "mean_imposed_shear": float(driving.mean_strain[0, 1]),
        "mean_plastic_shear": float(np.mean(common.beta_p[..., 0, 1])),
        "mean_temperature_K": float(np.mean(common.temperature_K)),
        "peak_temperature_K": float(np.max(common.temperature_K)),
        "rms_raw_resolved_stress_Pa": float(np.sqrt(np.mean(
            resolved["raw_stress_Pa"]**2))),
        "rms_effective_resolved_stress_Pa": float(np.sqrt(np.mean(
            resolved["effective_stress_Pa"]**2))),
        "mean_speed_m_s": float(np.mean(resolved["speed_m_s"])),
        "maximum_speed_m_s": float(np.max(resolved["speed_m_s"])),
    }


def scalar_delta(after, before):
    return {key: float(after[key]-before[key]) for key in before}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strain-rate-s", type=float, default=100.0)
    args = parser.parse_args()
    context = resolved_bicrystal(
        grid=128, length_m=3.2e-6, interface_width_m=4e-7,
        temperature_K=1100.0, child_line_fraction=.35)
    context["extensive_parameters"] = replace(
        context["extensive_parameters"],
        ordering_finite_time_backend="matrix_free_projected_rk2",
        ordering_matrix_free_max_attempt_exposure=.05,
        ordering_asymptotic_minimum_attempt_exposure=1e300)
    initial, metadata = load_stage(args.checkpoint, context)
    cases = {}
    for name, protocol, rate in (
            ("hold", "hold", 0.0),
            ("continued_loading", "continued_deformation", args.strain_rate_s)):
        driving = driving_at_time(128, .01, protocol, rate, .5*DT_S)
        before = observables(context, initial, driving)
        started = time.perf_counter()
        result, audit = run_i3_cycle(
            context, initial, initial.eta.copy(), driving,
            I3Controls(
                mura_enabled=True, front_enabled=False, trial_dt_s=DT_S,
                front_dt_s=DT_S,
                mura_transport_operator="compatible_dealiased"))
        elapsed = time.perf_counter()-started
        after = observables(context, result, driving)
        mura = audit["mura"]
        cases[name] = {
            "protocol": protocol, "strain_rate_s": rate,
            "accepted_interval_s": float(mura["accepted_dt_s"]),
            "wall_seconds": elapsed, "before": before, "after": after,
            "change": scalar_delta(after, before),
            "ordering_integration_method": mura.get(
                "ordering_integration_method"),
            "ordering_stiff_dispatch": mura.get("ordering_stiff_dispatch"),
            "ordering_finite_time_kinetic_accuracy_certified": mura.get(
                "ordering_finite_time_kinetic_accuracy_certified_by_this_solve"),
            "mura_event_scale": float(mura["event_scale"]),
            "plastic_work_J_m3_cells": float(
                mura["plastic_work_increment_J_m3_cells"]),
            "dissipative_drag_and_heat_J_m3_cells": float(
                mura["deposited_heat_increment_J_m3_cells"]),
            "front_channel_disabled_for_bounded_bulk_pair": True,
        }
    differences = {
        key: cases["continued_loading"]["after"][key]
             -cases["hold"]["after"][key]
        for key in cases["hold"]["after"]}
    payload = {
        "schema": "asb-drx/v47/current-source-physical-pair/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": digest(args.checkpoint),
        "checkpoint_physical_time_s": metadata["physical_time_s"],
        "finite_time_ordering_forced": True,
        "comparison": "100/s continued loading versus hold over 0.48828125 us",
        "expected_discriminant": (
            "incremental stress, plastic shear, irreversible heat, and "
            "temperature response at bounded accumulated strain"),
        "cases": cases,
        "continued_minus_hold_after": differences,
        "classification": "VALID_BOUNDED_BULK_RESPONSE_PAIR",
        "existing_boundary_migration_claimed": False,
        "drx_claimed": False,
        "asb_claimed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"classification": payload["classification"],
                      "differences": differences,
                      "sha256": digest(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
