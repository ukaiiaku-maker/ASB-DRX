#!/usr/bin/env python3
"""Eight-versus-sixteen Mura substeps from the same retained late state."""

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
    I3Controls, resolved_bicrystal, run_i3_cycle,
)
from full_model.analysis.run_v36_recurrent_physical_response import driving_at_time
from full_model.analysis.run_v39_common_horizon import load_stage, save_stage
from full_model.production.density_state_map import derived_density_fields
from full_model.production.tensorial_nye import nye_from_plastic_distortion


EXPOSURE_S = 3.90625e-6


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fields(state, context):
    derived = derived_density_fields(state.mechanical.density,
                                     context["topologies"])
    return {
        "curl_nye": nye_from_plastic_distortion(
            state.mechanical.common.beta_p, context["spacing_m"]),
        "ordered_density": derived["rho_wall_ordered_m2"],
        "total_density": derived["rho_total_m2"],
        "plastic_distortion": state.mechanical.common.beta_p,
        "temperature_rise": state.mechanical.common.temperature_K-1100.0,
    }


def coefficients(value, half=63):
    n = value.shape[0]; center = n//2
    spectrum = np.fft.fftshift(
        np.fft.fftn(value, axes=(0, 1))/(n*n), axes=(0, 1))
    return spectrum[center-half:center+half+1,
                    center-half:center+half+1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    context = resolved_bicrystal(
        grid=128, length_m=3.2e-6, interface_width_m=4e-7,
        temperature_K=1100.0, child_line_fraction=.35)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    root, metadata = load_stage(args.checkpoint, context)
    results = {}; outputs = {}
    for count in (8, 16):
        state = root; dt = EXPOSURE_S/count; records = []
        started = time.perf_counter()
        for index in range(count):
            driving_time = float(metadata["physical_time_s"])+(index+.5)*dt
            driving = driving_at_time(128, .01, "hold", 0.0, driving_time)
            state, audit = run_i3_cycle(
                context, state, state.eta.copy(), driving,
                I3Controls(mura_enabled=True, front_enabled=False,
                           trial_dt_s=dt, front_dt_s=dt,
                           mura_transport_operator="compatible_dealiased"))
            ledger = audit["mura"]
            records.append({
                "index": index+1, "requested_dt_s": dt,
                "accepted_dt_s": float(ledger["accepted_dt_s"]),
                "event_scale": float(ledger["event_scale"]),
                "ordering_method": ledger.get("ordering_integration_method"),
                "ordering_finite_time_certified": ledger.get(
                    "ordering_finite_time_kinetic_accuracy_certified_by_this_solve"),
            })
        checkpoint = args.output/f"late_{count}_substeps.npz"
        save_stage(checkpoint, state, context, {
            "source_sha": source, "stage": f"v47_late_{count}_substeps",
            "grid": 128,
            "physical_time_s": float(metadata["physical_time_s"])+EXPOSURE_S,
            "completed_operation_count": count, "records": records})
        results[str(count)] = {
            "requested_substeps": count,
            "accepted_substeps": len(records),
            "accepted_exposure_s": sum(x["accepted_dt_s"] for x in records),
            "wall_seconds": time.perf_counter()-started,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": digest(checkpoint), "records": records,
        }
        outputs[count] = fields(state, context)
    comparisons = {}
    for name in outputs[8]:
        left, right = coefficients(outputs[8][name]), coefficients(outputs[16][name])
        comparisons[name] = float(np.linalg.norm(left-right)/max(
            np.linalg.norm(left), np.linalg.norm(right), 1e-300))
    maximum = max(comparisons.values())
    payload = {
        "schema": "asb-drx/v47/late-temporal-refinement/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": source,
        "parent_checkpoint": str(args.checkpoint.resolve()),
        "parent_checkpoint_sha256": digest(args.checkpoint),
        "parent_physical_time_s": metadata["physical_time_s"],
        "operator_exposure_s": EXPOSURE_S,
        "common_band_half_width": 63,
        "runs": results, "relative_differences": comparisons,
        "classification": ("CURRENT_SOURCE_TEMPORAL_REFINEMENT_PASSED_5PCT"
                           if maximum <= .05 else
                           "CURRENT_SOURCE_TEMPORAL_REFINEMENT_FAILED_5PCT"),
    }
    path = args.output/"v47_late_temporal_refinement.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"classification": payload["classification"],
                      "maximum_relative_difference": maximum,
                      "sha256": digest(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
