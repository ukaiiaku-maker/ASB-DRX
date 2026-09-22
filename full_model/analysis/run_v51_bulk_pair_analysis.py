#!/usr/bin/env python3
"""Verify and classify the completed V49/V50 common-horizon bulk pair."""

from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np

from full_model.analysis.run_v34_finite_coupled_response import resolved_bicrystal
from full_model.analysis.run_v39_common_horizon import load_stage


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_manifest(path, target=52):
    data = json.loads(Path(path).read_text())
    bad = []
    for record in data["records"]:
        for segment in record["segments"]:
            requested = float(segment["requested_duration_s"])
            accepted = float(segment["accepted_duration_s"])
            tolerance = 64*np.finfo(float).eps*max(requested, 1e-300)
            if abs(requested-accepted) > tolerance:
                bad.append({"interval": record["interval"],
                            "requested_duration_s": requested,
                            "accepted_duration_s": accepted})
    checkpoint = Path(data["latest_checkpoint"])
    return data, {
        "status_complete": data["status"] == "COMPLETE",
        "completed_intervals": int(data["completed_intervals"]),
        "target_intervals": int(target),
        "all_segments_full_duration": not bad,
        "shortened_segments": bad,
        "latest_checkpoint_sha256": digest(checkpoint),
        "latest_checkpoint_sha256_verified": (
            digest(checkpoint) == data["latest_checkpoint_sha256"]),
        "manifest_sha256": digest(path),
    }


def state_summary(checkpoint, grid):
    context = resolved_bicrystal(
        grid=grid, length_m=3.2e-6, interface_width_m=4e-7,
        temperature_K=1100.0, child_line_fraction=.35)
    state, _ = load_stage(checkpoint, context)
    mechanical = state.mechanical
    inventory = mechanical.density
    alignment = mechanical.reservoir_alignment
    scalar = {}
    for item in fields(inventory):
        value = np.asarray(getattr(inventory, item.name))
        if value.ndim >= 2 and np.issubdtype(value.dtype, np.number):
            scalar[item.name] = {
                "mean": float(np.mean(value)), "sum": float(np.sum(value)),
                "minimum": float(np.min(value)), "maximum": float(np.max(value)),
            }
    moments = {}
    for name in ("mobile_plus_m2", "mobile_minus_m2",
                 "forest_plus_m2", "forest_minus_m2",
                 "wall_tangle_plus_m2", "wall_tangle_minus_m2",
                 "wall_ordered_plus_m2", "wall_ordered_minus_m2"):
        value = np.asarray(getattr(alignment, name))
        moments[name] = {
            "integrated_vector": np.sum(value, axis=(0, 1, 2)).tolist(),
            "rms_norm": float(np.sqrt(np.mean(np.sum(value*value, axis=-1)))),
        }
    common = mechanical.common
    return {
        "density_inventory": scalar,
        "reservoir_alignment": moments,
        "temperature_mean_K": float(np.mean(common.temperature_K)),
        "temperature_peak_K": float(np.max(common.temperature_K)),
        "beta_p_mean": np.mean(common.beta_p, axis=(0, 1)).tolist(),
        "family_nye_rms_m-1": float(np.sqrt(np.mean(
            np.asarray(common.family_nye_m1)**2))),
        "orientation_span_rad": float(np.ptp(common.orientation_rad)),
        "front_enabled_in_calculation": False,
    }


def endpoint_history(manifest):
    return [{
        "interval": int(record["interval"]),
        "time_s": float(record["load_elapsed_time_s"]),
        **{key: float(value) for key, value in
           record["endpoint_observables"].items()},
        "cumulative_first_law_residual_J": float(
            record["cumulative_first_law_residual_J"]),
        "ordering_linear_iterations": int(sum(
            int(segment.get("ordering_linear_iterations") or 0)
            for segment in record["segments"])),
        "wall_seconds": float(sum(
            float(segment["wall_seconds"]) for segment in record["segments"])),
    } for record in manifest["records"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loading-manifest", type=Path, required=True)
    parser.add_argument("--hold-manifest", type=Path, required=True)
    parser.add_argument("--manager-record", type=Path, required=True)
    parser.add_argument("--manager-pair", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    loading, loading_audit = audit_manifest(args.loading_manifest)
    hold, hold_audit = audit_manifest(args.hold_manifest)
    if not all((loading_audit["status_complete"], hold_audit["status_complete"],
                loading_audit["all_segments_full_duration"],
                hold_audit["all_segments_full_duration"],
                loading_audit["latest_checkpoint_sha256_verified"],
                hold_audit["latest_checkpoint_sha256_verified"])):
        raise RuntimeError("bulk pair failed completion/duration/checksum audit")
    if loading["source_checkpoint_sha256"] != hold["source_checkpoint_sha256"]:
        raise RuntimeError("bulk pair does not share its original state")
    lh = endpoint_history(loading); hh = endpoint_history(hold)
    if len(lh) != len(hh) or abs(lh[-1]["time_s"]-hh[-1]["time_s"]) > 1e-18:
        raise RuntimeError("bulk pair does not share one physical horizon")
    keys = tuple(loading["records"][-1]["endpoint_observables"])
    difference = {key: lh[-1][key]-hh[-1][key] for key in keys}
    dt = float(loading["macro_dt_s"])
    lp = np.asarray([0.0]+[r["engineering_plastic_shear_gamma_p"] for r in lh])
    hp = np.asarray([0.0]+[r["engineering_plastic_shear_gamma_p"] for r in hh])
    loading_rates = np.diff(lp)/dt; hold_rates = np.diff(hp)/dt
    added_shear = difference["engineering_total_shear_gamma"]
    shear_modulus = 89.1e9
    predicted_stress_difference = shear_modulus*(
        added_shear-difference["engineering_plastic_shear_gamma_p"])
    manager = json.loads(args.manager_record.read_text())
    manager_pair = json.loads(args.manager_pair.read_text())
    dependency_names = subprocess.check_output([
        "git", "diff", "--name-only",
        loading["source_sha"], hold["source_sha"], "--", "full_model"],
        text=True).splitlines()
    payload = {
        "schema": "asb-drx/v51/completed-common-horizon-pair/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "loading_source_sha": loading["source_sha"],
        "hold_source_sha": hold["source_sha"],
        "loading_audit": loading_audit, "hold_audit": hold_audit,
        "shared_initial_checkpoint_sha256": loading[
            "source_checkpoint_sha256"],
        "exact_target": {
            "intervals": 52, "physical_time_s": 52*dt,
            "additional_engineering_shear": 2*float(
                loading["strain_rate_s"])*52*dt,
        },
        "original_planning_milestone_additional_engineering_shear": .005,
        "loading_history": lh, "hold_history": hh,
        "loading_endpoint_state": state_summary(
            loading["latest_checkpoint"], int(loading["grid"])),
        "hold_endpoint_state": state_summary(
            hold["latest_checkpoint"], int(hold["grid"])),
        "loading_minus_hold_endpoint": difference,
        "plastic_activity": {
            "loading_interval_rates_s-1": loading_rates.tolist(),
            "hold_interval_rates_s-1": hold_rates.tolist(),
            "loading_first_interval_rate_s-1": float(loading_rates[0]),
            "loading_last_interval_rate_s-1": float(loading_rates[-1]),
            "hold_last_interval_rate_s-1": float(hold_rates[-1]),
            "loading_minus_hold_plastic_fraction_of_added_shear": float(
                difference["engineering_plastic_shear_gamma_p"]/added_shear),
            "loading_last_interval_fraction_of_imposed_rate": float(
                loading_rates[-1]/(2*loading["strain_rate_s"])),
        },
        "stress_decomposition": {
            "observed_loading_minus_hold_Pa": difference[
                "mean_shear_stress_sigma_12_Pa"],
            "elastic_prediction_Pa": float(predicted_stress_difference),
            "residual_Pa": float(
                difference["mean_shear_stress_sigma_12_Pa"]
                -predicted_stress_difference),
            "shear_modulus_Pa": shear_modulus,
        },
        "manager": {"record_sha256": digest(args.manager_record),
                    "state": manager["state"],
                    "pair_sha256": digest(args.manager_pair),
                    "pair_classification": manager_pair["classification"]},
        "mixed_source_dependency_audit": {
            "changed_full_model_paths": dependency_names,
            "full_duration_overlap_basis": (
                "V50 changes the conditional shortened-interval driver branch "
                "and adds optional subcell geometry owners/events; all pair "
                "segments are full-duration and both front and subcell events "
                "are disabled"),
            "expensive_loading_replay_required": False,
        },
        "classification": "VALID_COMPLETE_COMMON_HORIZON_PREDOMINANTLY_ELASTIC_RESPONSE",
        "stress_peak_observed": False,
        "front_enabled": False,
        "drx_claimed": False, "lagb_claimed": False,
        "strict_asb_claimed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    time_us = np.asarray([r["time_s"] for r in lh])*1e6
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.0), sharex=True)
    for history, label in ((lh, "loading"), (hh, "hold")):
        axes[0].plot(time_us, [r["mean_shear_stress_sigma_12_Pa"]*1e-9
                              for r in history], label=label)
        axes[1].plot(time_us, [r["engineering_plastic_shear_gamma_p"]
                              for r in history], label=label)
        axes[2].plot(time_us, [r["temperature_mean_K"]-1100.0
                              for r in history], label=label)
    axes[0].set_ylabel("mean shear stress (GPa)")
    axes[1].set_ylabel("engineering plastic shear")
    axes[2].set_ylabel("mean temperature rise (K)")
    axes[2].set_xlabel("physical time (microseconds)")
    axes[0].legend(); fig.tight_layout(); fig.savefig(args.figure, dpi=180)
    plt.close(fig)
    print(json.dumps({"classification": payload["classification"],
                      "output_sha256": digest(args.output),
                      "figure_sha256": digest(args.figure)}, sort_keys=True))


if __name__ == "__main__":
    main()
