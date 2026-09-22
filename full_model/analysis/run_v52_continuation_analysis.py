#!/usr/bin/env python3
"""Postprocess the V52 continued-loading trajectory in physical units."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    records = manifest["records"]
    time_s = np.asarray([row["physical_time_end_s"] for row in records])
    obs = [row["endpoint_observables"] for row in records]
    stress = np.asarray([row["mean_shear_stress_sigma_12_Pa"] for row in obs])
    plastic = np.asarray([row["engineering_plastic_shear_gamma_p"] for row in obs])
    total = np.asarray([row["engineering_total_shear_gamma"] for row in obs])
    mean_temperature = np.asarray([row["temperature_mean_K"] for row in obs])
    peak_temperature = np.asarray([row["temperature_peak_K"] for row in obs])
    dt = np.diff(np.r_[0.0, time_s])
    plastic_rate = np.diff(np.r_[0.0, plastic])/dt
    # The initial plastic state is exactly zero in the constructed benchmark.
    imposed_rate = 200.0
    flow_mask = plastic_rate >= .8*imposed_rate
    flow_windows = np.convolve(flow_mask.astype(int), np.ones(5, dtype=int),
                               mode="valid") if len(flow_mask) >= 5 else []
    flow_approach = bool(np.any(np.asarray(flow_windows) == 5))
    ds = np.diff(stress)
    uncertainty_pa = 15391.50089263916
    peak_index = int(np.argmax(stress))
    peak_resolved = bool(
        peak_index < len(stress)-3
        and np.all(ds[peak_index:peak_index+3] < -uncertainty_pa)
        and stress[peak_index]-stress[-1] > uncertainty_pa)
    new = [row for row in records if row["interval"] > 52]
    wall = [sum(segment["wall_seconds"] for segment in row["segments"])
            for row in new]
    residual = np.asarray([
        row["cumulative_first_law_residual_J"] for row in records])
    payload = {
        "schema": "asb-drx/v52/physical-continuation-analysis/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": digest(args.manifest),
        "source_sha": manifest["source_sha"],
        "status": manifest["status"],
        "completed_intervals": manifest["completed_intervals"],
        "physical_time_s": manifest["physical_time_s"],
        "total_engineering_shear": float(total[-1]),
        "additional_engineering_shear": float(total[-1]-.02),
        "endpoint": {
            **obs[-1],
            "temperature_rise_K": float(mean_temperature[-1]-1100.0),
            "peak_minus_mean_temperature_K": float(
                peak_temperature[-1]-mean_temperature[-1]),
            "plastic_fraction_of_total_added_shear": float(
                plastic[-1]/max(total[-1]-.02, 1e-300)),
            "last_interval_plastic_rate_s-1": float(plastic_rate[-1]),
            "last_interval_fraction_of_imposed_rate": float(
                plastic_rate[-1]/imposed_rate),
            "stress_tangent_from_rate_fraction_Pa": float(
                89.1e9*(1.0-plastic_rate[-1]/imposed_rate)),
        },
        "flow_approach_preregistered_control": {
            "criterion": (
                "plastic rate >=80% of imposed engineering rate for five "
                "accepted intervals"),
            "universal_material_criterion": False,
            "passed": flow_approach,
            "maximum_consecutive_qualifying_intervals": int(max(
                [0]+[len(list(group)) for value, group in __import__(
                    "itertools").groupby(flow_mask) if value])),
        },
        "stress_peak": {
            "observed_maximum_interval": int(records[peak_index]["interval"]),
            "observed_maximum_stress_Pa": float(stress[peak_index]),
            "endpoint_is_maximum": bool(peak_index == len(stress)-1),
            "resolution_scale_Pa": uncertainty_pa,
            "resolved_peak": peak_resolved,
        },
        "new_source_performance": {
            "new_intervals": len(new),
            "median_wall_seconds": float(np.median(wall)) if wall else None,
            "maximum_wall_seconds": float(np.max(wall)) if wall else None,
            "minimum_wall_seconds": float(np.min(wall)) if wall else None,
        },
        "energy_balance": {
            "endpoint_cumulative_first_law_residual_J": float(residual[-1]),
            "maximum_absolute_cumulative_first_law_residual_J": float(
                np.max(np.abs(residual))),
            "all_available_segment_incremental_checks_passed": bool(all(
                segment.get("first_law_passed", True) for row in records
                for segment in row["segments"])),
            "segments_with_explicit_incremental_check": int(sum(
                "first_law_passed" in segment for row in records
                for segment in row["segments"])),
        },
        "physical_interpretation_limits": {
            "front_enabled": False,
            "drx_claimed": False,
            "lagb_claimed": False,
            "strict_asb_claimed": False,
            "thermal_causality_claimed": False,
        },
        "history": [{
            "interval": int(row["interval"]),
            "time_s": float(time_s[index]),
            "stress_Pa": float(stress[index]),
            "total_engineering_shear": float(total[index]),
            "plastic_engineering_shear": float(plastic[index]),
            "plastic_rate_s-1": float(plastic_rate[index]),
            "temperature_mean_K": float(mean_temperature[index]),
            "temperature_peak_K": float(peak_temperature[index]),
            "cumulative_first_law_residual_J": float(residual[index]),
        } for index, row in enumerate(records)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    x = (total-.02)*100.0
    axes[0, 0].plot(x, stress/1e9); axes[0, 0].set_ylabel("mean shear stress (GPa)")
    axes[0, 1].plot(x, plastic_rate); axes[0, 1].axhline(160, ls="--", color="k")
    axes[0, 1].set_ylabel("engineering plastic rate (s$^{-1}$)")
    axes[1, 0].plot(x, mean_temperature-1100.0, label="mean")
    axes[1, 0].plot(x, peak_temperature-1100.0, label="peak")
    axes[1, 0].set_ylabel("temperature rise (K)"); axes[1, 0].legend()
    axes[1, 1].plot(x, plastic/np.maximum(total-.02, 1e-300)*100)
    axes[1, 1].set_ylabel("plastic / added shear (%)")
    for axis in axes.flat: axis.set_xlabel("added engineering shear (%)")
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=180); plt.close(fig)
    print(json.dumps({
        "output_sha256": digest(args.output), "plot_sha256": digest(args.plot),
        "completed_intervals": payload["completed_intervals"],
        "last_plastic_rate_s-1": payload["endpoint"][
            "last_interval_plastic_rate_s-1"],
        "flow_approach": flow_approach, "resolved_peak": peak_resolved,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
