#!/usr/bin/env python3
"""Classify and plot the V48 endpoint-loading/hold continuation pair."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def series(manifest, key):
    return np.asarray([row["endpoint_observables"][key]
                       for row in manifest["records"]], dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loading", type=Path, required=True)
    parser.add_argument("--hold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    loading = json.loads(args.loading.read_text())
    hold = json.loads(args.hold.read_text())
    for value in (loading, hold):
        if value["status"] != "COMPLETE":
            raise ValueError("physical continuation is incomplete")
        if value["completed_intervals"] != value["requested_intervals"]:
            raise ValueError("physical continuation interval count mismatch")
    time_us = np.asarray([row["physical_time_end_s"]
                          for row in loading["records"]])*1e6
    keys = (
        "mean_shear_stress_sigma_12_Pa",
        "engineering_plastic_shear_gamma_p",
        "plastic_tensor_strain_e_p12",
        "temperature_mean_K", "temperature_peak_K")
    final_difference = {key: float(series(loading, key)[-1]
                                   -series(hold, key)[-1]) for key in keys}
    maximum_first_law = max(
        row["first_law_relative_residual"]
        for value in (loading, hold) for row in value["records"])
    local_control = all(
        row["ordering_local_error_control_passed"]
        for value in (loading, hold) for row in value["records"])
    start = loading["records"][0]["initial_observables"]
    end = loading["records"][-1]["endpoint_observables"]
    record = {
        "schema": "asb-drx/v48/physical-pair/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "postprocessor_source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "simulation_source_sha": loading["source_sha"],
        "loading_manifest": str(args.loading.resolve()),
        "loading_manifest_sha256": digest(args.loading),
        "hold_manifest": str(args.hold.resolve()),
        "hold_manifest_sha256": digest(args.hold),
        "grid": loading["grid"],
        "achieved_physical_time_s": loading["physical_time_s"],
        "accepted_intervals": loading["completed_intervals"],
        "loading_tensor_shear_rate_s-1": loading["strain_rate_s"],
        "loading_engineering_shear_rate_s-1": 2*loading["strain_rate_s"],
        "additional_tensor_shear_e12": (
            end["total_strain_tensor_e12"]-start["total_strain_tensor_e12"]),
        "additional_engineering_shear_gamma": (
            end["engineering_total_shear_gamma"]
            -start["engineering_total_shear_gamma"]),
        "final_loading_minus_hold": final_difference,
        "maximum_first_law_relative_residual": maximum_first_law,
        "all_ordering_local_error_controls_passed": local_control,
        "ordering_global_endpoint_qualified_by_n128_tightening": False,
        "classification": "CONTROLLED_PHYSICAL_RESPONSE_N16_SHORT_HORIZON",
        "scope": (
            "Eight current-source intervals with exact endpoint load/work "
            "clock; numerical capability evidence, not spatial refinement or "
            "an original-horizon DRX/ASB claim"),
        "drx_claimed": False,
        "lagb_claimed": False,
        "strict_asb_claimed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True)+"\n")

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), constrained_layout=True)
    for value, label in ((loading, "100 s$^{-1}$ tensor shear"),
                         (hold, "fixed-strain hold")):
        axes[0, 0].plot(time_us, series(value,
            "mean_shear_stress_sigma_12_Pa")/1e6, marker="o", label=label)
        axes[0, 1].plot(time_us, series(value,
            "engineering_plastic_shear_gamma_p"), marker="o", label=label)
        axes[1, 0].plot(time_us, series(value,
            "temperature_mean_K")-1100.0, marker="o", label=label)
        axes[1, 1].semilogy(time_us, [max(row[
            "first_law_relative_residual"], 1e-20) for row in value["records"]],
            marker="o", label=label)
    axes[0, 0].set_ylabel("mean $\\sigma_{12}$ (MPa)")
    axes[0, 1].set_ylabel("engineering plastic shear $\\gamma_p$")
    axes[1, 0].set_ylabel("mean temperature rise (K)")
    axes[1, 1].set_ylabel("relative first-law residual")
    for axis in axes.flat:
        axis.set_xlabel("physical time (µs)")
        axis.grid(alpha=.25)
    axes[0, 0].legend(frameon=False)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=180)
    plt.close(fig)
    print(json.dumps({
        "classification": record["classification"],
        "maximum_first_law_relative_residual": maximum_first_law,
        "output_sha256": digest(args.output),
        "figure_sha256": digest(args.figure),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
