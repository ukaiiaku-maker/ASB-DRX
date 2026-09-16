#!/usr/bin/env python3
"""Plot the force/history and grid-scaling evidence used in the V27 decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SIBM_CASES = ("equal", "favorable", "reversed", "mobility_off", "label_swapped")
COLORS = {
    "equal": "0.35",
    "favorable": "tab:blue",
    "reversed": "tab:red",
    "mobility_off": "tab:purple",
    "label_swapped": "tab:green",
}


def plot_sibm(record, output):
    grid = record["grids"]["128"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    for case in SIBM_CASES:
        row = grid[case]
        time_us = np.asarray(row["time_s"])*1e6
        width = row["interface_width_m"]
        axes[0, 0].plot(time_us, np.asarray(row["material_frame_displacement_m"])/width,
                        marker="o", ms=3, label=case.replace("_", " "), color=COLORS[case])
        axes[0, 1].plot(time_us, np.asarray(row["complete_variational_pressure_Pa"])*1e-6,
                        marker="o", ms=3, color=COLORS[case])
        axes[1, 0].plot(time_us, np.asarray(row["area_velocity_m_s"]),
                        marker="o", ms=3, color=COLORS[case])
    axes[0, 0].axhline(0, color="k", lw=.6)
    axes[0, 0].set(ylabel="material displacement / interface width", xlabel="time (µs)")
    axes[0, 0].legend(fontsize=8, ncol=2)
    axes[0, 1].axhline(0, color="k", lw=.6)
    axes[0, 1].set(ylabel="complete pressure (MPa)", xlabel="time (µs)")
    axes[1, 0].axhline(0, color="k", lw=.6)
    axes[1, 0].set(ylabel="area-equivalent velocity (m/s)", xlabel="time (µs)")

    symmetry = record["near_equal_symmetry"]
    perturbation = []
    velocity = []
    for row in symmetry["records"]:
        if "relative_density_perturbation" not in row:
            continue
        perturbation.append(row["relative_density_perturbation"])
        velocity.append(row["initial_window_velocity_m_s"])
    order = np.argsort(perturbation)
    axes[1, 1].plot(np.asarray(perturbation)[order], np.asarray(velocity)[order], "o-")
    axes[1, 1].axhline(0, color="k", lw=.6)
    axes[1, 1].axvline(0, color="k", lw=.6)
    axes[1, 1].set(xscale="symlog", xlabel="relative defect-density perturbation",
                   ylabel="initial velocity (m/s)")
    fig.suptitle(record["classification"].replace("_", " "))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_asb(record, output):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    colors = {"96": "tab:orange", "128": "tab:blue", "192": "tab:green",
              "homogeneous": "0.35"}
    for name, row in record["records"].items():
        data = pd.read_csv(Path(row["directory"])/"drx_v25_restart_asb_diagnostics.csv")
        axes[0, 0].plot(data["eps_pct"], data["sigma_MPa"], label=name,
                        color=colors[name])
        axes[0, 1].plot(data["t_us"], data["T_max"]-1100.0, color=colors[name])
    axes[0, 0].set(xlabel="strain (%)", ylabel="stress (MPa)")
    axes[0, 0].legend(title="grid", fontsize=8)
    axes[0, 1].set(xlabel="time (µs)", ylabel="maximum temperature excess (K)")

    metrics = ("peak_stress_Pa", "plastic_rate_max_s", "active_plastic_fraction",
               "effective_band_width_m")
    labels = ("peak stress", "max plastic rate", "active fraction", "band width")
    differences = record["relative_differences"]["128_vs_192"]
    axes[1, 0].bar(labels, [100*differences[m] for m in metrics], color="tab:red")
    axes[1, 0].axhline(5, color="k", ls="--", lw=1, label="5% limit")
    axes[1, 0].tick_params(axis="x", rotation=20)
    axes[1, 0].set(ylabel="128/192 relative difference (%)")
    axes[1, 0].legend(fontsize=8)

    first_law = [100*record["records"][name]["first_law"]["system_relative_residual"]
                 for name in ("96", "128", "192", "homogeneous")]
    axes[1, 1].bar(("96", "128", "192", "hom."), first_law, color="tab:blue")
    axes[1, 1].axhline(5, color="k", ls="--", lw=1)
    axes[1, 1].set(ylabel="first-law relative residual (%)", xlabel="case")
    fig.suptitle(record["classification"].replace("_", " "))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sibm", required=True, type=Path)
    parser.add_argument("--asb", required=True, type=Path)
    parser.add_argument("--sibm-output", required=True, type=Path)
    parser.add_argument("--asb-output", required=True, type=Path)
    args = parser.parse_args()
    args.sibm_output.parent.mkdir(parents=True, exist_ok=True)
    args.asb_output.parent.mkdir(parents=True, exist_ok=True)
    plot_sibm(json.loads(args.sibm.read_text()), args.sibm_output)
    plot_asb(json.loads(args.asb.read_text()), args.asb_output)


if __name__ == "__main__":
    main()
