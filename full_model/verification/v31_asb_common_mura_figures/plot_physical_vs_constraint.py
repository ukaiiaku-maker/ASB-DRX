#!/usr/bin/env python3
"""Plot the V30 rejection against the V31 common-Mura ASB qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    here = Path(__file__).resolve().parent
    verification = here.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v30", type=Path,
        default=verification/"v30_asb_energy_dissipation_decision.json")
    parser.add_argument(
        "--v31", type=Path,
        default=verification/"v31_asb_common_mura_decision.json")
    parser.add_argument("--output-dir", type=Path, default=here)
    args = parser.parse_args()

    v30 = json.loads(args.v30.read_text())
    v31 = json.loads(args.v31.read_text())
    grids = np.array([32, 64, 128])

    v30_smokes = {int(row["grid"]): row for row in v30["local_smokes"]}
    v31_smokes = {
        int(grid): row for grid, row in v31["production_grid_smokes"].items()
        if grid.isdigit()
    }
    v30_relative = np.array([
        v30_smokes[int(grid)]["relative_first_law_residual"] for grid in grids])
    v31_relative = np.array([
        v31_smokes[int(grid)]["relative_first_law_residual"] for grid in grids])
    v30_first_law = np.array([
        v30_smokes[int(grid)]["first_law_residual_J_m3"] for grid in grids])
    orientation = np.array([
        v30["suboperator_physical_stored_change_J_m3"][str(grid)][
            "orientation_boundary_sources"] for grid in grids])
    phase_front = np.array([
        v30["suboperator_physical_stored_change_J_m3"][str(grid)][
            "phase_front_topology"] for grid in grids])
    constraint = np.array([
        abs(v30["constraint_contamination"][
            "numerical_constraint_change_J_m3"][str(grid)])
        for grid in grids])

    blue, orange, red, gray = "#2468a2", "#e18d2d", "#bd3b32", "#5f6670"
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)

    ax = axes[0]
    width = 0.24
    x = np.arange(len(grids))
    ax.bar(x-width, orientation/1e6, width, color=orange,
           label="V30 orientation/boundary term")
    ax.bar(x, phase_front/1e6, width, color=blue,
           label="V30 phase/front term")
    ax.bar(x+width, v30_first_law/1e6, width, color=red,
           label="V30 first-law residual")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, grids)
    ax.set_xlabel("Grid")
    ax.set_ylabel(r"Physical-scale energy change (MJ m$^{-3}$)")
    ax.set_title("(a) V30 physical-scale imbalance")
    ax.legend(fontsize=8, frameon=False)
    ax.text(
        0.02, 0.03,
        "V31 removes the legacy phase/orientation operators\n"
        "from the common-Mura physical transaction.",
        transform=ax.transAxes, fontsize=8, va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9,
                  edgecolor="#bbbbbb"))

    ax = axes[1]
    ax.semilogy(grids, v30_relative, "o-", color=red, linewidth=2,
                label="V30 rejected")
    ax.semilogy(grids, v31_relative, "s-", color=blue, linewidth=2,
                label="V31 common-Mura")
    ax.axhline(0.05, color=gray, linestyle="--", linewidth=1.3,
               label="5% hard gate")
    ax.set_xticks(grids)
    ax.set_xlabel("Grid")
    ax.set_ylabel("Relative cumulative first-law residual")
    ax.set_title("(b) Physical ledger closure")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[2]
    bars = ax.bar(grids.astype(str), constraint, color=orange, width=0.62)
    ax.set_yscale("log")
    ax.set_ylabel(r"$|\Delta C_{\rm numerical}|$ (J m$^{-3}$)")
    ax.set_xlabel("Grid")
    ax.set_title("(c) Numerical-constraint scale (V30)")
    ax.bar_label(
        bars, labels=[f"{value:.2e}" for value in constraint],
        padding=3, fontsize=8, rotation=90)
    ax.set_ylim(3e21, 3e23)
    ax.text(
        0.5, 0.07,
        "V31: numerical constraint excluded from physical energy\n"
        "and does not drive lattice orientation (declared exactly zero)",
        ha="center", va="bottom", transform=ax.transAxes, fontsize=8,
        color=blue,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.95,
                  edgecolor=blue))

    fig.suptitle(
        "ASB energy audit: V30 numerical-contamination rejection versus "
        "V31 common-Mura qualification",
        fontsize=13)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir/"v31_asb_physical_vs_constraint_energy"
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)

    plotted = {
        "schema": "asb-drx/v31/asb-physical-vs-constraint-figure-data/v1",
        "source_files": {"v30": str(args.v30), "v31": str(args.v31)},
        "grids": grids.tolist(),
        "v30": {
            "relative_first_law_residual": v30_relative.tolist(),
            "first_law_residual_J_m3": v30_first_law.tolist(),
            "orientation_boundary_physical_change_J_m3": orientation.tolist(),
            "phase_front_physical_change_J_m3": phase_front.tolist(),
            "absolute_numerical_constraint_change_J_m3": constraint.tolist(),
        },
        "v31": {
            "relative_first_law_residual": v31_relative.tolist(),
            "numerical_constraint_in_physical_energy": False,
            "numerical_constraint_drives_physical_state": False,
        },
        "scale_separation": "physical and numerical-constraint values use separate panels and axes"
    }
    (args.output_dir/"v31_asb_physical_vs_constraint_energy_data.json").write_text(
        json.dumps(plotted, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
