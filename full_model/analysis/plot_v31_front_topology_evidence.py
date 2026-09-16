#!/usr/bin/env python3
"""Plot V31 component topology and failed-V30 continuation evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = ROOT/"full_model"/"production"
if str(PRODUCTION) not in sys.path:
    sys.path.insert(0, str(PRODUCTION))

from front_topology import (  # noqa: E402
    cut_cell_receiver_fraction, initialize_front_topology)


ARCHIVE = Path(
    "/Users/sdillon/HPC3/hpc3-results/asb-drx-full-v34-recovery/"
    "v31-v30-closeout-20260916/v30_front")
PREFLIGHT = Path(
    "/Users/sdillon/HPC3/local-results/"
    "asb-drx-v31-front-preflight-20260916")


def _load(path):
    with np.load(path, allow_pickle=True) as data:
        return {
            "eta": np.asarray(data["eta"], dtype=float),
            "active": np.asarray(data["sibm_active_mask"], dtype=bool),
            "coupled": json.loads(str(data[
                "coupled_front_metadata_json"].item())),
            "experiment": json.loads(str(data["sibm_experiment_json"].item())),
            "step": int(data["step"]),
        }


def _component_overlay(axis, phi, active, components, title):
    image = axis.imshow(phi.T, origin="lower", cmap="coolwarm", vmin=-1, vmax=1)
    axis.contour(active.T.astype(float), levels=[.5], colors="0.2",
                 linewidths=.8, linestyles="--")
    axis.contour(phi.T, levels=[0], colors="white", linewidths=1.4)
    colors = plt.get_cmap("tab10")
    for index, component in enumerate(components):
        points = np.asarray(component["points_grid"])
        if len(points):
            axis.scatter(points[:, 0], points[:, 1], s=5,
                         color=colors(index), label=f"component {component['component_id']}")
        centroid = np.asarray(component["centroid_grid"])
        normal = np.asarray(component["receiver_normal"])
        axis.plot(*centroid, marker="x", ms=7, mew=1.5, color="yellow")
        axis.arrow(*centroid, *(3*normal), width=.12, color="yellow",
                   length_includes_head=True)
    axis.set_title(title)
    axis.set_xlabel("grid x")
    axis.set_ylabel("grid y")
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--preflight", type=Path, default=PREFLIGHT)
    parser.add_argument("--decision", type=Path, default=ROOT/"full_model"/
                        "verification"/"v31_front_continuation_preflight.json")
    parser.add_argument("--output", type=Path, default=ROOT/"full_model"/
                        "verification"/"v31_front_topology_figures"/
                        "component_topology_and_preflight.png")
    args = parser.parse_args()
    frozen_path = (args.archive/"a1_n64_favorable"/"segments"/"segment-00"/
                   "attempt-000"/"v30_front_restart_000000.npz")
    replay_path = (args.preflight/"a1_n64_favorable"/"attempt-000"/
                   "v31_front_restart_000012.npz")
    frozen = _load(frozen_path)
    replay = _load(replay_path)
    decision = json.loads(args.decision.read_text())
    phi0 = frozen["eta"][:, :, 1]-frozen["eta"][:, :, 0]
    phi1 = replay["eta"][:, :, 1]-replay["eta"][:, :, 0]
    initial = initialize_front_topology(
        phi0, active_mask=frozen["active"], periodic=True)
    initial_components = [
        {**component.__dict__} for component in initial.components]
    final_components = replay["coupled"]["topology"]["components"]
    sweep = (cut_cell_receiver_fraction(
        phi1, active_mask=replay["active"], periodic=True)
        -cut_cell_receiver_fraction(
            phi0, active_mask=frozen["active"], periodic=True))

    records = sorted(decision["records"], key=lambda row: (
        row["grid"], row["comparison"]))
    labels = [row["case"].replace("a1_", "").replace("_", "\n", 2)
              for row in records]
    signed = np.asarray([row["signed_volume_m3"] for row in records])*1e22
    accepted = np.asarray([row["accepted"] for row in records])
    rejected = np.asarray([row["rejected_direction"] for row in records])
    closure = np.asarray([row["maximum_abs_line_closure_m"] for row in records])

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    _component_overlay(
        axes[0, 0], phi0, frozen["active"], initial_components,
        "Frozen V30 checkpoint (64², step 0)\ncomponent inferred from pair-local $\\phi$")
    _component_overlay(
        axes[0, 1], phi1, replay["active"], final_components,
        "V31 topology replay (step 12)\nID 0 persists; no topology event")
    limit = max(float(np.max(np.abs(sweep))), 1e-12)
    im = axes[0, 2].imshow(
        sweep.T, origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit)
    axes[0, 2].contour(phi0.T, levels=[0], colors="black", linewidths=.8)
    axes[0, 2].contour(phi1.T, levels=[0], colors="lime", linewidths=.8)
    axes[0, 2].set_title(
        "Conservative cut-cell sweep\n"
        f"$\\Sigma\\Delta f_B$={np.sum(sweep):.3f} cell²")
    axes[0, 2].set_xlabel("grid x"); axes[0, 2].set_ylabel("grid y")
    fig.colorbar(im, ax=axes[0, 2], label="$\\Delta f_B$ per cell")

    x = np.arange(len(records))
    colors = np.where(signed >= 0, "#2878b5", "#d9534f")
    axes[1, 0].bar(x, signed, color=colors)
    axes[1, 0].axhline(0, color="black", lw=.7)
    axes[1, 0].set_xticks(x, labels, rotation=45, ha="right", fontsize=7)
    axes[1, 0].set_ylabel("cumulative signed sweep ($10^{-22}$ m³)")
    axes[1, 0].set_title("Correct favorable/reversed and ±$10^{-6}$ signs")

    axes[1, 1].bar(x, accepted, label="accepted", color="#2ca02c")
    axes[1, 1].bar(x, rejected, bottom=accepted, label="rate rejected",
                   color="#ffbf00")
    axes[1, 1].set_xticks(x, labels, rotation=45, ha="right", fontsize=7)
    axes[1, 1].set_ylabel("cumulative front attempts")
    axes[1, 1].set_title(
        "Atomic accept/reject decisions\n1 component / 0 events in every case")
    axes[1, 1].legend(frameon=False)

    axes[1, 2].semilogy(x, np.maximum(closure, 1e-30), "o-", color="#6f42c1")
    axes[1, 2].axhline(1e-15, color="black", ls="--", lw=.8,
                       label="hard limit")
    axes[1, 2].set_xticks(x, labels, rotation=45, ha="right", fontsize=7)
    axes[1, 2].set_ylabel("maximum line closure (m)")
    axes[1, 2].set_title(
        "All front ledgers close\n"
        f"maximum {np.max(closure):.2e} m vs $10^{{-15}}$ m limit")
    axes[1, 2].legend(frameon=False)

    fig.suptitle(
        "V31 topology-aware SIBM front: component identity replaces ray-count authority\n"
        "Frozen V30 source 675d741 · topology source 6d2a025", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
