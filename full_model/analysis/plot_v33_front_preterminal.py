#!/usr/bin/env python3
"""Plot the symmetric n128 near-equal phase trials at their terminal step."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--field-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    case_ids = ("a1_n128_delta_p1e-6", "a1_n128_delta_m1e-6")
    figure, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    for row, case_id in enumerate(case_ids):
        with np.load(args.field_root/case_id/"sibm_front_terminal_fields.npz") as data:
            before = data["eta_before"][:, :, 1]-data["eta_before"][:, :, 0]
            trial = data["eta_trial"][:, :, 1]-data["eta_trial"][:, :, 0]
            spacing_um = float(data["spacing_m"])*1e6
        changed = np.signbit(before) != np.signbit(trial)
        extent = (0, before.shape[1]*spacing_um,
                  before.shape[0]*spacing_um, 0)
        image0 = axes[row, 0].imshow(before, vmin=-1, vmax=1, cmap="coolwarm",
                                     extent=extent, interpolation="nearest")
        axes[row, 0].contour(before, levels=[0], colors="k", linewidths=.5,
                             extent=extent)
        axes[row, 0].set_title(f"{case_id}\nbefore")
        axes[row, 1].imshow(trial, vmin=-1, vmax=1, cmap="coolwarm",
                            extent=extent, interpolation="nearest")
        axes[row, 1].contour(trial, levels=[-.2, 0, .2],
                             colors=["navy", "k", "darkred"], linewidths=.6,
                             extent=extent)
        axes[row, 1].set_title("trial: -0.2 / 0 / +0.2 contours")
        delta = trial-before
        axes[row, 2].imshow(delta, vmin=-.04, vmax=.04, cmap="PiYG",
                            extent=extent, interpolation="nearest")
        yy, xx = np.where(changed)
        axes[row, 2].scatter((xx+.5)*spacing_um, (yy+.5)*spacing_um,
                             s=3, c="black", label=f"{len(xx)} sign flips")
        axes[row, 2].legend(fontsize=7)
        axes[row, 2].set_title("trial − before")
        mean_before = np.mean(before, axis=1)
        mean_trial = np.mean(trial, axis=1)
        coordinate = (np.arange(before.shape[0])+.5)*spacing_um
        axes[row, 3].plot(coordinate, mean_before, label="before")
        axes[row, 3].plot(coordinate, mean_trial, label="trial")
        axes[row, 3].axhline(0, color="k", lw=.5)
        if len(yy):
            axes[row, 3].set_xlim(max(0, (yy.min()-5)*spacing_um),
                                  min(10, (yy.max()+6)*spacing_um))
        axes[row, 3].set(title="normal mean profile", xlabel="x (µm)",
                         ylabel="η_child − η_parent")
        axes[row, 3].legend(fontsize=7)
        for axis in axes[row, :3]:
            axis.set(xlabel="y (µm)", ylabel="x (µm)")
    figure.colorbar(image0, ax=axes[:, :2], shrink=.7, label="phase difference")
    figure.suptitle(
        "V33 n128 preterminal audit: zero-level, one-cell filaments without a pure core")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
