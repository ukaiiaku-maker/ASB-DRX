#!/usr/bin/env python3
"""Compact decision figures for V42 topology, gradient, and front branches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verification-root", type=Path, required=True)
    args = parser.parse_args(); root = args.verification_root
    failure = json.loads((root/"v42_topology_first_failure.json").read_text())
    repair = json.loads((root/"v42_topology_repair_decision.json").read_text())
    gradient = json.loads((root/"v42_matched_gradient_refinement.json").read_text())
    front = json.loads((root/"v42_front_physical_alternative.json").read_text())

    off = failure["legacy_control_metrics"]
    old = failure["legacy_topology_on_metrics"]
    new = repair["repaired_metrics"]
    labels = ("control", "legacy on", "repaired on")
    metrics = (
        ("ordered_fraction_wall_local", "wall-local ordered fraction"),
        ("authoritative_source_offset_relative_rms", "source-offset residual"),
        ("normalized_line_continuity_residual", "line-continuity residual"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
    for axis, (key, title) in zip(axes, metrics):
        values = [off[key], old[key], new[key]]
        axis.bar(labels, np.maximum(values, 1e-16), color=("#4c78a8", "#e45756", "#54a24b"))
        axis.set_yscale("log"); axis.set_title(title); axis.tick_params(axis="x", rotation=25)
    fig.savefig(root/"v42_topology_repair.png", dpi=180); plt.close(fig)

    rel = gradient["relative_differences"]
    names = ("Nye RMS", "Nye maximum", "Nye L1", "ordered line")
    values = [rel[key] for key in (
        "curl_nye_rms_m1", "curl_nye_maximum_tensor_norm_m1",
        "curl_nye_l1_tensor_integral_m", "ordered_line_m_per_m")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    axes[0].bar(names, 100*np.asarray(values), color="#f58518")
    axes[0].axhline(5, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("n128/n192 relative difference (%)")
    axes[0].tick_params(axis="x", rotation=25)
    for case, color in zip(front["cases"], ("#4c78a8", "#54a24b")):
        smallest = [row for row in case["trials"]
                    if row["proposal_fraction"] == .0078125]
        axes[1].scatter(
            [row["proposal_direction"] for row in smallest],
            [row["actual_complete_candidate_delta_helmholtz_J"] for row in smallest],
            label=case["name"], color=color)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xlabel("proposal direction"); axes[1].set_ylabel("trial delta Helmholtz (J)")
    axes[1].legend(fontsize=8)
    fig.savefig(root/"v42_gradient_front_decision.png", dpi=180); plt.close(fig)


if __name__ == "__main__":
    main()
