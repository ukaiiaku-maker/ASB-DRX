#!/usr/bin/env python3
"""Plot V41 directional-energy and declared-Nye decision evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from full_model.analysis.run_v34_finite_coupled_response import resolved_bicrystal
from full_model.analysis.run_v39_common_horizon import load_stage
from full_model.production.common_front_state import reconstruct_common
from full_model.production.tensorial_nye import nye_from_plastic_distortion
from full_model.production.wall_topology_supply import reservoir_nye_m1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--front", type=Path, required=True)
    parser.add_argument("--post-front", type=Path, required=True)
    parser.add_argument("--front-figure", type=Path, required=True)
    parser.add_argument("--gradient-figure", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.front.read_text())
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharex="col")
    for row_index, state in enumerate(raw["states"]):
        rows = state["trials"]
        for direction, marker, label in ((1, "o", "forward proposal"),
                                         (-1, "s", "reverse proposal")):
            chosen = sorted((r for r in rows
                             if r["proposal_direction"] == direction
                             and r["deterministic_rate_law"] ==
                             "complete_dissipation"),
                            key=lambda r: r["proposal_fraction"])
            x = [r["proposal_fraction"] for r in chosen]
            axes[row_index, 0].plot(
                x, [r["a_to_b_event_J"] for r in chosen], marker=marker,
                label=label+" A→B")
            axes[row_index, 0].plot(
                x, [r["b_to_a_event_J"] for r in chosen], marker=marker,
                linestyle="--", label=label+" B→A")
        for law, style in (("legacy_independent_metropolis", "--"),
                           ("complete_dissipation", "-")):
            chosen = sorted((r for r in rows
                             if r["proposal_direction"] == 1
                             and r["deterministic_rate_law"] == law),
                            key=lambda r: r["proposal_fraction"])
            axes[row_index, 1].plot(
                [r["proposal_fraction"] for r in chosen],
                [r["net_velocity_a_to_b_m_s"] for r in chosen],
                linestyle=style, marker="o", label=law)
        axes[row_index, 0].axhline(0.0, color="k", linewidth=.7)
        axes[row_index, 1].axhline(0.0, color="k", linewidth=.7)
        axes[row_index, 0].set_ylabel(f"{state['name']}\ncomplete event ΔF (J)")
        axes[row_index, 1].set_ylabel("net velocity (m/s)")
        for axis in axes[row_index]:
            axis.grid(alpha=.25)
    axes[0, 0].legend(fontsize=7, ncol=2)
    axes[0, 1].legend(fontsize=7)
    axes[1, 0].set_xlabel("proposal fraction")
    axes[1, 1].set_xlabel("proposal fraction")
    figure.tight_layout(); args.front_figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.front_figure, dpi=180); plt.close(figure)

    context = resolved_bicrystal(
        grid=128, length_m=3.2e-6, interface_width_m=4e-7,
        temperature_K=1100.0, child_line_fraction=.35)
    state, _ = load_stage(args.post_front, context)
    common, _ = reconstruct_common(state.common_front, context["spacing_m"])
    curl = nye_from_plastic_distortion(common.beta_p, context["spacing_m"])
    bulk = reservoir_nye_m1(
        state.mechanical.reservoir_alignment, context["systems"],
        common.orientation_rad, context["topologies"])["total"]
    interface = np.asarray(state.common_front.interface_nye_m1)
    fields = (("curl βᵖ", curl), ("reservoir bulk", bulk),
              ("support-gradient", interface),
              ("identity residual", curl-bulk-interface))
    figure, axes = plt.subplots(1, 4, figsize=(13, 3.3))
    extent = (0.0, 3.2, 0.0, 3.2)
    for axis, (title, value) in zip(axes, fields):
        norm = np.linalg.norm(value, axis=(-2, -1))
        image = axis.imshow(norm.T, origin="lower", extent=extent, cmap="magma")
        axis.set_title(title); axis.set_xlabel("x (µm)")
        figure.colorbar(image, ax=axis, shrink=.8, label="tensor norm (m⁻¹)")
    axes[0].set_ylabel("y (µm)")
    figure.tight_layout(); args.gradient_figure.parent.mkdir(
        parents=True, exist_ok=True)
    figure.savefig(args.gradient_figure, dpi=180); plt.close(figure)


if __name__ == "__main__":
    main()
