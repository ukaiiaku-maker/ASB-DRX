#!/usr/bin/env python3
"""Render compact matched-horizon evidence from the V67 classifier output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def series(payload: dict, control: str) -> dict[str, np.ndarray]:
    rows = payload["cases"][control]
    return {
        "strain": np.asarray([row["load"] for row in rows], dtype=float),
        "stress_GPa": np.asarray([
            abs(row["mean_shear_stress_Pa"]) / 1e9 for row in rows]),
        "temperature_range_K": np.asarray([
            row["temperature_range_K"] for row in rows]),
        "band_fraction": np.asarray([
            row["active_physical_band_intervals"] / row["target_steps"]
            for row in rows]),
        "power_maximum_to_mean": np.asarray([
            row["final_power_localization"]["maximum_to_mean"]
            for row in rows]),
        "grain20_increment_percent": np.asarray([
            100.0 * row["owner_results"]["20"]["relative_area_change"]
            for row in rows]),
    }


def render(payload: dict, output_prefix: Path) -> None:
    ad = series(payload, "adiabatic")
    bath = series(payload, "strong_bath")
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.2), constrained_layout=True)
    fields = (
        ("stress_GPa", "Mean shear stress (GPa)"),
        ("temperature_range_K", "Temperature range (K)"),
        ("band_fraction", "Band-active chunk fraction"),
        ("power_maximum_to_mean", "Final max/mean plastic power"),
        ("grain20_increment_percent", "Grain 20 area increment (%)"),
    )
    for axis, (field, label) in zip(axes.flat, fields):
        axis.plot(ad["strain"], ad[field], "o-", label="adiabatic")
        axis.plot(bath["strain"], bath[field], "s--", label="strong bath")
        axis.set_xlabel("Applied shear strain")
        axis.set_ylabel(label)
        axis.grid(alpha=.25)
    axes.flat[0].legend(frameon=False)
    axes.flat[-1].axis("off")
    decision = payload["decision"]
    observation = payload["observations"]
    axes.flat[-1].text(
        0.0, 1.0,
        "Classification\n" + decision["classification"] + "\n\n"
        f"Adiabatic coarse softening: "
        f"{observation['adiabatic_coarse_endpoint_softening_fraction']:.4f}\n"
        f"Bath coarse softening: "
        f"{observation['bath_coarse_endpoint_softening_fraction']:.4f}\n"
        f"Persistent RX band: "
        f"{decision['persistent_physical_band_with_existing_boundary_rx']}\n"
        f"Heat-dependent ASB: "
        f"{decision['heat_dependent_polycrystal_asb_with_rx']}",
        ha="left", va="top", family="monospace", fontsize=9)
    fig.suptitle("V67 matched polycrystal RX / localization horizon")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=180)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("classification", type=Path)
    parser.add_argument("output_prefix", type=Path)
    args = parser.parse_args()
    render(json.loads(args.classification.read_text()), args.output_prefix)


if __name__ == "__main__":
    main()
