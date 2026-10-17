#!/usr/bin/env python3
"""Plot the immutable Hall--Petch EXP-floor capacity envelope."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from full_model.analysis.analyze_v67_hall_petch_capacity import (
    hall_petch_capacity_ceiling_s,
)
from full_model.production.arrhenius_dislocation_sources import (
    ArrheniusSourceParameters,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    temperature = np.linspace(600.0, 1400.0, 401)
    parameters = ArrheniusSourceParameters(grain_size_m=2.4e-6)
    capacity = np.asarray([
        hall_petch_capacity_ceiling_s(value, parameters)[
            "maximum_all_family_shear_rate_s"]
        for value in temperature])
    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    ax.plot(temperature, capacity, color="#1769aa", linewidth=2.4,
            label="four-family HP nucleation ceiling")
    ax.axhline(2.0, color="#2e7d32", linestyle="--", linewidth=1.8,
               label="qualified lower-rate RX: 2 s$^{-1}$")
    ax.axhline(8.0e4, color="#c62828", linestyle="--", linewidth=1.8,
               label="polycrystal high-rate demand: 80000 s$^{-1}$")
    ax.set_yscale("log")
    ax.set_xlim(600.0, 1400.0)
    ax.set_ylim(1.0, 2.0e5)
    ax.set_xlabel("temperature (K)")
    ax.set_ylabel("shear rate (s$^{-1}$)")
    ax.set_title("Immutable Hall–Petch EXP-floor nucleation capacity")
    ax.grid(True, which="both", alpha=.25)
    ax.legend(loc="center right", frameon=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".png"), dpi=220)
    fig.savefig(args.out.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
