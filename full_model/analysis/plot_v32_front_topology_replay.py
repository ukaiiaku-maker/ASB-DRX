#!/usr/bin/env python3
"""Plot the V31 named terminal and its V32 topology-aware replay terminal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _event(path):
    value = json.loads(Path(path).read_text())
    return value["front_decision"]["topology_event"], value


def _draw(ax, components, title):
    for component in components:
        points = component["points_grid"]
        ax.plot([p[1] for p in points], [p[0] for p in points], ".-", ms=2,
                label=(f"id {component['component_id']}; "
                       f"L={component['interface_length_cells']:.2f}"))
    ax.set(title=title, xlabel="grid y", ylabel="grid x", aspect="equal")
    ax.invert_yaxis()
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=.2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v31-terminal", type=Path, required=True)
    parser.add_argument("--v32-terminal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original, original_root = _event(args.v31_terminal)
    replay, replay_root = _event(args.v32_terminal)
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    _draw(axes[0, 0], original["old_components"],
          f"V31 step {original_root['step']}: before")
    _draw(axes[0, 1], original["new_components"],
          "V31: sub-cell island appears\n"
          f"reported {original['classification']}")
    _draw(axes[1, 0], replay["old_components"],
          f"V32 replay step {replay_root['step']}: before")
    _draw(axes[1, 1], replay["new_components"],
          "V32: 73→165 ray crossings\n"
          f"reported {replay['classification']}")
    figure.suptitle(
        "Front topology audit: sub-grid filtering permits conservative motion;\n"
        "resolved discontinuous graph change remains fail-closed", fontsize=12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
