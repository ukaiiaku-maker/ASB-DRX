#!/usr/bin/env python3
"""Render fixed-scale accepted-state fields for a V59 trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from full_model.analysis.postprocess_v59_physical_asb import (
    _checkpoint_map, _digest, _wall_parameters,
)
from full_model.analysis.run_v58_three_grain_production import _load_checkpoint
from full_model.production.common_tensorial_wall import CommonWallDriving
from full_model.production.multigrain_common_state import (
    audit_multigrain_nye, reconstruct_multigrain_common,
)
from full_model.production.multigrain_production import (
    multigrain_instantaneous_dissipation_fields,
)
from full_model.production.tensorial_nye import bcc_four_family_systems


def _frame(path: Path) -> dict:
    (state, runtime, step, gamma, _, configuration,
     provenance) = _load_checkpoint(path)
    spacing = float(configuration["length_m"])/int(configuration["n"])
    common, _ = reconstruct_multigrain_common(state, spacing)
    wall = _wall_parameters(configuration, spacing)
    strain = np.array([[0.0, .5*gamma], [.5*gamma, 0.0]])
    rates = multigrain_instantaneous_dissipation_fields(
        state, driving=CommonWallDriving(mean_strain=strain),
        systems=bcc_four_family_systems(), topologies=(),
        wall_parameters=wall)
    audit = audit_multigrain_nye(state, spacing)
    total_density = sum(
        np.sum(np.asarray(getattr(common, name)), axis=2)
        for name in ("mobile_plus_m2", "mobile_minus_m2",
                     "forest_plus_m2", "forest_minus_m2",
                     "wall_plus_m2", "wall_minus_m2"))
    return {
        "step": step, "time_s": runtime.ledger.physical_time_s,
        "strain": gamma, "source_commit": (provenance or {}).get("source_commit"),
        "checkpoint": str(path.resolve()), "checkpoint_sha256": _digest(path),
        "fields": (
            np.argmax(state.supports, axis=0).astype(float),
            np.asarray(total_density)/1e17,
            np.log10(np.maximum(np.linalg.norm(
                audit.exact_reconstructed_m1, axis=(-2, -1)), 1.0)),
            np.asarray(common.temperature_K),
            np.sum(np.abs(np.asarray(common.slip)), axis=2),
            np.log10(np.maximum(np.asarray(
                rates["plastic_power_W_m3"]), 1.0)),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-frames", type=int, default=60)
    parser.add_argument("--fps", type=float, default=6.0)
    args = parser.parse_args()
    checkpoints = [path for _, path in sorted(_checkpoint_map(args.run).items())]
    if not checkpoints:
        raise ValueError("movie requires accepted checkpoints")
    count = min(len(checkpoints), max(int(args.maximum_frames), 2))
    selection = np.unique(np.linspace(
        0, len(checkpoints)-1, count, dtype=int))
    frames = [_frame(checkpoints[index]) for index in selection]
    titles = (
        "dominant owner", "line density / $10^{17}$ m$^{-2}$",
        "log10 Nye norm [m$^{-1}$]", "temperature [K]",
        "accumulated |slip|", "log10 plastic power [W m$^{-3}$]",
    )
    cmaps = ("tab10", "magma", "inferno", "coolwarm", "viridis", "plasma")
    limits = []
    for field_index in range(6):
        values = np.stack([frame["fields"][field_index] for frame in frames])
        limits.append((float(np.nanmin(values)), float(np.nanmax(values))))
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    images = []
    for axis, title, cmap, (low, high), field in zip(
            axes.flat, titles, cmaps, limits, frames[0]["fields"]):
        if high <= low:
            high = low+1.0
        image = axis.imshow(field, origin="lower", cmap=cmap, vmin=low, vmax=high)
        axis.set_title(title); axis.set_xticks(()); axis.set_yticks(())
        figure.colorbar(image, ax=axis, shrink=.75)
        images.append(image)
    heading = figure.suptitle("")

    def update(index):
        frame = frames[index]
        for image, field in zip(images, frame["fields"]):
            image.set_data(field)
        heading.set_text(
            f"step {frame['step']}   t={frame['time_s']*1e6:.3f} us   "
            f"gamma={frame['strain']:.4f}")
        return (*images, heading)

    animation = FuncAnimation(
        figure, update, frames=len(frames), interval=1000.0/args.fps,
        blit=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(args.output, writer=PillowWriter(fps=args.fps), dpi=110)
    plt.close(figure)
    manifest = {
        "schema": "asb-drx-v59-physical-field-movie-v1",
        "output": str(args.output.resolve()),
        "fixed_color_limits": {
            title: list(limit) for title, limit in zip(titles, limits)},
        "frames": [{key: value for key, value in frame.items() if key != "fields"}
                   for frame in frames],
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
