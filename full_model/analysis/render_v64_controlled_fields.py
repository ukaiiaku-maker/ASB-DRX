#!/usr/bin/env python3
"""Render controlled V64 physical fields with native spatial axes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from full_model.analysis.postprocess_v59_physical_asb import (
    _checkpoint_map, _wall_parameters,
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame(path: Path) -> dict:
    state, runtime, step, load, _, configuration, provenance = _load_checkpoint(path)
    physical = (configuration["physical_configuration"]
                if configuration.get("schema") ==
                "asb-drx-v64-adaptive-production-v1" else configuration)
    n = int(physical["n"]); length = float(physical["length_m"])
    spacing = length/n
    common, _ = reconstruct_multigrain_common(state, spacing)
    wall = _wall_parameters(physical, spacing)
    strain = np.array([[0.0, .5*load], [.5*load, 0.0]])
    rates = multigrain_instantaneous_dissipation_fields(
        state, driving=CommonWallDriving(mean_strain=strain),
        systems=bcc_four_family_systems(), topologies=(),
        wall_parameters=wall)
    nye = audit_multigrain_nye(state, spacing).exact_reconstructed_m1
    density = sum(
        np.sum(np.asarray(getattr(common, name)), axis=2)
        for name in ("mobile_plus_m2", "mobile_minus_m2",
                     "forest_plus_m2", "forest_minus_m2",
                     "wall_plus_m2", "wall_minus_m2"))
    return {
        "path": str(path.resolve()), "sha256": sha256(path), "step": step,
        "time_s": runtime.ledger.physical_time_s, "load": load,
        "source_commit": (provenance or {}).get("source_commit"),
        "extent_um": [0.0, length*1e6, 0.0, length*1e6],
        "fields": (
            np.argmax(state.supports, axis=0).astype(float),
            np.asarray(density)/1e17,
            np.log10(np.maximum(np.linalg.norm(nye, axis=(-2, -1)), 1.0)),
            np.asarray(common.temperature_K),
            np.sum(np.abs(np.asarray(common.slip)), axis=2),
            np.log10(np.maximum(np.asarray(
                rates["plastic_power_W_m3"]), 1.0)),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-frames", type=int, default=24)
    parser.add_argument("--fps", type=float, default=5.0)
    args = parser.parse_args()
    checkpoints = [path for _, path in sorted(_checkpoint_map(args.run).items())]
    if not checkpoints:
        raise ValueError("render requires accepted V64 checkpoints")
    count = min(len(checkpoints), max(2, int(args.maximum_frames)))
    selected = np.unique(np.linspace(0, len(checkpoints)-1, count, dtype=int))
    frames = [frame(checkpoints[index]) for index in selected]
    titles = (
        "dominant prepared owner", "total line density / $10^{17}$ m$^{-2}$",
        "log10 exact Nye norm [m$^{-1}$]", "temperature [K]",
        "accumulated |slip|", "log10 plastic power [W m$^{-3}$]",
    )
    cmaps = ("tab10", "magma", "inferno", "coolwarm", "viridis", "plasma")
    limits = []
    for field_index in range(len(titles)):
        values = np.stack([item["fields"][field_index] for item in frames])
        limits.append((float(np.nanmin(values)), float(np.nanmax(values))))
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    images = []
    extent = frames[0]["extent_um"]
    for axis, title, cmap, limits_i, field in zip(
            axes.flat, titles, cmaps, limits, frames[0]["fields"]):
        low, high = limits_i
        if high <= low:
            high = low+1.0
        image = axis.imshow(
            field.T, origin="lower", extent=extent, cmap=cmap,
            vmin=low, vmax=high, interpolation="nearest", aspect="equal")
        axis.set_title(title); axis.set_xlabel("x [µm]"); axis.set_ylabel("y [µm]")
        figure.colorbar(image, ax=axis, shrink=.72)
        images.append(image)
    heading = figure.suptitle("")

    def update(index):
        item = frames[index]
        for image, field in zip(images, item["fields"]):
            image.set_data(field.T)
        heading.set_text(
            f"controlled V64: t={item['time_s']*1e6:.6f} µs, "
            f"engineering shear={item['load']:.6f}")
        return (*images, heading)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    update(0); first = args.output_dir/"first_controlled_fields.png"
    figure.savefig(first, dpi=160)
    update(len(frames)-1); final = args.output_dir/"final_controlled_fields.png"
    figure.savefig(final, dpi=160)
    movie = args.output_dir/"controlled_fields.gif"
    animation = FuncAnimation(
        figure, update, frames=len(frames), interval=1000.0/args.fps,
        blit=False)
    animation.save(movie, writer=PillowWriter(fps=args.fps), dpi=105)
    plt.close(figure)
    manifest = {
        "schema": "asb-drx-v64-controlled-field-render-v1",
        "coordinate_semantics": (
            "native node-centered production arrays on physical [0,L) axes; "
            "no interpolation, registration, or spatial projection"),
        "fixed_color_limits": {
            title: list(bound) for title, bound in zip(titles, limits)},
        "frames": [{key: value for key, value in item.items()
                    if key != "fields"} for item in frames],
        "files": {
            "first_png": {"path": str(first.resolve()), "sha256": sha256(first)},
            "final_png": {"path": str(final.resolve()), "sha256": sha256(final)},
            "gif": {"path": str(movie.resolve()), "sha256": sha256(movie)},
        },
    }
    output = args.output_dir/"render_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"manifest": str(output), "sha256": sha256(output),
                      "frame_count": len(frames)}))


if __name__ == "__main__":
    main()
