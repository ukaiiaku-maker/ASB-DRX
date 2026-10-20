#!/usr/bin/env python3
"""Render matched V67 microstructure, defect, thermal, and power fields."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from full_model.analysis.run_v58_three_grain_production import _load_checkpoint
from full_model.analysis.run_v66_coupled_rx_asb import _physics
from full_model.production.coupled_multigrain_interval import make_endpoint
from full_model.production.multigrain_common_state import (
    audit_multigrain_nye, reconstruct_multigrain_common,
)


LINE_FIELDS = (
    "mobile_plus_m2", "mobile_minus_m2", "forest_plus_m2",
    "forest_minus_m2", "wall_plus_m2", "wall_minus_m2",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame(root: Path) -> dict:
    checkpoint = root / "checkpoint_latest.npz"
    state, runtime, step, load, _, config, provenance = _load_checkpoint(checkpoint)
    spacing = config["length_m"] / config["n"]
    common, _ = reconstruct_multigrain_common(state, spacing)
    nye = audit_multigrain_nye(state, spacing).exact_reconstructed_m1
    args = SimpleNamespace(
        n=config["n"], length=config["length_m"],
        temperature=config["temperature_K"],
        shear_rate=config["shear_rate_s"], dt=config["dt_s"],
        thermal_control=config["thermal_control"],
        bath_rate=config["thermal_bath_rate_s"])
    endpoint = make_endpoint(state, runtime, load, _physics(args)[-1])
    power = np.maximum(np.asarray(
        endpoint.observables["instantaneous_plastic_power_W_m3"]), 0.0)
    density = sum(np.sum(np.asarray(getattr(common, name)), axis=2)
                  for name in LINE_FIELDS)
    owner_index = np.argmax(np.asarray(state.supports), axis=0)
    owner_id = np.asarray(state.grain_ids, dtype=float)[owner_index]
    return {
        "case": root.name,
        "path": str(root.resolve()),
        "checkpoint_sha256": _sha(checkpoint),
        "source_commit": (provenance or {}).get("source_commit"),
        "step": step,
        "load": load,
        "time_s": runtime.ledger.physical_time_s,
        "thermal_control": config["thermal_control"],
        "extent_um": [0.0, config["length_m"] * 1e6,
                      0.0, config["length_m"] * 1e6],
        "fields": (
            owner_id,
            np.rad2deg(np.asarray(common.orientation_rad)),
            np.asarray(density) / 1e17,
            np.log10(np.maximum(np.linalg.norm(nye, axis=(-2, -1)), 1.0)),
            np.asarray(endpoint.observables["common_temperature_K"]),
            np.log10(np.maximum(power, 1.0)),
        ),
    }


def fixed_limits(frames: list[dict]) -> list[tuple[float, float]]:
    limits = []
    for field_index in range(len(frames[0]["fields"])):
        values = np.stack([item["fields"][field_index] for item in frames])
        low, high = float(np.nanmin(values)), float(np.nanmax(values))
        limits.append((low, high if high > low else low + 1.0))
    return limits


def render(roots: list[Path], output_dir: Path) -> dict:
    frames = [frame(root) for root in roots]
    limits = fixed_limits(frames)
    titles = (
        "dominant grain ID", "orientation (degrees)",
        "total line density / $10^{17}$ m$^{-2}$",
        "log10 exact Nye norm [m$^{-1}$]", "temperature [K]",
        "log10 plastic power [W m$^{-3}$]",
    )
    cmaps = ("tab10", "twilight", "magma", "inferno", "coolwarm", "plasma")
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for item in frames:
        figure, axes = plt.subplots(
            2, 3, figsize=(12.5, 7.8), constrained_layout=True)
        for axis, field, title, cmap, bound in zip(
                axes.flat, item["fields"], titles, cmaps, limits):
            image = axis.imshow(
                field.T, origin="lower", extent=item["extent_um"], cmap=cmap,
                vmin=bound[0], vmax=bound[1], interpolation="nearest",
                aspect="equal")
            axis.set_title(title)
            axis.set_xlabel("x [µm]")
            axis.set_ylabel("y [µm]")
            figure.colorbar(image, ax=axis, shrink=.72)
        figure.suptitle(
            f"{item['case']}: shear={item['load']:.5f}, "
            f"t={item['time_s'] * 1e6:.3f} µs", fontsize=9)
        target = output_dir / f"{item['case']}_fields.png"
        figure.savefig(target, dpi=180)
        plt.close(figure)
        files[item["case"]] = {
            "path": str(target.resolve()), "sha256": _sha(target)}
    manifest = {
        "schema": "v67-poly-rx-asb-field-render-v1",
        "coordinate_semantics": (
            "native node-centered production arrays on physical [0,L) axes; "
            "no interpolation, registration, or spatial projection"),
        "fixed_color_limits": {
            title: list(bound) for title, bound in zip(titles, limits)},
        "frames": [{key: value for key, value in item.items()
                    if key != "fields"} for item in frames],
        "files": files,
    }
    target = output_dir / "render_manifest.json"
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = render(args.case, args.output_dir)
    print(json.dumps({"files": manifest["files"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
