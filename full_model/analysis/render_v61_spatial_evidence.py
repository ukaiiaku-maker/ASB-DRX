"""Render physical-axis V61 fields and package original animation frames."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from full_model.analysis.postprocess_v59_physical_asb import (
    _wall_parameters, field_metrics,
)
from full_model.analysis.run_v58_three_grain_production import _load_checkpoint
from full_model.production.common_tensorial_wall import CommonWallDriving
from full_model.production.multigrain_common_state import (
    reconstruct_multigrain_common,
)
from full_model.production.multigrain_production import (
    multigrain_instantaneous_dissipation_fields,
)
from full_model.production.tensorial_nye import bcc_four_family_systems


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fields(path: Path):
    state, runtime, step, gamma, *_rest, configuration, provenance = (
        _load_checkpoint(path))
    n = int(configuration["n"])
    length = float(configuration["length_m"])
    spacing = length/n
    thickness = 2.0*2.48e-10
    wall = _wall_parameters(configuration, spacing)
    strain = np.array([[0.0, .5*gamma], [.5*gamma, 0.0]])
    rates = multigrain_instantaneous_dissipation_fields(
        state, driving=CommonWallDriving(mean_strain=strain),
        systems=bcc_four_family_systems(), topologies=(), wall_parameters=wall)
    common, _ = reconstruct_multigrain_common(state, spacing)
    power = np.asarray(rates["plastic_power_W_m3"])
    _, mask = field_metrics(power, spacing)
    with np.load(path, allow_pickle=False) as data:
        front = (np.asarray(data["diagnostic_front_heat_source_J_by_cell"])
                 /(float(configuration["dt_s"])*spacing**2*thickness)
                 if "diagnostic_front_heat_source_J_by_cell" in data.files
                 else np.zeros((n, n)))
        mechanical = (np.asarray(data["diagnostic_mechanical_heat_J_m3_cells"])
                      /float(configuration["dt_s"])
                      if "diagnostic_mechanical_heat_J_m3_cells" in data.files
                      else np.asarray(rates["irreversible_heat_rate_W_m3"]))
    ids = np.asarray(state.grain_ids, dtype=float)[
        np.argmax(state.supports, axis=0)]
    return {
        "dominant_grain_id": ids,
        "signed_accumulated_slip": np.sum(common.slip, axis=2),
        "signed_slip_rate_s": np.sum(rates["signed_slip_rate_s"], axis=2),
        "temperature_K": np.asarray(common.temperature_K),
        "mechanical_heat_rate_W_m3": mechanical,
        "front_heat_rate_W_m3": front,
        "candidate_mask": mask.astype(np.uint8),
        "step": step, "time_s": runtime.ledger.physical_time_s,
        "length_m": length,
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": _sha(path),
        "source_commit": (provenance or {}).get("source_commit"),
    }


def render(run: Path, output: Path, frame_stride: int = 1) -> dict:
    checkpoints = sorted(run.glob("checkpoint_*.npz"))[::max(frame_stride, 1)]
    if not checkpoints:
        raise ValueError("run contains no checkpoints")
    output.mkdir(parents=True, exist_ok=True)
    frames_dir = output/"frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    records = []
    frame_paths = []
    raw_final = None
    for checkpoint in checkpoints:
        item = _fields(checkpoint)
        extent = (0.0, item["length_m"]*1e6, 0.0, item["length_m"]*1e6)
        figure, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
        panels = (
            (item["dominant_grain_id"], "dominant grain ID", "tab10"),
            (item["signed_accumulated_slip"],
             "signed accumulated family slip", "coolwarm"),
            (item["signed_slip_rate_s"],
             "signed family slip rate [s$^{-1}$]", "coolwarm"),
            (item["temperature_K"], "temperature [K]", "inferno"),
            (item["mechanical_heat_rate_W_m3"],
             "mechanical/reaction heat [W m$^{-3}$]", "magma"),
            (item["front_heat_rate_W_m3"],
             "front heat [W m$^{-3}$]", "plasma"),
        )
        for index, (axis, (field, title, cmap)) in enumerate(
                zip(axes.flat, panels)):
            kwargs = {}
            if cmap == "coolwarm":
                bound = max(float(np.max(np.abs(field))), 1e-300)
                kwargs.update(vmin=-bound, vmax=bound)
            image = axis.imshow(
                field.T, origin="lower", extent=extent, cmap=cmap,
                interpolation="nearest", aspect="equal", **kwargs)
            if index in (2, 3, 4, 5) and np.any(item["candidate_mask"]):
                coordinates = np.linspace(0.0, item["length_m"]*1e6,
                                          item["candidate_mask"].shape[0])
                axis.contour(coordinates, coordinates,
                             item["candidate_mask"].T,
                             levels=[.5], colors="cyan", linewidths=.7)
            axis.set_title(title)
            axis.set_xlabel("x [μm]"); axis.set_ylabel("y [μm]")
            figure.colorbar(image, ax=axis, shrink=.75)
        figure.suptitle(
            f"step {item['step']}, physical time {item['time_s']*1e6:.3f} μs")
        frame = frames_dir/f"frame_{item['step']:06d}.png"
        figure.savefig(frame, dpi=160)
        plt.close(figure)
        frame_paths.append(frame)
        records.append({
            "step": item["step"], "physical_time_s": item["time_s"],
            "checkpoint": item["checkpoint"],
            "checkpoint_sha256": item["checkpoint_sha256"],
            "frame": str(frame.resolve()), "frame_sha256": _sha(frame),
        })
        raw_final = item
    gif = output/"spatial_evolution.gif"
    images = [Image.open(path).convert("RGB") for path in frame_paths]
    images[0].save(gif, save_all=True, append_images=images[1:],
                   duration=180, loop=0, optimize=False)
    for image in images:
        image.close()
    raw_path = output/"selected_final_raw_arrays.npz"
    np.savez_compressed(raw_path, **{
        key: value for key, value in raw_final.items()
        if isinstance(value, np.ndarray)})
    archive = output/"original_animation_bytes.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(gif, gif.name)
        for frame in frame_paths:
            bundle.write(frame, f"frames/{frame.name}")
    result = {
        "schema": "asb-drx-v61-spatial-evidence-v1",
        "run": str(run.resolve()),
        "frame_count": len(frame_paths),
        "frame_stride": max(frame_stride, 1),
        "frames": records,
        "gif": str(gif.resolve()), "gif_sha256": _sha(gif),
        "archive": str(archive.resolve()), "archive_sha256": _sha(archive),
        "raw_final_arrays": str(raw_path.resolve()),
        "raw_final_arrays_sha256": _sha(raw_path),
        "selected_late_png": str(frame_paths[-1].resolve()),
        "selected_late_png_sha256": _sha(frame_paths[-1]),
        "field_semantics": {
            "signed_accumulated_slip": "sum of signed family slips",
            "signed_slip_rate": "sum of instantaneous signed family rates",
            "candidate_outline": (
                "largest periodic instantaneous plastic-power component above "
                "mean plus one standard deviation"),
            "mechanical_heat": "preceding accepted macro-interval average",
            "front_heat": "preceding accepted macro-interval local event average",
        },
    }
    manifest = output/"spatial_evidence_manifest.json"
    manifest.write_text(json.dumps(result, indent=2)+"\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frame-stride", type=int, default=1)
    args = parser.parse_args()
    result = render(args.run, args.output, args.frame_stride)
    print(json.dumps({key: result[key] for key in (
        "frame_count", "gif_sha256", "archive_sha256",
        "selected_late_png")}, indent=2))


if __name__ == "__main__":
    main()
