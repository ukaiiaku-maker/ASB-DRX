#!/usr/bin/env python3
"""Attach a response-derived cap mode to an exact common checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

PRODUCTION = Path(__file__).resolve().parents[1]/"production"
sys.path.insert(0, str(PRODUCTION))
from sibm_geometry import displace_pair_boundary  # noqa: E402


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def build(common: Path, sub: Path, super_: Path, destination: Path,
          record: Path):
    with np.load(common, allow_pickle=True) as state:
        arrays = {name: np.asarray(state[name]).copy() for name in state.files}
    with np.load(sub, allow_pickle=True) as lo, np.load(super_, allow_pickle=True) as hi:
        experiment = json.loads(str(arrays["sibm_experiment_json"].item()))
        parent, child = int(experiment["parent_label"]), int(experiment["child_label"])
        eta = arrays["eta"]
        response_mode = np.asarray(
            hi["eta"][..., child]-lo["eta"][..., child], dtype=float)
    parameters = json.loads(str(arrays["P_json"].item()))
    grid = eta.shape[0]
    spacing = float(parameters["L_phys"])/grid
    width = max(float(np.sqrt(parameters["kappa_eta"]/parameters["W_eta"])),
                spacing)
    amplitude = float(experiment["initial_bulge_radius_m"])
    epsilon = min(1e-9, 0.01*amplitude)
    kwargs = dict(
        parent=parent, child=child, spacing_m=spacing,
        interface_width_m=width,
        centre_index=tuple(experiment["centre_index"]),
        advance_direction_index=tuple(experiment["advance_direction_index"]),
        half_chord_m=float(experiment["seed_half_chord_m"]),
        active_window_radius_m=float(experiment["active_window_radius_m"]),
        profile=str(experiment["seed_profile"]))
    reference = np.asarray(arrays["sibm_reference_eta"], dtype=float)
    plus = displace_pair_boundary(
        reference, amplitude_m=amplitude+epsilon, **kwargs).eta
    minus = displace_pair_boundary(
        reference, amplitude_m=amplitude-epsilon, **kwargs).eta
    mode = np.asarray(
        (plus[..., child]-minus[..., child])/(2.0*epsilon), dtype=float)
    interior = ((eta[..., child] > 1e-6) & (eta[..., child] < 1.0-1e-6)
                & (eta[..., parent] > 1e-6) & (eta[..., parent] < 1.0-1e-6)
                & np.asarray(arrays["sibm_active_mask"], dtype=bool))
    mode = np.where(interior, mode, 0.0)
    scale = float(np.max(np.abs(mode)))
    if scale <= 1e-14:
        raise ValueError("sub/super pair does not define a resolved response mode")
    mode /= scale
    response_scale = float(np.max(np.abs(response_mode)))
    normalized_response = response_mode/max(response_scale, 1e-300)
    overlap = interior & (np.abs(mode) > 0.0)
    correlation = (float(np.corrcoef(mode[overlap], normalized_response[overlap])[0, 1])
                   if np.count_nonzero(overlap) >= 3 else None)
    arrays["sibm_coupled_neutral_direction"] = mode
    experiment.update({
        "coupled_neutral_direction": "centered derivative of prescribed pinned-cap family",
        "coupled_neutral_direction_amplitude_m": amplitude,
        "coupled_neutral_direction_perturbation_m": epsilon,
        "coupled_neutral_direction_sub_checkpoint": str(sub),
        "coupled_neutral_direction_super_checkpoint": str(super_),
        "coupled_neutral_direction_sub_sha256": digest(sub),
        "coupled_neutral_direction_super_sha256": digest(super_),
    })
    arrays["sibm_experiment_json"] = np.array(json.dumps(
        experiment, sort_keys=True, separators=(",", ":")))
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    payload = {
        "schema": "full-v34-v15-neutral-direction-checkpoint/v1",
        "common": str(common), "common_sha256": digest(common),
        "subcritical": str(sub), "subcritical_sha256": digest(sub),
        "supercritical": str(super_), "supercritical_sha256": digest(super_),
        "output": str(destination), "output_sha256": digest(destination),
        "grid": list(mode.shape), "mode_max_abs": float(np.max(np.abs(mode))),
        "mode_nonzero_cells": int(np.count_nonzero(mode)),
        "geometric_response_mode_correlation": correlation,
        "amplitude_m": amplitude, "perturbation_m": epsilon,
    }
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("common", type=Path)
    parser.add_argument("subcritical", type=Path)
    parser.add_argument("supercritical", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("record", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.common, args.subcritical, args.supercritical,
                           args.destination, args.record), sort_keys=True))


if __name__ == "__main__":
    main()
