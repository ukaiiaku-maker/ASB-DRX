#!/usr/bin/env python3
"""Build a controlled full-v34 bicrystal checkpoint from a complete state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def build(source: Path, destination: Path, manifest: Path, *, width_m=3.16e-7,
          parent_density_m2=4.0e17, child_density_m2=1.0e17):
    with np.load(source, allow_pickle=True) as state:
        arrays = {name: np.asarray(state[name]).copy() for name in state.files
                  if not name.startswith("sparse_front__") and name not in {
                      "sparse_front_metadata_json", "sibm_experiment_json",
                      "sibm_reference_parent_mask", "sibm_reference_child_fraction",
                      "sibm_initial_child_fraction", "sibm_active_mask",
                      "sibm_reference_eta"}}
    eta = arrays["eta"]
    nx, ny, phases = eta.shape
    dx = 10e-6/nx
    binary = np.zeros((nx, ny), bool)
    binary[nx//2:, :] = True
    signed = dx*(ndimage.distance_transform_edt(binary)
                 - ndimage.distance_transform_edt(~binary))
    child = 0.5*(1.0+np.tanh(signed/(np.sqrt(2.0)*width_m)))
    eta.fill(0.0)
    eta[..., 0], eta[..., 1] = 1.0-child, child
    arrays["eta"] = eta
    arrays["lab"] = np.argmax(eta, axis=2).astype(arrays["lab"].dtype)
    angles = arrays["psi_gv"]
    angles[0], angles[1] = 0.0, np.deg2rad(30.0)
    arrays["psi_gv"] = angles
    arrays["psi_lat"] = (eta[..., 0]*angles[0] + eta[..., 1]*angles[1]
                         + arrays["psi_plastic"])
    target = parent_density_m2*(1.0-child)+child_density_m2*child
    current = np.maximum(np.sum(arrays["rp"]+arrays["rm"]+arrays["rho_forest"], axis=2)
                         +arrays["rho_wall"], 1.0)
    scale = target/current
    for name in ("rp", "rm", "rho_forest"):
        arrays[name] *= scale[..., None]
    arrays["rho_wall"] *= scale
    arrays["rho"] = target
    arrays["rho_mobile"] = np.sum(arrays["rp"]+arrays["rm"], axis=2)
    arrays["kappa_tot"] = np.sum(arrays["rp"]-arrays["rm"], axis=2)
    arrays["atomic_promotion_attempt_total"] = np.array(0)
    arrays["atomic_promotion_commit_total"] = np.array(0)
    arrays["atomic_promotion_rollback_total"] = np.array(0)
    arrays["atomic_promotion_events_json"] = np.array("[]")
    arrays["step"] = np.array(0, dtype=np.int32)
    arrays["sim_time"] = np.array(0.0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    record = {
        "schema": "full-v34-v13-canonical-bicrystal-source/v1",
        "source_checkpoint": str(source), "source_sha256": digest(source),
        "output_checkpoint": str(destination), "output_sha256": digest(destination),
        "grid": [nx, ny], "domain_m": [10e-6, 10e-6],
        "phase_count_preserved": phases, "active_pair": [0, 1],
        "orientations_deg": [0.0, 30.0], "interface_width_m": width_m,
        "parent_density_m-2": parent_density_m2,
        "child_density_m-2": child_density_m2,
        "purpose": "controlled full-state numerical fixture, not material calibration",
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True)+"\n")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--parent-density", type=float, default=4.0e17)
    parser.add_argument("--child-density", type=float, default=1.0e17)
    args = parser.parse_args()
    print(json.dumps(build(
        args.source, args.destination, args.manifest,
        parent_density_m2=args.parent_density,
        child_density_m2=args.child_density), sort_keys=True))


if __name__ == "__main__":
    main()
