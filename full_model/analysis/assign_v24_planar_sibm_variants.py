#!/usr/bin/env python3
"""Assign defect-energy contrasts to one unchanged equilibrated phase geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


EXCLUDE = {
    "sparse_front_metadata_json", "sibm_experiment_json",
    "sibm_reference_parent_mask", "sibm_reference_child_fraction",
    "sibm_initial_child_fraction", "sibm_active_mask", "sibm_reference_eta",
}


def digest_bytes(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with np.load(args.source, allow_pickle=True) as state:
        base = {name: np.asarray(state[name]).copy() for name in state.files
                if not name.startswith("sparse_front__") and name not in EXCLUDE}
    eta_hash = digest_bytes(base["eta"])
    h = base["eta"]**2*(3.0-2.0*base["eta"])
    child = h[..., 1]/np.maximum(h[..., 0]+h[..., 1], 1e-300)
    args.output.mkdir(parents=True, exist_ok=True)
    variants = {
        "equal": (2.5e17, 2.5e17),
        "parent_high_child_low": (4.0e17, 1.0e17),
        "reversed": (1.0e17, 4.0e17),
        "mobility_off": (4.0e17, 1.0e17),
    }
    records = []
    for name, (parent, child_value) in variants.items():
        arrays = {key: value.copy() for key, value in base.items()}
        target = parent*(1.0-child)+child_value*child
        current = np.maximum(
            np.sum(arrays["rp"]+arrays["rm"]+arrays["rho_forest"], axis=2)
            +arrays["rho_wall"], 1.0)
        scale = target/current
        for key in ("rp", "rm", "rho_forest"):
            arrays[key] *= scale[..., None]
        arrays["rho_wall"] *= scale
        arrays["rho"] = target
        arrays["rho_mobile"] = np.sum(arrays["rp"]+arrays["rm"], axis=2)
        arrays["kappa_tot"] = np.sum(arrays["rp"]-arrays["rm"], axis=2)
        arrays["step"] = np.array(0, dtype=np.int32)
        arrays["sim_time"] = np.array(0.0)
        path = args.output/f"{name}.npz"
        np.savez_compressed(path, **arrays)
        if digest_bytes(arrays["eta"]) != eta_hash:
            raise RuntimeError("variant assignment changed phase geometry")
        records.append({
            "name": name, "path": str(path), "sha256": digest(path),
            "eta_sha256": eta_hash, "parent_density_m2": parent,
            "child_density_m2": child_value,
        })
    manifest = {
        "schema": "asb-drx/v24-common-planar-sibm-state/v1",
        "equilibrated_source": str(args.source),
        "equilibrated_source_sha256": digest(args.source),
        "phase_geometry_sha256": eta_hash,
        "zero_cumulative_front_sweep_on_assignment": True,
        "variants": records,
    }
    (args.output/"manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
