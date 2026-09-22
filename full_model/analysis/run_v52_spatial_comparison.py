#!/usr/bin/env python3
"""Compare one common-history macro from analytic n128/n192 initial states."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from full_model.analysis.run_v34_finite_coupled_response import resolved_bicrystal
from full_model.analysis.run_v39_common_horizon import load_stage
from full_model.analysis.run_v48_physical_continuation import observables
from full_model.analysis.run_v49_physical_continuation import driving


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(a, b):
    return float(abs(float(a)-float(b))/max(abs(float(b)), 1e-300))


def common_band_relative(coarse, fine, half_width=32):
    arrays = []
    for value in (coarse, fine):
        value = np.asarray(value, dtype=float)
        spectrum = np.fft.fftshift(
            np.fft.fftn(value, axes=(0, 1)), axes=(0, 1))/np.prod(value.shape[:2])
        cx, cy = np.asarray(value.shape[:2])//2
        arrays.append(spectrum[
            cx-half_width:cx+half_width+1,
            cy-half_width:cy+half_width+1])
    return float(np.linalg.norm(arrays[0]-arrays[1])
                 /max(np.linalg.norm(arrays[1]), 1e-300))


def fine_only_tail_fraction(field, coarse_nyquist=64):
    value = np.asarray(field, dtype=float)
    spectrum = np.fft.fftshift(
        np.fft.fftn(value, axes=(0, 1)), axes=(0, 1))
    nx, ny = value.shape[:2]; mx = np.arange(nx)-nx//2; my = np.arange(ny)-ny//2
    tail = (np.abs(mx[:, None]) > coarse_nyquist) \
        | (np.abs(my[None, :]) > coarse_nyquist)
    trailing = (1,)*(value.ndim-2)
    numerator = np.sum(np.abs(spectrum)**2*tail.reshape(tail.shape+trailing))
    return float(np.sqrt(numerator/max(np.sum(np.abs(spectrum)**2), 1e-300)))


def integrated_inventory(state, spacing):
    result = {}
    for name in state.mechanical.density.__dataclass_fields__:
        result[name] = float(np.sum(
            getattr(state.mechanical.density, name), dtype=np.longdouble)
            *spacing**2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n128-checkpoint", type=Path, required=True)
    parser.add_argument("--n128-manifest", type=Path, required=True)
    parser.add_argument("--n192-checkpoint", type=Path, required=True)
    parser.add_argument("--n192-manifest", type=Path, required=True)
    parser.add_argument("--initialization-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    kwargs = dict(length_m=3.2e-6, interface_width_m=4e-7,
                  temperature_K=1100.0, child_line_fraction=.35)
    contexts = {n: resolved_bicrystal(grid=n, **kwargs) for n in (128, 192)}
    states = {}; manifests = {}; metadatas = {}; observations = {}; initials = {}
    for n, checkpoint, manifest_path in (
            (128, args.n128_checkpoint, args.n128_manifest),
            (192, args.n192_checkpoint, args.n192_manifest)):
        state, metadata = load_stage(checkpoint, contexts[n])
        manifest = json.loads(manifest_path.read_text())
        if metadata["completed_intervals"] != 1:
            raise RuntimeError("spatial comparison requires the first common macro")
        endpoint_drive = driving(n, .01, "continued_deformation", 100.0,
                                 metadata["physical_time_s"])
        initial_drive = driving(n, .01, "continued_deformation", 100.0, 0.0)
        states[n] = state; manifests[n] = manifest; metadatas[n] = metadata
        observations[n] = observables(contexts[n], state, endpoint_drive)
        initials[n] = observables(contexts[n], contexts[n]["state"], initial_drive)
    increments = {}
    keys = ("mean_shear_stress_sigma_12_Pa",
            "engineering_plastic_shear_gamma_p", "temperature_mean_K",
            "temperature_peak_K")
    for n in (128, 192):
        increments[n] = {key: float(observations[n][key]-initials[n][key])
                         for key in keys}
    primary = {
        key: {
            "n128_increment": increments[128][key],
            "n192_increment": increments[192][key],
            "relative_difference": relative(
                increments[128][key], increments[192][key]),
        } for key in keys
    }
    inventories = {n: integrated_inventory(states[n], contexts[n]["spacing_m"])
                   for n in (128, 192)}
    inventory_relative = {
        key: relative(inventories[128][key], inventories[192][key])
        for key in inventories[128]
    }
    fields = {
        "beta_p": (states[128].mechanical.common.beta_p,
                   states[192].mechanical.common.beta_p),
        "family_nye": (states[128].mechanical.common.family_nye_m1,
                       states[192].mechanical.common.family_nye_m1),
        "temperature_rise": (
            states[128].mechanical.common.temperature_K-1100.0,
            states[192].mechanical.common.temperature_K-1100.0),
        "wall_ordered_plus": (
            states[128].mechanical.density.wall_ordered_plus_m2,
            states[192].mechanical.density.wall_ordered_plus_m2),
    }
    structure = {name: {
        "common_band_half_width_32_relative_l2": common_band_relative(*pair),
        "n192_fine_only_above_n128_nyquist_relative_l2": (
            fine_only_tail_fraction(pair[1])),
        "n128_rms": float(np.sqrt(np.mean(np.asarray(pair[0])**2))),
        "n192_rms": float(np.sqrt(np.mean(np.asarray(pair[1])**2))),
    } for name, pair in fields.items()}
    audit = json.loads(args.initialization_audit.read_text())
    payload = {
        "schema": "asb-drx/v52/analytic-common-history-spatial-estimate/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "one first macro from independently evaluated common analytic "
            "physical initial fields; not an evolved-state transfer and not "
            "a convergence certificate for the 25.39 us history"),
        "route": audit["route"],
        "initialization_audit_sha256": digest(args.initialization_audit),
        "physical_time_s": metadatas[128]["physical_time_s"],
        "common_checkpoint_time_exact": bool(
            metadatas[128]["physical_time_s"]
            ==metadatas[192]["physical_time_s"]),
        "source_shas": {str(n): manifests[n]["source_sha"] for n in (128, 192)},
        "checkpoint_sha256": {
            "128": digest(args.n128_checkpoint), "192": digest(args.n192_checkpoint)},
        "primary_increment_comparison": primary,
        "primary_all_below_5_percent": bool(all(
            row["relative_difference"] < .05 for row in primary.values())),
        "integrated_signed_reservoir_relative_differences": inventory_relative,
        "spatial_structure": structure,
        "interpretation": (
            "mean-response increments are judged separately from ordered/Nye "
            "full-band structure; a common-band pass cannot hide fine-only content"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps({
        "sha256": digest(args.output),
        "primary_all_below_5_percent": payload["primary_all_below_5_percent"],
        "primary": primary,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
