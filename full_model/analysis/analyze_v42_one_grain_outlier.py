#!/usr/bin/env python3
"""Local mechanics and resolution audit of the retained 5% orientation outlier."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from full_model.analysis.run_v30_mura_tier_b1_case import create_case, load_checkpoint
from full_model.production.common_tensorial_wall import CommonWallDriving
from full_model.production.extensive_wall import extensive_wall_energy_components_J_m3
from full_model.production.tensorial_nye import nye_from_plastic_distortion
from full_model.production.v24_mechanical_wall import resolved_driving_components
from full_model.production.wall_topology_supply import reservoir_nye_m1


def patch_stats(field, index, radius=2):
    nx, ny = field.shape[:2]
    ii = [(index[0]+d) % nx for d in range(-radius, radius+1)]
    jj = [(index[1]+d) % ny for d in range(-radius, radius+1)]
    values = np.asarray(field)[np.ix_(ii, jj)]
    return {"mean": float(np.mean(values)), "maximum": float(np.max(values)),
            "minimum": float(np.min(values)), "rms": float(np.sqrt(np.mean(values**2)))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    context = create_case(64, "mechanical_heterogeneity", 42, 1e-5)
    _, fixed, _, systems, topologies, common_parameters, extensive, _, dx = context
    state, metadata = load_checkpoint(args.checkpoint, systems, topologies)
    theta = np.asarray(state.common.orientation_rad)
    theta_unwrapped = np.unwrap(np.unwrap(theta, axis=0), axis=1)
    alpha = nye_from_plastic_distortion(state.common.beta_p, dx)
    reservoir = reservoir_nye_m1(
        state.reservoir_alignment, systems, theta, topologies)
    strain = float(metadata["applied_strain"])
    driving = CommonWallDriving(
        mean_strain=np.array([[0.0, .5*strain], [.5*strain, 0.0]]),
        fixed_eigenstrain=fixed)
    drive = resolved_driving_components(
        state.common, driving, systems, topologies, common_parameters)
    stress_norm = np.linalg.norm(drive["raw_stress_Pa"], axis=2)
    slip_norm = np.linalg.norm(state.common.slip, axis=2)
    beta_norm = np.linalg.norm(state.common.beta_p, axis=(-2, -1))
    alpha_norm = np.linalg.norm(alpha, axis=(-2, -1))
    donor = np.sum(
        state.density.wall_tangle_plus_m2+state.density.wall_tangle_minus_m2,
        axis=2)
    zero = np.zeros(theta.shape+(3, 3))
    energy = extensive_wall_energy_components_J_m3(
        state.density, systems, topologies, theta, zero, extensive)["total"]
    extrema = {"minimum": np.unravel_index(np.argmin(theta), theta.shape),
               "maximum": np.unravel_index(np.argmax(theta), theta.shape)}
    patches = {}
    for label, index in extrema.items():
        patches[label] = {
            "index": [int(index[0]), int(index[1])],
            "orientation_deg": float(theta[index]*180/np.pi),
            "slip_norm": patch_stats(slip_norm, index),
            "plastic_distortion_norm": patch_stats(beta_norm, index),
            "stress_norm_Pa": patch_stats(stress_norm, index),
            "curl_nye_norm_m1": patch_stats(alpha_norm, index),
            "tangle_donor_m2": patch_stats(donor, index),
            "defect_energy_J_m3": patch_stats(energy, index),
        }
    robust = np.percentile(theta*180/np.pi, [1, 5, 50, 95, 99])
    payload = {
        "schema": "asb-drx/v42/one-grain-response/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "step": int(metadata["step"]), "strain": strain, "grid": 64,
        "spacing_m": dx,
        "orientation": {
            "raw_span_deg": float(np.ptp(theta)*180/np.pi),
            "unwrapped_span_deg": float(np.ptp(theta_unwrapped)*180/np.pi),
            "percentiles_deg": {name: float(value) for name, value in zip(
                ("p01", "p05", "p50", "p95", "p99"), robust)},
            "p95_minus_p05_deg": float(robust[3]-robust[1]),
            "coordinate_wrapping_detected": bool(
                abs(np.ptp(theta_unwrapped)-np.ptp(theta)) > 1e-12),
        },
        "extrema_patches": patches,
        "global": {
            "curl_nye_rms_m1": float(np.sqrt(np.mean(alpha*alpha))),
            "reservoir_nye_rms_m1": float(np.sqrt(np.mean(
                reservoir["total"]**2))),
            "stress_norm_maximum_Pa": float(np.max(stress_norm)),
            "tangle_donor_maximum_m2": float(np.max(donor)),
        },
        "classification": "NO_QUALIFIED_BOUNDARY;UNDERRESOLVED_LOCAL_ORIENTATION_EXTREMUM",
        "interpretation": (
            "the global span is localized rather than a pair of resolved plateaus; "
            "this diagnostic does not establish that a finer model generated the history"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    fields = ((theta*180/np.pi, "orientation (deg)"),
              (slip_norm, "slip norm"), (beta_norm, "beta-p norm"),
              (alpha_norm, "curl-Nye norm (1/m)"),
              (stress_norm/1e9, "stress norm (GPa)"),
              (donor, "tangle donor (1/m2)"))
    for axis, (value, title) in zip(axes.ravel(), fields):
        image = axis.imshow(value.T, origin="lower", cmap="viridis")
        for label, index in extrema.items():
            axis.plot(index[0], index[1], "o", ms=4,
                      color="white" if label == "maximum" else "red")
        axis.set_title(title); fig.colorbar(image, ax=axis, shrink=.78)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=180); plt.close(fig)


if __name__ == "__main__":
    main()
