#!/usr/bin/env python3
"""Field-level weak/strong audit of the declared moving-front Nye split."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from full_model.analysis.run_v34_finite_coupled_response import resolved_bicrystal
from full_model.analysis.run_v39_common_horizon import load_stage
from full_model.production.common_front_state import reconstruct_common
from full_model.production.tensorial_nye import nye_from_plastic_distortion
from full_model.production.wall_topology_supply import reservoir_nye_m1


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rms(value):
    return float(np.sqrt(np.mean(np.asarray(value)**2)))


def modes(value, half_width=24):
    value = np.asarray(value)
    transformed = np.fft.fftshift(
        np.fft.fftn(value, axes=(0, 1))/(value.shape[0]*value.shape[1]),
        axes=(0, 1))
    cx, cy = value.shape[0]//2, value.shape[1]//2
    return transformed[cx-half_width:cx+half_width+1,
                       cy-half_width:cy+half_width+1]


def fields(path, grid):
    context = resolved_bicrystal(
        grid=grid, length_m=3.2e-6, interface_width_m=4e-7,
        temperature_K=1100.0, child_line_fraction=.35)
    state, metadata = load_stage(path, context)
    common, correction = reconstruct_common(
        state.common_front, context["spacing_m"])
    curl = nye_from_plastic_distortion(
        common.beta_p, context["spacing_m"])
    bulk = reservoir_nye_m1(
        state.mechanical.reservoir_alignment, context["systems"],
        common.orientation_rad, context["topologies"])["total"]
    interface = np.asarray(state.common_front.interface_nye_m1)
    residual = curl-(bulk+interface)
    chi = np.asarray(state.common_front.front.chi)
    support = (chi > .05)&(chi < .95)
    dx = context["spacing_m"]; area = dx*dx
    def record(value):
        tensor_norm = np.linalg.norm(value, axis=(-2, -1))
        signed_circuit = np.sum(value, axis=0, dtype=np.longdouble)*dx
        return {
            "rms_m1": rms(value),
            "maximum_tensor_norm_m1": float(np.max(tensor_norm)),
            "l1_tensor_integral_m": float(np.sum(
                tensor_norm, dtype=np.longdouble)*area),
            "interface_support_rms_m1": (rms(value[support])
                                         if np.any(support) else 0.0),
            "bulk_support_rms_m1": (rms(value[~support])
                                    if np.any(~support) else 0.0),
            "x_normal_signed_circuit_rms": rms(signed_circuit),
            "x_normal_signed_circuit_components": np.asarray(np.mean(
                signed_circuit, axis=0), dtype=float).tolist(),
        }
    return {
        "metadata": metadata,
        "checkpoint": str(Path(path).resolve()),
        "checkpoint_sha256": digest(path),
        "grid": grid, "spacing_m": dx,
        "interface_support_cells": int(np.count_nonzero(support)),
        "interface_support_width_m": float(np.count_nonzero(support)*area
                                            /(2.0*3.2e-6)),
        "curl": record(curl), "reservoir_bulk": record(bulk),
        "interface_product_rule": record(interface),
        "declared_identity_residual": record(residual),
        "_modes": {name: modes(value) for name, value in (
            ("curl", curl), ("interface_product_rule", interface),
            ("declared_identity_residual", residual))},
    }


def main():
    parser = argparse.ArgumentParser()
    for stage in ("post_front", "post_second_mura"):
        for grid in (128, 192):
            parser.add_argument(f"--{stage.replace('_','-')}-n{grid}",
                                type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = {}
    for stage in ("post_front", "post_second_mura"):
        data[stage] = {}
        for grid in (128, 192):
            path = getattr(args, f"{stage}_n{grid}")
            data[stage][grid] = fields(path, grid)
    comparisons = {}
    for stage, grids in data.items():
        comparisons[stage] = {}
        for name in ("curl", "interface_product_rule",
                     "declared_identity_residual"):
            a = grids[128]["_modes"][name]
            b = grids[192]["_modes"][name]
            delta = rms(np.abs(a-b)); scale = max(rms(np.abs(a)), rms(np.abs(b)), 1e-300)
            comparisons[stage][name] = {
                "common_physical_mode_half_width": 24,
                "coefficient_rms_difference": delta,
                "coefficient_relative_difference": delta/scale,
            }
    for grids in data.values():
        for record in grids.values():
            record.pop("_modes")
    result = {
        "schema": "asb-drx/v41/gradient-field-audit/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "stages": {stage: {str(grid): value for grid, value in grids.items()}
                   for stage, grids in data.items()},
        "common_mode_comparisons": comparisons,
        "normalization": (
            "FFT coefficients divide by grid-cell count; real-space integrals "
            "use dx*dy. No filtering or projection changes authoritative state."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
