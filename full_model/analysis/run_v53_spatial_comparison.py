#!/usr/bin/env python3
"""V52 comparison plus the previously unreported common spectral band."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from full_model.analysis.run_v34_finite_coupled_response import resolved_bicrystal
from full_model.analysis.run_v39_common_horizon import atomic_json, load_stage


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spectrum(value):
    array = np.asarray(value, dtype=float)
    return np.fft.fftshift(
        np.fft.fftn(array, axes=(0, 1)), axes=(0, 1))/np.prod(array.shape[:2])


def relative_l2(left, right, mask=None):
    if mask is not None:
        trailing = (1,)*(left.ndim-2)
        mask = mask.reshape(mask.shape+trailing)
        left = left*mask; right = right*mask
    return float(np.linalg.norm(left-right)/max(np.linalg.norm(right), 1e-300))


def band_comparison(left, right, mask):
    trailing = (1,)*(left.ndim-2)
    selected = mask.reshape(mask.shape+trailing)
    difference = float(np.linalg.norm((left-right)*selected))
    reference = float(np.linalg.norm(right*selected))
    total = float(np.linalg.norm(right))
    resolved = reference > 1e-14*max(total, 1e-300)
    return {
        "difference_l2": difference,
        "reference_l2": reference,
        "relative_l2": difference/reference if resolved else None,
        "relative_measure_resolved": bool(resolved),
        "classification": ("RESOLVED_BAND" if resolved else
                           "REFERENCE_BAND_NUMERICALLY_NEGLIGIBLE"),
    }


def spectral_ledger(coarse, fine):
    coarse_hat = spectrum(coarse); fine_hat = spectrum(fine)
    # Strict common non-Nyquist modes are -63..63 in each direction.  The
    # n128 -64 lines are explicitly excluded because an even-grid real field
    # has only one Nyquist coefficient, while the n192 grid represents +/-64
    # separately and the production real-derivative convention zeros its own
    # Nyquist entry.  No translation or phase alignment is fitted.
    common_coarse = coarse_hat[1:, 1:]
    common_fine = fine_hat[33:160, 33:160]
    modes = np.arange(-63, 64)
    radius = np.maximum(np.abs(modes[:, None]), np.abs(modes[None, :]))
    low = radius <= 32
    intermediate = (radius >= 33) & (radius <= 63)
    fine_modes = np.arange(-96, 96)
    fine_only = (np.abs(fine_modes[:, None]) > 63) | (
        np.abs(fine_modes[None, :]) > 63)
    fine_total_sq = float(np.sum(np.abs(fine_hat)**2))
    fine_only_sq = float(np.sum(
        np.abs(fine_hat)**2*fine_only.reshape(fine_only.shape+(1,)*(fine_hat.ndim-2))))
    intermediate_sq = float(np.sum(
        np.abs(common_fine)**2*intermediate.reshape(
            intermediate.shape+(1,)*(common_fine.ndim-2))))
    coarse_nyquist = np.zeros(coarse_hat.shape[:2], dtype=bool)
    coarse_nyquist[0, :] = True; coarse_nyquist[:, 0] = True
    coarse_total_sq = float(np.sum(np.abs(coarse_hat)**2))
    coarse_nyquist_sq = float(np.sum(
        np.abs(coarse_hat)**2*coarse_nyquist.reshape(
            coarse_nyquist.shape+(1,)*(coarse_hat.ndim-2))))
    return {
        "common_non_nyquist_modes": "integer Fourier modes -63..63 on each axis",
        "low_common_band_max_abs_mode_32": band_comparison(
            common_coarse, common_fine, low),
        "remaining_common_band_max_abs_mode_33_to_63": band_comparison(
            common_coarse, common_fine, intermediate),
        "whole_common_non_nyquist_band_relative_l2": relative_l2(
            common_coarse, common_fine),
        "fine_only_beyond_abs_mode_63_squared_norm_fraction": (
            fine_only_sq/max(fine_total_sq, 1e-300)),
        "fine_only_beyond_abs_mode_63_rms_norm_fraction": float(np.sqrt(
            fine_only_sq/max(fine_total_sq, 1e-300))),
        "fine_intermediate_33_to_63_squared_norm_fraction": (
            intermediate_sq/max(fine_total_sq, 1e-300)),
        "fine_intermediate_33_to_63_rms_norm_fraction": float(np.sqrt(
            intermediate_sq/max(fine_total_sq, 1e-300))),
        "coarse_excluded_nyquist_lines_squared_norm_fraction": (
            coarse_nyquist_sq/max(coarse_total_sq, 1e-300)),
        "coarse_excluded_nyquist_lines_rms_norm_fraction": float(np.sqrt(
            coarse_nyquist_sq/max(coarse_total_sq, 1e-300))),
        "normalization": "DFT divided by number of spatial cells; Parseval ratios",
        "coordinate_origin": "shared cell-centred analytic periodic origin",
        "alignment": "none",
        "even_grid_nyquist_policy": (
            "exclude n128 -64 lines from the strict common band and report their norm separately"),
    }


def main():
    parser = argparse.ArgumentParser()
    for option in ("n128-checkpoint", "n128-manifest", "n192-checkpoint",
                   "n192-manifest", "initialization-audit", "output"):
        parser.add_argument("--"+option, type=Path, required=True)
    args = parser.parse_args()
    base_path = args.output.with_suffix(".v52-base.tmp.json")
    command = [
        sys.executable, "full_model/analysis/run_v52_spatial_comparison.py",
        "--n128-checkpoint", str(args.n128_checkpoint),
        "--n128-manifest", str(args.n128_manifest),
        "--n192-checkpoint", str(args.n192_checkpoint),
        "--n192-manifest", str(args.n192_manifest),
        "--initialization-audit", str(args.initialization_audit),
        "--output", str(base_path),
    ]
    subprocess.run(command, check=True)
    payload = json.loads(base_path.read_text()); base_path.unlink()
    kwargs = dict(length_m=3.2e-6, interface_width_m=4e-7,
                  temperature_K=1100.0, child_line_fraction=.35)
    contexts = {n: resolved_bicrystal(grid=n, **kwargs) for n in (128, 192)}
    coarse, _ = load_stage(args.n128_checkpoint, contexts[128])
    fine, _ = load_stage(args.n192_checkpoint, contexts[192])
    fields = {
        "beta_p": (coarse.mechanical.common.beta_p,
                   fine.mechanical.common.beta_p),
        "family_nye": (coarse.mechanical.common.family_nye_m1,
                       fine.mechanical.common.family_nye_m1),
        "temperature_rise": (coarse.mechanical.common.temperature_K-1100.0,
                             fine.mechanical.common.temperature_K-1100.0),
        "wall_ordered_plus": (
            coarse.mechanical.density.wall_ordered_plus_m2,
            fine.mechanical.density.wall_ordered_plus_m2),
    }
    payload["schema"] = "asb-drx/v53/analytic-common-history-spatial-estimate/v1"
    payload["generated_utc"] = datetime.now(timezone.utc).isoformat()
    payload["v52_comparison_algorithm_retained"] = True
    payload["extended_spectral_structure"] = {
        name: spectral_ledger(*pair) for name, pair in fields.items()}
    payload["spectral_claim_scope"] = (
        "observable-specific Fourier ledger; not a full-field convergence certificate")
    atomic_json(args.output, payload)
    print(json.dumps({"output": str(args.output), "sha256": digest(args.output),
                      "interval": payload["comparison_interval"],
                      "primary_all_below_5_percent": payload[
                          "primary_all_below_5_percent"]}, sort_keys=True))


if __name__ == "__main__":
    main()
