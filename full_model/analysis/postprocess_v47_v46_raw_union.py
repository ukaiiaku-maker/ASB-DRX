#!/usr/bin/env python3
"""Direct raw-array spectral-union audit of the retained V46 endpoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from full_model.analysis.run_v43_spatial_first_difference import fields


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized_spectrum(value):
    n = value.shape[0]
    return np.fft.fftshift(
        np.fft.fftn(value, axes=(0, 1))/(n*n), axes=(0, 1))


def spectral_rms(coefficients):
    trailing = int(np.prod(coefficients.shape[2:])) if coefficients.ndim > 2 else 1
    return float(np.sqrt(np.sum(np.abs(coefficients)**2)/trailing))


def union_comparison(coarse, fine):
    if coarse.shape[:2] != (128, 128) or fine.shape[:2] != (192, 192):
        raise ValueError("V47 union audit requires the retained n128/n192 pair")
    ca = normalized_spectrum(coarse)
    cb = normalized_spectrum(fine)
    embedded = np.zeros_like(cb)
    # For an even grid the shifted coarse indices are [-64,63].  The -64
    # Nyquist row/column have no unambiguous signed counterpart under a finer
    # real-grid embedding, so retain the common non-Nyquist square [-63,63]
    # and report the excluded coarse Nyquist content independently.
    embedded[96-63:96+64, 96-63:96+64] = ca[64-63:64+64, 64-63:64+64]
    shared_fine = np.zeros_like(cb)
    shared_fine[96-63:96+64, 96-63:96+64] = cb[
        96-63:96+64, 96-63:96+64]
    coarse_nyquist = ca.copy()
    coarse_nyquist[64-63:64+64, 64-63:64+64] = 0.0
    fine_tail = cb-shared_fine
    fine_rms = spectral_rms(cb)
    shared_difference = spectral_rms(embedded-shared_fine)
    union_difference = spectral_rms(embedded-cb)
    return {
        "coarse_physical_rms": float(np.sqrt(np.mean(np.asarray(coarse)**2))),
        "fine_physical_rms": float(np.sqrt(np.mean(np.asarray(fine)**2))),
        "coarse_spectral_rms": spectral_rms(ca),
        "fine_spectral_rms": fine_rms,
        "fine_shared_spectral_rms": spectral_rms(shared_fine),
        "fine_tail_spectral_rms": spectral_rms(fine_tail),
        "fine_tail_squared_norm_fraction": (
            spectral_rms(fine_tail)**2/max(fine_rms**2, 1e-300)),
        "fine_tail_rms_fraction": spectral_rms(fine_tail)/max(fine_rms, 1e-300),
        "coarse_excluded_nyquist_spectral_rms": spectral_rms(coarse_nyquist),
        "coarse_excluded_nyquist_rms_fraction": (
            spectral_rms(coarse_nyquist)/max(spectral_rms(ca), 1e-300)),
        "shared_difference_spectral_rms": shared_difference,
        "shared_difference_relative_to_fine_full_rms": (
            shared_difference/max(fine_rms, 1e-300)),
        "zero_extension_union_difference_spectral_rms": union_difference,
        "zero_extension_union_difference_relative_to_fine_full_rms": (
            union_difference/max(fine_rms, 1e-300)),
        "pythagorean_union_identity_relative_residual": abs(
            union_difference**2-shared_difference**2
            -spectral_rms(fine_tail)**2)/max(union_difference**2, 1e-300),
        "shared_mode_indices_per_axis": [-63, 63],
        "coordinate_shift_optimized": False,
        "interpretation": (
            "difference between two discrete reconstructions after zero-extending "
            "the non-Nyquist n128 spectrum; not error against an exact solution"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v46-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = {n: args.v46_root/f"n{n}" for n in (128, 192)}
    manifests = {
        n: json.loads((roots[n]/"run_manifest.json").read_text())
        for n in roots}
    for n, manifest in manifests.items():
        if (manifest["terminal_state"] != "COMPLETE"
                or manifest["completed_operation_count"] != 119
                or abs(manifest["physical_time_s"]-62.5e-6) > 1e-18):
            raise ValueError(f"retained n{n} V46 continuation is incomplete")
        checkpoint = Path(manifest["latest_checkpoint"])
        if digest(checkpoint) != manifest["latest_checkpoint_sha256"]:
            raise ValueError(f"retained n{n} checkpoint checksum mismatch")
    values = {}
    metadata = {}
    for n in roots:
        values[n], metadata[n], _ = fields(
            Path(manifests[n]["latest_checkpoint"]), n)
    result = union_comparison(values[128]["curl_nye"], values[192]["curl_nye"])
    payload = {
        "schema": "asb-drx/v47/v46-raw-spectral-union/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "v46_root": str(args.v46_root.resolve()),
        "scientific_source_sha": manifests[128]["configuration"][
            "parent_scientific_source_sha"],
        "continuation_source_sha": manifests[128]["configuration"][
            "continuation_source_sha"],
        "physical_time_s": 62.5e-6,
        "field": "curl_nye from compatible Curl(beta_p)",
        "fft_normalization": "complex coefficients divided by n^2",
        "component_norm": "Parseval L2 with mean over trailing tensor components",
        "members": {
            str(n): {
                "manifest_sha256": digest(roots[n]/"run_manifest.json"),
                "checkpoint_sha256": manifests[n]["latest_checkpoint_sha256"],
                "metadata_physical_time_s": metadata[n]["physical_time_s"],
            } for n in roots},
        "raw_array_comparison": result,
        "classification": (
            "RAW_UNION_SHORT_WAVE_DIFFERENCE_CONFIRMED"
            if result["zero_extension_union_difference_relative_to_fine_full_rms"] > .05
            else "RAW_UNION_DIFFERENCE_WITHIN_FIVE_PERCENT"),
        "exact_continuum_error_claimed": False,
        "thermodynamic_energy_fraction_claimed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps({
        "classification": payload["classification"],
        "fine_tail_squared_norm_fraction": result[
            "fine_tail_squared_norm_fraction"],
        "union_relative_difference": result[
            "zero_extension_union_difference_relative_to_fine_full_rms"],
        "sha256": digest(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
