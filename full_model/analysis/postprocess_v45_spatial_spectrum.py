#!/usr/bin/env python3
"""Band-resolved spectrum of the selected V45 n128/n192 trajectory."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from full_model.analysis.run_v43_spatial_first_difference import fields


ROOT = Path("full_model/production/results-local/v45-current-final/sub8")


def spectrum(value):
    n = value.shape[0]
    return np.fft.fftshift(
        np.fft.fftn(value, axes=(0, 1))/(n*n), axes=(0, 1))


def crop(value, half):
    center = value.shape[0]//2
    return value[center-half:center+half+1, center-half:center+half+1]


def rms(value):
    return float(np.sqrt(np.mean(np.abs(value)**2)))


def main():
    stages = ("initial", "after_first_mura", "after_front", "after_second_mura")
    rows = {}
    for stage in stages:
        a = fields(ROOT/"n128"/f"{stage}.npz", 128)[0]["curl_nye"]
        b = fields(ROOT/"n192"/f"{stage}.npz", 192)[0]["curl_nye"]
        sa = spectrum(a); sb = spectrum(b)
        total_a = float(np.sum(np.abs(sa)**2, dtype=np.longdouble))
        total_b = float(np.sum(np.abs(sb)**2, dtype=np.longdouble))
        bands = []
        for half in (8, 16, 24, 32, 48, 63):
            ca = crop(sa, half); cb = crop(sb, half)
            scale = max(rms(ca), rms(cb), 1e-300)
            bands.append({
                "half_width": half, "mode_count_per_axis": 2*half+1,
                "maximum_abs_wave_index": half,
                "complex_relative_rms": rms(ca-cb)/scale,
                "n128_retained_spectral_energy_fraction": float(
                    np.sum(np.abs(ca)**2, dtype=np.longdouble)
                    /max(total_a, 1e-300)),
                "n192_retained_spectral_energy_fraction": float(
                    np.sum(np.abs(cb)**2, dtype=np.longdouble)
                    /max(total_b, 1e-300)),
            })
        rows[stage] = {
            "whole_field_rms": {"n128": rms(a), "n192": rms(b)},
            "bands": bands,
        }
    output_payload = {
        "schema": "asb-drx/v45/spatial-spectrum/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "fft_normalization": "complex coefficient divided by n^2",
        "coordinate_domain_m": [0.0, 3.2e-6],
        "field": "curl_nye from spectral Curl(beta_p)",
        "stages": rows,
        "interpretation": (
            "band growth identifies whether common low modes or unresolved "
            "near-grid content dominates; no output filter is applied to the "
            "production state or pass/fail strong norm"),
    }
    output = Path("full_model/verification/v45_spatial_spectrum.json")
    output.write_text(json.dumps(output_payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps({
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "final_bands": rows["after_second_mura"]["bands"]}, sort_keys=True))


if __name__ == "__main__":
    main()
