#!/usr/bin/env python3
"""Subtract the no-source trajectory from matched V66 source-location cases."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from full_model.analysis.run_v58_three_grain_production import _load_checkpoint
from full_model.production.arrhenius_dislocation_sources import (
    boundary_tj_site_field, randomized_matched_site_field,
)
from full_model.production.multigrain_common_state import reconstruct_multigrain_common


ROOT = Path(__file__).resolve().parents[2]
VERIFY = ROOT/"full_model"/"verification"
CASES = {
    "disabled": VERIFY/"v66_coupled_source_disabled_v2",
    "gb_tj_only": VERIFY/"v66_coupled_source_gb_v2",
    "matched_random": VERIFY/"v66_coupled_source_random_v2",
}
OUT = VERIFY/"v66_source_location_common_comparison.json"


def _corr(a, b):
    aa = np.ravel(a)-np.mean(a); bb = np.ravel(b)-np.mean(b)
    scale = np.sqrt(np.sum(aa*aa)*np.sum(bb*bb))
    return 0.0 if scale == 0.0 else float(np.sum(aa*bb)/scale)


def _spectrum_peak(field):
    value = np.asarray(field)-np.mean(field)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(value)))**2
    center = tuple(n//2 for n in value.shape)
    spectrum[center] = 0.0
    peak = np.unravel_index(np.argmax(spectrum), spectrum.shape)
    return float(np.hypot(peak[0]-center[0], peak[1]-center[1]))


def _fields(directory):
    state, runtime, step, load, *_ = _load_checkpoint(
        directory/"checkpoint_latest.npz")
    result = json.loads((directory/"result.json").read_text())
    spacing = result["configuration"]["length_m"]/result["configuration"]["n"]
    common, _ = reconstruct_multigrain_common(state, spacing)
    mobile = np.sum(common.mobile_plus_m2+common.mobile_minus_m2, axis=2)
    total = sum(np.sum(np.asarray(getattr(common, name)), axis=2) for name in (
        "mobile_plus_m2", "mobile_minus_m2", "forest_plus_m2",
        "forest_minus_m2", "wall_plus_m2", "wall_minus_m2"))
    nye = np.sum(common.family_nye_m1, axis=2)
    return state, runtime, result, {
        "mobile_density_m2": mobile,
        "total_density_m2": total,
        "nye_m1": nye,
        "temperature_K": np.asarray(common.temperature_K),
        "beta_p": np.asarray(common.beta_p),
        "supports": np.asarray(state.supports),
    }


def main():
    loaded = {name: _fields(path) for name, path in CASES.items()}
    baseline = loaded["disabled"][3]
    boundary = boundary_tj_site_field(
        loaded["disabled"][0].supports, tj_site_weight=2.0)
    random = randomized_matched_site_field(boundary, 66)
    sites = {"gb_tj_only": boundary, "matched_random": random}
    metrics = {}
    deltas = {}
    for name in ("gb_tj_only", "matched_random"):
        _, runtime, result, fields = loaded[name]
        delta = {key: fields[key]-baseline[key] for key in fields}
        deltas[name] = delta
        density = delta["total_density_m2"]
        mobile = delta["mobile_density_m2"]
        metrics[name] = {
            "source_event_count": runtime.ledger.source_event_count,
            "created_line_cell_sum_m2": runtime.ledger.source_created_line_cell_sum_m2,
            "source_generated_heat_J": runtime.ledger.cumulative_source_generated_heat_J,
            "maximum_source_relative_first_law_residual": (
                runtime.ledger.maximum_source_relative_first_law_residual),
            "total_density_delta_mean_m2": float(np.mean(density)),
            "total_density_delta_rms_m2": float(np.sqrt(np.mean(density*density))),
            "mobile_density_delta_mean_m2": float(np.mean(mobile)),
            "density_delta_boundary_correlation": _corr(density, boundary),
            "density_delta_own_site_correlation": _corr(density, sites[name]),
            "density_delta_structure_factor_peak_index": _spectrum_peak(density),
            "nye_delta_rms_m1": float(np.sqrt(np.mean(delta["nye_m1"]**2))),
            "beta_p_delta_rms": float(np.sqrt(np.mean(delta["beta_p"]**2))),
            "temperature_delta_mean_K": float(np.mean(delta["temperature_K"])),
            "temperature_delta_rms_K": float(np.sqrt(np.mean(
                delta["temperature_K"]**2))),
            "support_delta_rms": float(np.sqrt(np.mean(delta["supports"]**2))),
            "stress_delta_from_disabled_Pa": (
                result["result"]["peak_stress_Pa"]
                -loaded["disabled"][2]["result"]["peak_stress_Pa"]),
            "target_growth_delta_from_disabled": (
                result["result"]["final_target_growth_fraction"]
                -loaded["disabled"][2]["result"]["final_target_growth_fraction"]),
        }
    between_density = (deltas["gb_tj_only"]["total_density_m2"]
                       -deltas["matched_random"]["total_density_m2"])
    baseline_rms = float(np.sqrt(np.mean(baseline["total_density_m2"]**2)))
    payload = {
        "schema": "v66-common-source-location-comparison-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": "6b104d5",
        "protocol": {
            "grid": [16, 16], "temperature_K": 1000.0,
            "shear_rate_s": 4e4, "terminal_strain": .04,
            "source_site_control": (
                "matched_random is an exact permutation of current GB/TJ weights"),
            "source_amplitude": "geometric 1/d^2 line per sign and b/d slip",
            "retuning": False,
        },
        "cases": metrics,
        "comparison": {
            "source_event_count_relative_difference": float(abs(
                metrics["gb_tj_only"]["source_event_count"]
                -metrics["matched_random"]["source_event_count"])/max(
                    metrics["gb_tj_only"]["source_event_count"], 1e-300)),
            "gb_minus_random_density_delta_rms_m2": float(np.sqrt(np.mean(
                between_density*between_density))),
            "gb_minus_random_density_delta_relative_to_baseline_rms": float(
                np.sqrt(np.mean(between_density*between_density))/baseline_rms),
            "macroscopic_location_effect_detected": bool(
                abs(metrics["gb_tj_only"]["stress_delta_from_disabled_Pa"]
                    -metrics["matched_random"]["stress_delta_from_disabled_Pa"])
                > 1e6),
            "spatial_location_effect_detected": bool(
                np.sqrt(np.mean(between_density*between_density)) > 0.0),
            "physical_wall_claim": False,
            "interpretation": (
                "The energy-complete geometric source produces a resolvable spatial "
                "perturbation but its location has negligible macroscopic effect over "
                "4% shear; it is not a wall/nucleation forcing mechanism."),
        },
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload["comparison"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
