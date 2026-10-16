#!/usr/bin/env python3
"""Convergence classification for the V67 prepared ASB discriminator."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def _case(path):
    root = Path(path)
    result = json.loads((root/"result.json").read_text())
    history = json.loads((root/"history.json").read_text())
    active = [row for row in history if row["power_localization"]["band_like"]]
    return {
        "path": str(root),
        "source_commit": result["source_commit"],
        "source_tree_clean": result["source_tree_clean"],
        "configuration": result["configuration"],
        "initial_state_digest": result["initial_state_digest"],
        "result": result["result"],
        "final_power_localization": result["final_power_localization"],
        "band_onset_strain": (None if not active else
                              active[0]["applied_shear_strain"]),
        "band_onset_time_s": (None if not active else
                              active[0]["physical_time_s"]),
    }


def _relative(a, b):
    return abs(float(a)-float(b))/max(abs(float(b)), 1e-300)


def classify(coarse_ad, coarse_bath, fine_ad, fine_bath, half_ad, half_bath):
    cases = {
        "n32_dt2_ad": coarse_ad, "n32_dt2_bath": coarse_bath,
        "n48_dt2_ad": fine_ad, "n48_dt2_bath": fine_bath,
        "n32_dt1_ad": half_ad, "n32_dt1_bath": half_bath,
    }
    ad_metrics = (
        "final_temperature_contrast_K", "post_peak_stress_drop_fraction")
    grid_errors = {
        key: _relative(coarse_ad["result"][key], fine_ad["result"][key])
        for key in ad_metrics}
    timestep_errors = {
        key: _relative(coarse_ad["result"][key], half_ad["result"][key])
        for key in ad_metrics}
    grid_errors["final_power_maximum_to_mean"] = _relative(
        coarse_ad["final_power_localization"]["maximum_to_mean"],
        fine_ad["final_power_localization"]["maximum_to_mean"])
    timestep_errors["final_power_maximum_to_mean"] = _relative(
        coarse_ad["final_power_localization"]["maximum_to_mean"],
        half_ad["final_power_localization"]["maximum_to_mean"])
    grid_errors["band_onset_strain"] = _relative(
        coarse_ad["band_onset_strain"], fine_ad["band_onset_strain"])
    timestep_errors["band_onset_strain"] = _relative(
        coarse_ad["band_onset_strain"], half_ad["band_onset_strain"])
    all_sources_clean = all(case["source_tree_clean"] for case in cases.values())
    all_adiabatic_band = all(case["final_power_localization"]["band_like"]
                              for case in (coarse_ad, fine_ad, half_ad))
    no_bath_band = all(not case["final_power_localization"]["band_like"]
                       for case in (coarse_bath, fine_bath, half_bath))
    residuals_pass = all(
        case["result"]["maximum_mechanical_relative_first_law_residual"] < 1e-3
        for case in cases.values())
    convergence = (max(grid_errors.values()) < .05
                   and max(timestep_errors.values()) < .05)
    same_physical_horizon = all(abs(
        case["configuration"]["initial_shear_strain"]
        + case["configuration"]["shear_rate_s"]
        * case["configuration"]["dt_s"]
        * case["configuration"]["target_steps"] - .124) < 1e-12
        for case in cases.values())
    passed = all((all_sources_clean, all_adiabatic_band, no_bath_band,
                  residuals_pass, convergence, same_physical_horizon))
    return {
        "schema": "v67-single-crystal-asb-refinement-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cases": cases,
        "relative_errors": {
            "n32_to_n48_fixed_physical_width": grid_errors,
            "dt2ns_to_dt1ns_fixed_horizon": timestep_errors,
            "provisional_threshold": .05,
        },
        "audits": {
            "all_source_trees_clean": all_sources_clean,
            "adiabatic_band_at_all_resolutions": all_adiabatic_band,
            "no_strong_bath_band_at_any_resolution": no_bath_band,
            "mechanical_first_law_residuals_below_1e-3": residuals_pass,
            "grid_and_timestep_errors_below_5_percent": convergence,
            "same_terminal_strain_0p124": same_physical_horizon,
        },
        "decision": {
            "strict_prepared_asb_qualified": passed,
            "classification": (
                "QUALIFIED_HEAT_TRIGGERED_SINGLE_CRYSTAL_ASB"
                if passed else "NOT_QUALIFIED"),
            "zero_imperfection_instability_claimed": False,
            "material_calibration_claimed": False,
        },
        "claim_boundary": (
            "Grid- and timestep-qualified heat-triggered localization from a "
            "fixed 20 K mean-zero prepared imperfection. This is a physical "
            "response-family result, not a zero-imperfection bifurcation or a "
            "calibration to a named material."),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("coarse_ad", type=Path)
    parser.add_argument("coarse_bath", type=Path)
    parser.add_argument("fine_ad", type=Path)
    parser.add_argument("fine_bath", type=Path)
    parser.add_argument("half_ad", type=Path)
    parser.add_argument("half_bath", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    payload = classify(*[_case(getattr(args, name)) for name in (
        "coarse_ad", "coarse_bath", "fine_ad", "fine_bath",
        "half_ad", "half_bath")])
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))
    print(json.dumps(payload["relative_errors"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
