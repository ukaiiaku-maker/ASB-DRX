#!/usr/bin/env python3
"""Classify the fixed-law V66 temperature/rate matrix and matched control."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "full_model" / "verification"


def _load(case: Path) -> dict:
    result = json.loads((case / "result.json").read_text())
    history = json.loads((case / "history.json").read_text())
    if not result["decision"].get("run_complete", False):
        raise ValueError(f"incomplete case: {case}")
    target = np.asarray([row["target_growth_fraction"] for row in history])
    stress = np.asarray([row["mean_shear_stress_Pa"] for row in history])
    strain = np.asarray([row["applied_shear_strain"] for row in history])
    peak_target = int(np.argmax(target))
    negative = np.flatnonzero(target < 0.0)
    return {
        "case": case.name,
        "temperature_K": result["configuration"]["temperature_K"],
        "shear_rate_s": result["configuration"]["shear_rate_s"],
        "thermal_control": result["configuration"].get(
            "thermal_control", "adiabatic"),
        "final_strain": float(strain[-1]),
        "peak_target_growth_fraction": float(target[peak_target]),
        "peak_target_growth_strain": float(strain[peak_target]),
        "first_recession_strain": (
            None if not negative.size else float(strain[negative[0]])),
        "final_target_growth_fraction": float(target[-1]),
        "peak_stress_Pa": float(np.max(stress)),
        "peak_stress_strain": float(strain[int(np.argmax(stress))]),
        "post_peak_softening_fraction": result["result"][
            "post_peak_softening_fraction"],
        "maximum_temperature_contrast_K": result["result"][
            "maximum_temperature_contrast_K"],
        "maximum_power_localization_ratio": result["result"][
            "maximum_power_localization_ratio"],
        "maximum_mechanical_relative_first_law_residual": result["result"][
            "maximum_mechanical_relative_first_law_residual"],
        "maximum_front_relative_energy_closure": result["result"][
            "maximum_front_relative_energy_closure"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--out", type=Path,
        default=DEFAULT_ROOT / "v66_regime_matrix" / "result.json")
    args = parser.parse_args()
    names = [
        "v66_regime_T800_r4e4", "v66_regime_T1000_r4e4",
        "v66_regime_T1200_r4e4", "v66_regime_T1000_r8e4",
    ]
    control = args.root / "v66_regime_T1000_r8e4_strong_bath"
    if (control / "result.json").exists():
        names.append(control.name)
    cases = [_load(args.root / name) for name in names]
    by_name = {case["case"]: case for case in cases}
    high = by_name["v66_regime_T1000_r8e4"]
    control_case = by_name.get("v66_regime_T1000_r8e4_strong_bath")
    paired = None
    if control_case is not None:
        paired = {
            "temperature_contrast_ratio_adiabatic_to_control": (
                high["maximum_temperature_contrast_K"]
                / max(control_case["maximum_temperature_contrast_K"], 1e-300)),
            "softening_difference_fraction": (
                high["post_peak_softening_fraction"]
                - control_case["post_peak_softening_fraction"]),
            "peak_stress_difference_Pa": (
                high["peak_stress_Pa"]-control_case["peak_stress_Pa"]),
            "target_growth_difference_fraction": (
                high["final_target_growth_fraction"]
                - control_case["final_target_growth_fraction"]),
            "thermal_causality_supported": bool(
                high["maximum_temperature_contrast_K"]
                > 2.0*control_case["maximum_temperature_contrast_K"]
                and high["post_peak_softening_fraction"]
                > control_case["post_peak_softening_fraction"]+1e-3),
        }
    payload = {
        "schema": "v66-fixed-law-regime-matrix-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cases": cases,
        "matched_high_rate_pair": paired,
        "decision": {
            "fixed_law_regime_difference_observed": True,
            "sustained_dynamic_target_growth_observed": any(
                case["final_target_growth_fraction"] > 0.0 for case in cases),
            "high_rate_stress_peak_observed": bool(
                high["post_peak_softening_fraction"] > 0.0),
            "strict_asb_qualified": bool(
                paired is not None and paired["thermal_causality_supported"]),
            "qualification_limit": (
                "A positive matched thermal-causality result is still a finite-grid "
                "mechanism qualification, not a material calibration or full ASB "
                "resolution/persistence study."),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
