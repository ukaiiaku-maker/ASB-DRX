#!/usr/bin/env python3
"""Classify the same-geometry V41 frozen-flow thermal control."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from full_model.analysis.run_v37_conduction_localization import endpoint


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--control-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = endpoint(args.baseline)
    control = endpoint(args.control)
    ledger = json.loads(args.control_ledger.read_text())
    bp = baseline["work_conjugate_plastic_power"]
    cp = control["work_conjugate_plastic_power"]
    bh = baseline["irreversible_heat_production"]
    ch = control["irreversible_heat_production"]
    bt = baseline["temperature"]
    ct = control["temperature"]
    same_geometry = all(
        baseline["parameters"][key] == control["parameters"][key]
        for key in ("grid", "domain_length_m", "conductivity_W_m_K",
                    "strain_rate_s", "initial_temperature_K",
                    "particle_radius_m", "heat_process_zone_sigma_m"))
    result = {
        "schema": "asb-drx/v41/thermal-response-decision/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": baseline, "frozen_flow_control": control,
        "same_geometry_and_transport": same_geometry,
        "intervention": "flow rates evaluate at T0 while heat and recovery temperature evolve",
        "causal_effect_control_minus_full": {
            "temperature_peak_K": ct["maximum"]-bt["maximum"],
            "temperature_peak_minus_mean_K": (
                ct["peak_minus_mean"]-bt["peak_minus_mean"]),
            "plastic_power_participation_fraction": (
                cp["inverse_participation_fraction"]
                -bp["inverse_participation_fraction"]),
            "heat_participation_fraction": (
                ch["inverse_participation_fraction"]
                -bh["inverse_participation_fraction"]),
            "plastic_power_minor_fwhm_m": (
                cp["second_moment_widths"]["minor_gaussian_fwhm_m"]
                -bp["second_moment_widths"]["minor_gaussian_fwhm_m"]),
        },
        "control_trajectory_ledger": {
            "path": str(args.control_ledger.resolve()),
            "sha256": digest(args.control_ledger),
            "classification": ledger["classification"],
            "accepted_steps": ledger["cumulative"]["accepted_steps"],
            "relative_first_law_residual": ledger[
                "relative_first_law_residual"],
            "all_channels_available": ledger["all_channels_available"],
            "numerical_constraints_in_physical_energy": ledger[
                "numerical_constraints_in_physical_energy"],
            "numerical_constraint_drives_physical_state": ledger[
                "numerical_constraint_drives_physical_state"],
        },
        "strict_asb_claimed": False,
        "classification": (
            "MATCHED_CAUSAL_CONTROL_COMPLETE_BROAD_OR_NONPERSISTENT"
            if same_geometry else "THERMAL_CONTROL_GEOMETRY_MISMATCH"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
