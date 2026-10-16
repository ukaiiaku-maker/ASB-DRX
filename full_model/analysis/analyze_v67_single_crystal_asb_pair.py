#!/usr/bin/env python3
"""Classify a matched single-crystal adiabatic/strong-bath pair."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def _load(path):
    return json.loads((Path(path)/"result.json").read_text())


def _matched_configuration(a, b):
    left = dict(a); right = dict(b)
    for value in (left, right):
        value.pop("thermal_control", None)
        value.pop("thermal_bath_rate_s", None)
    return left == right


def classify(adiabatic, bath):
    matched = _matched_configuration(
        adiabatic["configuration"], bath["configuration"])
    same_initial = (
        adiabatic["initial_state_digest"] == bath["initial_state_digest"])
    initially_subcritical = all(
        case["initial_power_localization"]["maximum_to_mean"] < 2.0
        and not case["initial_power_localization"]["band_like"]
        for case in (adiabatic, bath))
    hard_audits = all(
        case["result"]["maximum_mechanical_relative_first_law_residual"] < 1e-3
        and case["result"]["grain_ids_unchanged"]
        and case["source_tree_clean"]
        for case in (adiabatic, bath))
    ad_band = bool(adiabatic["final_power_localization"]["band_like"])
    bath_band = bool(bath["final_power_localization"]["band_like"])
    persistent = adiabatic["result"][
        "terminal_band_persistence_intervals"] >= 20
    thermal_amplification = (
        adiabatic["result"]["final_temperature_contrast_K"]
        > adiabatic["initial_temperature"]["contrast_K"]
        and bath["result"]["final_temperature_contrast_K"] < 1e-6)
    thermal_softening = (
        adiabatic["result"]["post_peak_stress_drop_fraction"] >= .05
        and adiabatic["result"]["post_peak_stress_drop_fraction"]
        > bath["result"]["post_peak_stress_drop_fraction"])
    causal = all((matched, same_initial, initially_subcritical, hard_audits,
                  ad_band, not bath_band, persistent,
                  thermal_amplification, thermal_softening))
    return {
        "schema": "v67-single-crystal-asb-pair-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cases": {"adiabatic": adiabatic, "strong_bath": bath},
        "audits": {
            "matched_configuration_except_heat_extraction": matched,
            "exact_same_initial_state": same_initial,
            "initial_seed_subcritical": initially_subcritical,
            "hard_audits_passed": hard_audits,
            "adiabatic_terminal_band": ad_band,
            "strong_bath_terminal_band": bath_band,
            "adiabatic_terminal_persistence_at_least_20_intervals": persistent,
            "temperature_perturbation_amplified_only_adiabatically": (
                thermal_amplification),
            "adiabatic_post_peak_softening_exceeds_bath": thermal_softening,
        },
        "decision": {
            "heat_triggered_single_crystal_asb_candidate": causal,
            "classification": (
                "QUALIFIED_HEAT_TRIGGERED_SINGLE_CRYSTAL_ASB_CANDIDATE"
                if causal else "NOT_QUALIFIED"),
            "strict_asb_qualified": False,
            "remaining_qualification": (
                "grid/timestep refinement and seed-amplitude sensitivity; "
                "this pair establishes the causal mechanism at one grid"),
        },
        "claim_boundary": (
            "One n-grid prepared single-crystal pair with a finite mean-zero "
            "thermal imperfection. The pair isolates heat feedback but does not "
            "claim zero-imperfection instability, grid convergence, or material "
            "calibration."),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("adiabatic", type=Path)
    parser.add_argument("strong_bath", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    payload = classify(_load(args.adiabatic), _load(args.strong_bath))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
