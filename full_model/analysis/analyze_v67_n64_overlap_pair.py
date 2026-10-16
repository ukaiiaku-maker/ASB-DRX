#!/usr/bin/env python3
"""Classify matched adiabatic/bath n64 RX-band coexistence continuations."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from full_model.analysis.run_v58_three_grain_production import _load_checkpoint


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _case(directory):
    result_path = directory/"result.json"
    checkpoint_path = directory/"checkpoint_latest.npz"
    result = json.loads(result_path.read_text())
    state, _, step, load, _, configuration, _ = _load_checkpoint(
        checkpoint_path)
    temperature = np.sum(
        np.asarray(state.supports)*np.stack([
            np.asarray(owner.temperature_K) for owner in state.owners]),
        axis=0)
    growing = {
        grain: row for grain, row in result["owner_results"].items()
        if row["net_area_change_m2"] > 0.0}
    decision = result["decision"]
    hard = bool(
        result["source_tree_clean"]
        and decision["all_hard_absorption_balances_passed"]
        and decision["mura_hard_balance_passed"]
        and decision["first_law_hard_balance_passed"]
        and decision["grain_ids_unchanged"])
    return {
        "directory": str(directory.resolve()),
        "result_sha256": _sha(result_path),
        "checkpoint_sha256": _sha(checkpoint_path),
        "source_commit": result["source_commit"],
        "source_tree_clean": result["source_tree_clean"],
        "parent_checkpoint_sha256": result["parent_checkpoint_sha256"],
        "step": int(step),
        "load": float(load),
        "thermal_control": configuration["thermal_control"],
        "thermal_bath_rate_s": float(configuration.get(
            "thermal_bath_rate_s", 0.0)),
        "front_transmission_fraction": decision[
            "front_transmission_fraction"],
        "hard_audits_passed": hard,
        "initial_power_localization": result["initial_power_localization"],
        "final_power_localization": result["final_power_localization"],
        "active_band_intervals": result["active_band_intervals"],
        "temperature_mean_K": float(np.mean(temperature)),
        "temperature_range_K": float(np.ptp(temperature)),
        "growing_owners": growing,
        "decision": decision,
        "hard_balance_maxima": result["hard_balance_maxima"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("adiabatic", type=Path)
    parser.add_argument("bath", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    adiabatic = _case(args.adiabatic)
    bath = _case(args.bath)
    matched_endpoint = bool(
        adiabatic["step"] == bath["step"]
        and adiabatic["load"] == bath["load"])
    same_transmission = (
        adiabatic["front_transmission_fraction"]
        == bath["front_transmission_fraction"])
    heat_controls_valid = bool(
        adiabatic["thermal_control"] == "adiabatic"
        and adiabatic["thermal_bath_rate_s"] == 0.0
        and bath["thermal_control"] == "strong_bath"
        and bath["thermal_bath_rate_s"] > 0.0)
    both_banded = bool(
        adiabatic["decision"]["final_connected_band"]
        and bath["decision"]["final_connected_band"])
    both_overlap = bool(
        adiabatic["decision"]["growing_owner_intersects_active_band"]
        and bath["decision"]["growing_owner_intersects_active_band"])
    hard = adiabatic["hard_audits_passed"] and bath["hard_audits_passed"]
    thermal_amplification = bool(
        adiabatic["temperature_range_K"] > bath["temperature_range_K"]
        and adiabatic["final_power_localization"]["maximum_to_mean"]
        > bath["final_power_localization"]["maximum_to_mean"])
    coexistence = bool(
        matched_endpoint and same_transmission and heat_controls_valid and both_banded
        and both_overlap and hard)
    payload = {
        "schema": "v67-n64-matched-rx-band-overlap-pair-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cases": {"adiabatic": adiabatic, "strong_bath": bath},
        "decision": {
            "matched_endpoint": matched_endpoint,
            "same_fixed_transmission_fraction": same_transmission,
            "heat_control_rates_valid": heat_controls_valid,
            "all_hard_audits_passed": hard,
            "both_controls_form_connected_power_bands": both_banded,
            "both_controls_have_RX_growth_intersecting_active_band": (
                both_overlap),
            "adiabatic_temperature_amplification": bool(
                adiabatic["temperature_range_K"]
                > bath["temperature_range_K"]),
            "adiabatic_power_localization_amplification": bool(
                adiabatic["final_power_localization"]["maximum_to_mean"]
                > bath["final_power_localization"]["maximum_to_mean"]),
            "thermal_amplification": thermal_amplification,
            "strict_heat_triggered_ASB": False,
            "RX_band_coexistence": coexistence,
            "classification": (
                "QUALIFIED_RX_IN_MECHANICAL_BAND_WITH_THERMAL_AMPLIFICATION"
                if coexistence and thermal_amplification else
                ("QUALIFIED_RX_IN_MECHANICAL_BAND"
                 if coexistence else "RX_BAND_PAIR_NOT_QUALIFIED")),
        },
        "claim_boundary": (
            "This matched n64 pair demonstrates existing-boundary RX growth "
            "intersecting an active connected power band. Because the strong-"
            "bath control also bands and the onset is not spatially refined "
            "here, it is a mechanically formed shear-band coexistence result "
            "with thermal amplification, not a strict heat-triggered ASB or "
            "spontaneous-grain-birth claim."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
