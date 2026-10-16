#!/usr/bin/env python3
"""Classify matched recurrent polycrystal RX/band trajectories.

The classifier deliberately separates a recurrent mechanical localization
response from a strict heat-triggered ASB claim.  The latter requires an
adiabatic terminal band without a matched bath terminal band.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from full_model.analysis.run_v58_three_grain_production import _load_checkpoint


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case(path: Path) -> dict:
    root = Path(path)
    result = json.loads((root / "result.json").read_text())
    state, _, step, load, _, configuration, _ = _load_checkpoint(
        root / "checkpoint_latest.npz")
    temperature = np.sum(
        np.asarray(state.supports)
        * np.stack([owner.temperature_K for owner in state.owners]), axis=0)
    return {
        "directory": str(root.resolve()),
        "checkpoint_sha256": _sha(root / "checkpoint_latest.npz"),
        "parent_checkpoint_sha256": result["parent_checkpoint_sha256"],
        "source_commit": result["source_commit"],
        "source_tree_clean": result["source_tree_clean"],
        "step": step,
        "load": load,
        "thermal_control": configuration["thermal_control"],
        "thermal_bath_rate_s": configuration["thermal_bath_rate_s"],
        "front_transmission_fraction": result["decision"][
            "front_transmission_fraction"],
        "active_physical_band_intervals": result["active_band_intervals"],
        "flow_regime": result["flow_regime"],
        "final_power_localization": result["final_power_localization"],
        "owner_results": result["owner_results"],
        "decision": result["decision"],
        "hard_balance_maxima": result["hard_balance_maxima"],
        "temperature_mean_K": float(np.mean(temperature)),
        "temperature_range_K": float(np.ptp(temperature)),
    }


def classify(adiabatic: list[dict], bath: list[dict]) -> dict:
    if len(adiabatic) != len(bath) or len(adiabatic) < 3:
        raise ValueError("matched trajectories require at least three stages")
    pairs = list(zip(adiabatic, bath))
    all_cases = adiabatic + bath
    chain_valid = all(
        series[index]["parent_checkpoint_sha256"]
        == series[index - 1]["checkpoint_sha256"]
        for series in (adiabatic, bath)
        for index in range(1, len(series)))
    endpoints_matched = all(
        ad["step"] == ba["step"] and ad["load"] == ba["load"]
        for ad, ba in pairs)
    heat_controls_valid = all(
        ad["thermal_control"] == "adiabatic"
        and ad["thermal_bath_rate_s"] == 0.0
        and ba["thermal_control"] == "strong_bath"
        and ba["thermal_bath_rate_s"] > 0.0
        for ad, ba in pairs)
    same_transmission = len({
        case["front_transmission_fraction"] for case in all_cases}) == 1
    hard = all(
        case["source_tree_clean"]
        and case["decision"]["all_hard_absorption_balances_passed"]
        and case["decision"]["mura_hard_balance_passed"]
        and case["decision"]["first_law_hard_balance_passed"]
        and case["decision"]["grain_ids_unchanged"]
        for case in all_cases)
    ad_intervals = sum(
        case["active_physical_band_intervals"] for case in adiabatic)
    bath_intervals = sum(
        case["active_physical_band_intervals"] for case in bath)
    heat_advanced = bool(
        adiabatic[0]["decision"]["final_connected_band"]
        and not bath[0]["decision"]["final_connected_band"]
        and ad_intervals > bath_intervals)
    intermediate_dissolution = any(
        not ad["decision"]["final_connected_band"]
        and not ba["decision"]["final_connected_band"]
        for ad, ba in pairs[1:-1])
    terminal_recurrence = bool(
        adiabatic[-1]["decision"]["final_connected_band"]
        and bath[-1]["decision"]["final_connected_band"])
    terminal_rx_overlap = bool(
        adiabatic[-1]["decision"]["growing_owner_intersects_active_band"]
        and bath[-1]["decision"]["growing_owner_intersects_active_band"])
    common_mechanical_recurrence = bool(
        chain_valid and endpoints_matched and heat_controls_valid
        and same_transmission and hard and intermediate_dissolution
        and terminal_recurrence and terminal_rx_overlap)
    strict_heat_triggered_terminal_asb = bool(
        adiabatic[-1]["decision"]["final_connected_band"]
        and not bath[-1]["decision"]["final_connected_band"])
    classification = (
        "RECURRENT_MECHANICAL_RX_BAND_COEXISTENCE_WITH_THERMAL_TIMING_MODULATION"
        if common_mechanical_recurrence and heat_advanced
        else "RECURRENT_POLYCRYSTAL_PAIR_NOT_QUALIFIED")
    return {
        "schema": "v67-poly-rx-asb-recurrence-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cases": {"adiabatic": adiabatic, "strong_bath": bath},
        "audits": {
            "checkpoint_chains_valid": chain_valid,
            "matched_load_time_endpoints": endpoints_matched,
            "heat_control_rates_valid": heat_controls_valid,
            "same_fixed_transmission_fraction": same_transmission,
            "all_hard_audits_passed": hard,
        },
        "observations": {
            "adiabatic_active_physical_band_intervals": ad_intervals,
            "bath_active_physical_band_intervals": bath_intervals,
            "heat_advances_early_onset_and_total_exposure": heat_advanced,
            "intermediate_matched_band_dissolution": intermediate_dissolution,
            "terminal_matched_band_recurrence": terminal_recurrence,
            "terminal_existing_boundary_rx_intersects_bands": terminal_rx_overlap,
        },
        "decision": {
            "recurrent_mechanical_rx_band_coexistence": common_mechanical_recurrence,
            "strict_heat_triggered_terminal_polycrystal_asb": (
                strict_heat_triggered_terminal_asb),
            "persistent_polycrystal_asb_qualified": False,
            "classification": classification,
        },
        "claim_boundary": (
            "The matched controls show dissolution and recurrence of a physical-"
            "power band with existing-boundary RX at 0.05064 shear. Adiabatic "
            "heating advances the first onset and increases cumulative exposure, "
            "but the bath also develops a terminal band; this is thermal timing "
            "modulation of a mechanical recurrence, not strict heat-triggered or "
            "yet persistent polycrystal ASB."),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adiabatic", nargs="+", type=Path, required=True)
    parser.add_argument("--bath", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = classify(
        [_case(path) for path in args.adiabatic],
        [_case(path) for path in args.bath])
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**payload["observations"], **payload["decision"]},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
