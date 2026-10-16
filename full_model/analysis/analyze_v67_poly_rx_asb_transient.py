#!/usr/bin/env python3
"""Classify the matched early polycrystal RX/thermal-band trajectory."""

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


def _case(path):
    root = Path(path)
    result = json.loads((root/"result.json").read_text())
    state, _, step, load, _, configuration, _ = _load_checkpoint(
        root/"checkpoint_latest.npz")
    temperature = np.sum(
        np.asarray(state.supports)*np.stack([
            owner.temperature_K for owner in state.owners]), axis=0)
    return {
        "directory": str(root.resolve()),
        "checkpoint_sha256": _sha(root/"checkpoint_latest.npz"),
        "parent_checkpoint_sha256": result["parent_checkpoint_sha256"],
        "source_commit": result["source_commit"],
        "source_tree_clean": result["source_tree_clean"],
        "step": step, "load": load,
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


def classify(ad_onset, bath_onset, ad_later, bath_later):
    cases = (ad_onset, bath_onset, ad_later, bath_later)
    chain_valid = bool(
        ad_later["parent_checkpoint_sha256"] == ad_onset["checkpoint_sha256"]
        and bath_later["parent_checkpoint_sha256"]
        == bath_onset["checkpoint_sha256"])
    endpoints_matched = bool(
        ad_onset["step"] == bath_onset["step"]
        and ad_onset["load"] == bath_onset["load"]
        and ad_later["step"] == bath_later["step"]
        and ad_later["load"] == bath_later["load"])
    heat_controls_valid = bool(
        ad_onset["thermal_control"] == ad_later["thermal_control"] == "adiabatic"
        and ad_onset["thermal_bath_rate_s"] == 0.0
        and ad_later["thermal_bath_rate_s"] == 0.0
        and bath_onset["thermal_control"] == bath_later["thermal_control"]
        == "strong_bath"
        and bath_onset["thermal_bath_rate_s"] > 0.0
        and bath_later["thermal_bath_rate_s"] > 0.0)
    same_transmission = len({case["front_transmission_fraction"]
                             for case in cases}) == 1
    hard = all(
        case["source_tree_clean"]
        and case["decision"]["all_hard_absorption_balances_passed"]
        and case["decision"]["mura_hard_balance_passed"]
        and case["decision"]["first_law_hard_balance_passed"]
        and case["decision"]["grain_ids_unchanged"] for case in cases)
    ad_intervals = (ad_onset["active_physical_band_intervals"]
                    +ad_later["active_physical_band_intervals"])
    bath_intervals = (bath_onset["active_physical_band_intervals"]
                      +bath_later["active_physical_band_intervals"])
    heat_advanced = bool(
        ad_onset["decision"]["final_connected_band"]
        and not bath_onset["decision"]["final_connected_band"]
        and ad_intervals > bath_intervals)
    rx_overlap = bool(
        ad_onset["decision"]["growing_owner_intersects_active_band"]
        and ad_later["decision"]["growing_owner_intersects_active_band"]
        and bath_later["decision"]["growing_owner_intersects_active_band"])
    terminal_dissolution = bool(
        not ad_later["decision"]["final_connected_band"]
        and not bath_later["decision"]["final_connected_band"])
    valid = all((chain_valid, endpoints_matched, heat_controls_valid,
                 same_transmission, hard, heat_advanced, rx_overlap,
                 terminal_dissolution))
    return {
        "schema": "v67-poly-rx-asb-transient-pair-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cases": {
            "adiabatic_onset": ad_onset, "bath_onset": bath_onset,
            "adiabatic_later": ad_later, "bath_later": bath_later},
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
            "heat_advances_band_onset_and_exposure": heat_advanced,
            "existing_boundary_rx_intersects_active_bands": rx_overlap,
            "both_bands_dissolve_by_later_endpoint": terminal_dissolution,
        },
        "decision": {
            "transient_heat_advanced_rx_band_coexistence": valid,
            "persistent_polycrystal_asb_qualified": False,
            "classification": (
                "TRANSIENT_HEAT_ADVANCED_RX_BAND_COEXISTENCE"
                if valid else "TRANSIENT_PAIR_NOT_QUALIFIED"),
        },
        "claim_boundary": (
            "Existing-boundary RX intersects transient physical-power bands. "
            "Heat advances onset and increases band exposure, but both controls "
            "are unbanded at 0.03464 shear; persistent polycrystal ASB is not "
            "claimed from this early horizon."),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ad_onset", type=Path)
    parser.add_argument("bath_onset", type=Path)
    parser.add_argument("ad_later", type=Path)
    parser.add_argument("bath_later", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    payload = classify(*[_case(getattr(args, name)) for name in (
        "ad_onset", "bath_onset", "ad_later", "bath_later")])
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps({**payload["observations"], **payload["decision"]},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
