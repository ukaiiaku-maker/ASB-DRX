#!/usr/bin/env python3
"""Classify the matched thermal response of the V67 arrival-limited candidate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def _load(directory: Path) -> dict:
    return json.loads((directory / "result.json").read_text())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hard(result: dict) -> bool:
    decision = result["decision"]
    return bool(
        decision["all_hard_absorption_balances_passed"]
        and decision["mura_hard_balance_passed"]
        and decision["first_law_hard_balance_passed"]
        and decision["grain_ids_unchanged"])


def _segment(directory: Path) -> dict:
    result = _load(directory)
    history = result["response_history"]
    bands = [row for row in history if row["connected_physical_band"]]
    endpoint = history[-1]
    checkpoint = directory / "checkpoint_latest.npz"
    return {
        "directory": str(directory.resolve()),
        "checkpoint_sha256": _sha(checkpoint),
        "source_commit": result["source_commit"],
        "hard_valid": _hard(result),
        "active_band_samples": len(bands),
        "first_band_load": bands[0]["load"] if bands else None,
        "last_band_load": bands[-1]["load"] if bands else None,
        "terminal_band_persistence_intervals": result["response_summary"][
            "terminal_band_persistence_intervals"],
        "endpoint_post_peak_softening_fraction": result["response_summary"][
            "endpoint_post_peak_softening_fraction"],
        "endpoint": endpoint,
        "grain20_relative_area_change": result["owner_results"]["20"][
            "relative_area_change"],
        "grain20_growth_inside_band_fraction": result["owner_results"]["20"][
            "positive_growth_inside_band_fraction"],
    }


def classify(adiabatic: list[dict], bath: list[dict]) -> dict:
    hard = all(row["hard_valid"] for row in adiabatic + bath)
    latest_ad = adiabatic[-1]
    latest_bath = bath[-1]
    heat_bias = (
        latest_ad["active_band_samples"] > latest_bath["active_band_samples"]
        and latest_ad["endpoint"]["temperature_range_K"]
        > latest_bath["endpoint"]["temperature_range_K"] + 1.0)
    asb = (
        latest_ad["terminal_band_persistence_intervals"] >= 20
        and latest_ad["endpoint_post_peak_softening_fraction"] >= .05
        and latest_bath["terminal_band_persistence_intervals"] < 20)
    rx = all(row["grain20_relative_area_change"] > 0.0
             for row in adiabatic + bath)
    if not hard:
        label = "HARD_INVALID_MATCHED_ARRIVAL_EXCHANGE_PAIR"
    elif heat_bias and rx and not asb:
        label = "HEAT_BIASED_BAND_RECURRENCE_WITH_RX_BUT_NO_ASB_SOFTENING"
    elif asb and rx:
        label = "THERMALLY_CAUSAL_ASB_WITH_EXISTING_BOUNDARY_RX"
    else:
        label = "NO_THERMALLY_CAUSAL_ASB_SEPARATION"
    return {
        "classification": label,
        "all_hard_audits_passed": hard,
        "heat_biases_late_band_recurrence": heat_bias,
        "existing_boundary_rx_preserved": rx,
        "asb_qualified": asb,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--adiabatic", nargs=4, type=Path, required=True)
    parser.add_argument("--bath", nargs=4, type=Path, required=True)
    parser.add_argument("--attribution", type=Path, required=True)
    args = parser.parse_args()
    adiabatic = [_segment(path) for path in args.adiabatic]
    bath = [_segment(path) for path in args.bath]
    attribution = json.loads(args.attribution.read_text())
    payload = {
        "schema": "v67-arrival-exchange-matched-thermal-pair-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "adiabatic_segments": adiabatic,
        "strong_bath_segments": bath,
        "exp_floor_attribution": {
            "path": str(args.attribution.resolve()),
            "sha256": _sha(args.attribution),
            "native_rate_map_checks": attribution[
                "native_independent_rate_map_checks"],
            "projected_speed_difference": attribution[
                "recomputed_after_projection_speed_difference"],
            "temperature_fraction_of_speed_discrepancy_removed": attribution[
                "one_input_at_a_time_substitutions"]["temperature"][
                    "fraction_of_baseline_error_removed"],
            "raw_stress_fraction_of_speed_discrepancy_removed": attribution[
                "one_input_at_a_time_substitutions"]["raw"][
                    "fraction_of_baseline_error_removed"],
        },
        "criteria": {
            "minimum_terminal_band_intervals": 20,
            "minimum_post_peak_softening_fraction": .05,
            "matched_bath_must_not_qualify": True,
        },
        "claim_boundary": (
            "The common carrier law is qualified against collapse and preserves "
            "existing-boundary RX. Intermittent adiabatic-only late recurrence "
            "without persistence or softening is not classified as ASB."),
    }
    payload["decision"] = classify(adiabatic, bath)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
