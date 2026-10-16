#!/usr/bin/env python3
"""Classify the finite-amplitude basin of the V67 ASB response."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def _load(path):
    root = Path(path)
    payload = json.loads((root/"result.json").read_text())
    return {
        "case": root.name,
        "seed_peak_K": payload["configuration"]["temperature_seed_peak_K"],
        "initial_temperature_contrast_K": payload[
            "initial_temperature"]["contrast_K"],
        "initial_power_maximum_to_mean": payload[
            "initial_power_localization"]["maximum_to_mean"],
        "final_power_maximum_to_mean": payload[
            "final_power_localization"]["maximum_to_mean"],
        "band_intervals": payload["result"]["band_intervals"],
        "terminal_band_persistence_intervals": payload["result"][
            "terminal_band_persistence_intervals"],
        "final_temperature_contrast_K": payload["result"][
            "final_temperature_contrast_K"],
        "post_peak_stress_drop_fraction": payload["result"][
            "post_peak_stress_drop_fraction"],
        "source_commit": payload["source_commit"],
        "source_tree_clean": payload["source_tree_clean"],
        "configuration": payload["configuration"],
    }


def classify(cases):
    rows = sorted(cases, key=lambda row: row["seed_peak_K"])
    reference = dict(rows[0]["configuration"])
    reference.pop("temperature_seed_peak_K")
    same_law = True
    for row in rows[1:]:
        candidate = dict(row["configuration"])
        candidate.pop("temperature_seed_peak_K")
        same_law &= candidate == reference
    onset_negative = [row for row in rows if row["band_intervals"] == 0]
    onset_positive = [row for row in rows if row["band_intervals"] > 0]
    persistent_negative = [row for row in rows
                           if row["terminal_band_persistence_intervals"] < 20]
    persistent_positive = [row for row in rows
                           if row["terminal_band_persistence_intervals"] >= 20]
    initial_subcritical = all(
        row["initial_power_maximum_to_mean"] < 2.0 for row in rows)
    ordered_response = all(
        left["final_power_maximum_to_mean"]
        < right["final_power_maximum_to_mean"]
        for left, right in zip(rows, rows[1:]))
    complete = all((onset_negative, onset_positive,
                    persistent_negative, persistent_positive))
    return {
        "schema": "v67-asb-seed-amplitude-bracket-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cases": rows,
        "audits": {
            "same_configuration_except_seed_amplitude": same_law,
            "all_source_trees_clean": all(row["source_tree_clean"] for row in rows),
            "all_initial_power_fields_subcritical": initial_subcritical,
            "terminal_localization_strictly_ordered_with_seed": ordered_response,
        },
        "brackets": {
            "any_band_onset_seed_peak_K": {
                "last_negative": max(row["seed_peak_K"] for row in onset_negative),
                "first_positive": min(row["seed_peak_K"] for row in onset_positive),
            } if complete else None,
            "twenty_interval_persistence_seed_peak_K": {
                "last_negative": max(
                    row["seed_peak_K"] for row in persistent_negative),
                "first_positive": min(
                    row["seed_peak_K"] for row in persistent_positive),
            } if complete else None,
        },
        "decision": {
            "finite_amplitude_basin_bracketed": bool(
                complete and same_law and initial_subcritical and ordered_response),
            "classification": (
                "BRACKETED_PREPARED_ASB_FINITE_AMPLITUDE_BASIN"
                if complete and same_law and initial_subcritical and ordered_response
                else "NOT_BRACKETED"),
        },
        "claim_boundary": (
            "The bracket is specific to n32, 0.124 terminal shear, the fixed "
            "physical seed width, and the declared twice-mean connected-band "
            "criterion. It is not a material imperfection calibration."),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("cases", nargs="+", type=Path)
    args = parser.parse_args()
    payload = classify([_load(path) for path in args.cases])
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps({
        **payload["decision"], **payload["brackets"]},
        indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
