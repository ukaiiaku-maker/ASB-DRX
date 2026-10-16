#!/usr/bin/env python3
"""Classify common-parent, matched-strain fixed-transmission rate forks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from full_model.analysis.analyze_v67_existing_boundary_growth import (
    _directional_bias,
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _case(directory, grain_id):
    result_path = directory/"result.json"
    checkpoint_path = directory/"checkpoint_latest.npz"
    result = json.loads(result_path.read_text())
    config = result["configuration"]
    decision = result["decision"]
    area = result["owner_area_m2"][str(grain_id)]
    increment_percent = 100.0*float(area["change"])/float(area["before"])
    hard = bool(
        result["source_tree_clean"]
        and decision["first_law_hard_balance_passed"]
        and decision["mura_hard_balance_passed"]
        and decision["grain_ids_unchanged"])
    return {
        "directory": str(directory.resolve()),
        "result_sha256": _sha(result_path),
        "checkpoint_sha256": _sha(checkpoint_path),
        "source_commit": result["source_commit"],
        "source_tree_clean": result["source_tree_clean"],
        "parent_checkpoint_sha256": result["parent_checkpoint_sha256"],
        "hard_audits_passed": hard,
        "shear_rate_s": float(config["shear_rate_s"]),
        "dt_s": float(config["dt_s"]),
        "intervals": int(config["intervals"]),
        "added_strain": (
            float(config["shear_rate_s"])*float(config["dt_s"])
            *int(config["intervals"])),
        "transmission_fraction": float(
            config["front_transmission_fraction"]),
        "target_area_before_m2": float(area["before"]),
        "target_area_change_m2": float(area["change"]),
        "target_growth_increment_percent": increment_percent,
        "endpoint_directional_bias": _directional_bias(
            result["after_directional_audit"], grain_id),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("high_rate", type=Path)
    parser.add_argument("low_rate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--grain-id", type=int, default=20)
    args = parser.parse_args()
    high = _case(args.high_rate, args.grain_id)
    low = _case(args.low_rate, args.grain_id)
    same_parent = (
        high["parent_checkpoint_sha256"]
        == low["parent_checkpoint_sha256"])
    same_transmission = (
        high["transmission_fraction"] == low["transmission_fraction"])
    matched_strain = math.isclose(
        high["added_strain"], low["added_strain"], rel_tol=0.0,
        abs_tol=32.0*math.ulp(max(high["added_strain"], low["added_strain"])))
    low_advantage = (
        low["target_growth_increment_percent"]
        > high["target_growth_increment_percent"])
    both_positive = (
        low["target_growth_increment_percent"] > 0.0
        and high["target_growth_increment_percent"] > 0.0)
    hard = high["hard_audits_passed"] and low["hard_audits_passed"]
    passed = bool(
        same_parent and same_transmission and matched_strain and hard
        and both_positive and low_advantage)
    payload = {
        "schema": "v67-fixed-transmission-matched-strain-rate-pair-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "grain_id": args.grain_id,
        "cases": {"high_rate": high, "low_rate": low},
        "decision": {
            "same_parent_checkpoint": same_parent,
            "same_fixed_transmission_fraction": same_transmission,
            "matched_added_strain": matched_strain,
            "all_hard_audits_passed": hard,
            "both_target_growth_increments_positive": both_positive,
            "lower_rate_growth_advantage": low_advantage,
            "growth_increment_ratio_low_over_high": (
                low["target_growth_increment_percent"]
                /high["target_growth_increment_percent"]
                if high["target_growth_increment_percent"] != 0.0 else None),
            "classification": (
                "QUALIFIED_FIXED_LAW_LOWER_RATE_RX_ADVANTAGE"
                if passed else "MATCHED_RATE_RESPONSE_NOT_QUALIFIED"),
        },
        "claim_boundary": (
            "Common-parent, matched-added-strain existing-boundary response "
            "under one fixed front-transmission law. This does not establish "
            "new-grain birth, ASB coexistence, or material calibration."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
