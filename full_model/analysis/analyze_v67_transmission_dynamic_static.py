#!/usr/bin/env python3
"""Classify a fixed-transmission dynamic-RX/retained-static-RX pair."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from full_model.analysis.analyze_v67_existing_boundary_growth import (
    _directional_bias,
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _hard(result):
    decision = result["decision"]
    return bool(
        result["source_tree_clean"]
        and decision["first_law_hard_balance_passed"]
        and decision["mura_hard_balance_passed"]
        and decision["grain_ids_unchanged"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dynamic_summary", type=Path)
    parser.add_argument("dynamic_endpoint", type=Path)
    parser.add_argument("retained_static", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--grain-id", type=int, default=20)
    args = parser.parse_args()

    summary = json.loads(args.dynamic_summary.read_text())
    dynamic_result_path = args.dynamic_endpoint/"result.json"
    dynamic_checkpoint_path = args.dynamic_endpoint/"checkpoint_latest.npz"
    static_result_path = args.retained_static/"result.json"
    static_checkpoint_path = args.retained_static/"checkpoint_latest.npz"
    dynamic = json.loads(dynamic_result_path.read_text())
    static = json.loads(static_result_path.read_text())
    dynamic_checkpoint_sha256 = _sha(dynamic_checkpoint_path)
    same_parent = (
        static["parent_checkpoint_sha256"] == dynamic_checkpoint_sha256)
    dynamic_fraction = float(
        dynamic["configuration"]["front_transmission_fraction"])
    static_fraction = float(
        static["configuration"]["front_transmission_fraction"])
    static_rate = float(static["configuration"]["shear_rate_s"])
    area_change = float(
        static["owner_area_m2"][str(args.grain_id)]["change"])
    static_bias = _directional_bias(
        static["after_directional_audit"], args.grain_id)
    static_bias_Pa = static_bias["net_growth_minus_shrink_Pa"]
    dynamic_pass = bool(
        summary["decision"]["classification"]
        == "QUALIFIED_ONE_PERCENT_EXISTING_BOUNDARY_RX"
        and summary["decision"]["one_percent_reached"]
        and summary["decision"]["growth_monotone"]
        and summary["decision"]["all_hard_audits_passed"])
    pair_pass = bool(
        dynamic_pass and _hard(dynamic) and _hard(static) and same_parent
        and dynamic_fraction == static_fraction
        and static_rate == 0.0 and area_change > 0.0 and static_bias_Pa > 0.0)
    payload = {
        "schema": "v67-fixed-transmission-dynamic-static-pair-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "grain_id": args.grain_id,
        "dynamic": {
            "summary": str(args.dynamic_summary.resolve()),
            "summary_sha256": _sha(args.dynamic_summary),
            "directory": str(args.dynamic_endpoint.resolve()),
            "result_sha256": _sha(dynamic_result_path),
            "checkpoint_sha256": dynamic_checkpoint_sha256,
            "source_commit": dynamic["source_commit"],
            "hard_audits_passed": _hard(dynamic),
            "transmission_fraction": dynamic_fraction,
            "growth_percent": summary["decision"]["latest_growth_percent"],
            "endpoint_bias_Pa": summary["decision"][
                "latest_net_directional_growth_bias_Pa"],
        },
        "retained_static": {
            "directory": str(args.retained_static.resolve()),
            "result_sha256": _sha(static_result_path),
            "checkpoint_sha256": _sha(static_checkpoint_path),
            "source_commit": static["source_commit"],
            "hard_audits_passed": _hard(static),
            "parent_checkpoint_sha256": static[
                "parent_checkpoint_sha256"],
            "shear_rate_s": static_rate,
            "transmission_fraction": static_fraction,
            "target_area_change_m2": area_change,
            "endpoint_bias": static_bias,
        },
        "decision": {
            "dynamic_one_percent_RX_passed": dynamic_pass,
            "exact_no_reset_parent_continuity": same_parent,
            "same_fixed_transmission_fraction": (
                dynamic_fraction == static_fraction),
            "retained_static_growth_positive": area_change > 0.0,
            "retained_static_endpoint_bias_positive": static_bias_Pa > 0.0,
            "all_hard_audits_passed": _hard(dynamic) and _hard(static),
            "classification": (
                "QUALIFIED_DYNAMIC_AND_RETAINED_STATIC_EXISTING_BOUNDARY_RX"
                if pair_pass else
                "DYNAMIC_STATIC_RX_PAIR_NOT_QUALIFIED"),
        },
        "claim_boundary": (
            "This pair qualifies existing-boundary growth after a late-state "
            "fixed-transmission fork. It does not establish spontaneous grain "
            "birth, a from-origin single-parameter trajectory, or material "
            "calibration; those require separate tests."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
