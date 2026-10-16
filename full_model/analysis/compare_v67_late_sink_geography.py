#!/usr/bin/env python3
"""Compare GB/TJ and matched-random bulk recovery from one late RX state."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from full_model.analysis.analyze_v67_existing_boundary_growth import (
    _directional_bias,
)
from full_model.analysis.compare_v67_late_source_geography import _component_bias


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _case(directory, grain_id):
    result_path = directory/"result.json"
    checkpoint_path = directory/"checkpoint_latest.npz"
    result = json.loads(result_path.read_text())
    decision = result["decision"]
    records = result["interval_records"]
    bias = _directional_bias(result["after_directional_audit"], grain_id)
    return {
        "directory": str(directory.resolve()),
        "result_sha256": _sha(result_path),
        "checkpoint_sha256": _sha(checkpoint_path),
        "source_commit": result["source_commit"],
        "source_tree_clean": result["source_tree_clean"],
        "parent_checkpoint_sha256": result["parent_checkpoint_sha256"],
        "configuration": result["configuration"],
        "target_area_change_m2": result["owner_area_m2"][str(grain_id)]["change"],
        "accepted_sink_intervals": decision["accepted_sink_intervals"],
        "annihilated_pair_cell_sum_m2": sum(
            row["sink_annihilated_pair_cell_sum_m2"] for row in records),
        "directional_bias": bias,
        "directional_component_growth_minus_shrink_Pa": _component_bias(
            result["after_directional_audit"]),
        "hard_audits_passed": bool(
            result["source_tree_clean"]
            and decision["first_law_hard_balance_passed"]
            and decision["mura_hard_balance_passed"]
            and decision["grain_ids_unchanged"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gb_tj", type=Path)
    parser.add_argument("matched_random", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--grain-id", type=int, default=20)
    args = parser.parse_args()
    boundary = _case(args.gb_tj, args.grain_id)
    random = _case(args.matched_random, args.grain_id)
    left = dict(boundary["configuration"])
    right = dict(random["configuration"])
    ignored = {"parent_checkpoint", "sink_mode"}
    config_match = all(
        left.get(key) == right.get(key) for key in set(left)|set(right)
        if key not in ignored)
    b_bias = boundary["directional_bias"]["net_growth_minus_shrink_Pa"]
    r_bias = random["directional_bias"]["net_growth_minus_shrink_Pa"]
    payload = {
        "schema": "v67-late-sink-geography-discriminator-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cases": {"gb_tj_only": boundary, "matched_random": random},
        "decision": {
            "same_parent_checkpoint": boundary["parent_checkpoint_sha256"]
            == random["parent_checkpoint_sha256"],
            "all_non_sink_configuration_matched": config_match,
            "all_hard_audits_passed": bool(
                boundary["hard_audits_passed"] and random["hard_audits_passed"]),
            "gb_tj_endpoint_bias_Pa": b_bias,
            "matched_random_endpoint_bias_Pa": r_bias,
            "matched_random_minus_gb_tj_bias_Pa": r_bias-b_bias,
            "gb_tj_sustains_RX": b_bias > 0.0,
            "matched_random_sustains_RX": r_bias > 0.0,
            "classification": (
                "BULK_RECOVERY_GEOGRAPHY_SUSTAINS_RX"
                if max(b_bias, r_bias) > 0.0 else
                "BULK_RECOVERY_ERASES_RX_DRIVING_CONTRAST"),
        },
        "claim_boundary": (
            "Same EXP-floor recovery parameters and common parent; geography "
            "is the only mechanism change. No barrier fitting is performed."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
