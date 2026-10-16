#!/usr/bin/env python3
"""Classify fixed moving-boundary dislocation-transmission candidates."""

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
    return {
        "directory": str(directory.resolve()),
        "result_sha256": _sha(result_path),
        "checkpoint_sha256": _sha(checkpoint_path),
        "source_commit": result["source_commit"],
        "source_tree_clean": result["source_tree_clean"],
        "parent_checkpoint_sha256": result["parent_checkpoint_sha256"],
        "intervals": result["configuration"]["intervals"],
        "transmission_fraction": result["configuration"][
            "front_transmission_fraction"],
        "target_area_change_m2": result["owner_area_m2"][str(grain_id)]["change"],
        "before_bias": _directional_bias(
            result["before_directional_audit"], grain_id),
        "after_bias": _directional_bias(
            result["after_directional_audit"], grain_id),
        "after_components": _component_bias(result["after_directional_audit"]),
        "hard_audits_passed": bool(
            result["source_tree_clean"]
            and decision["first_law_hard_balance_passed"]
            and decision["mura_hard_balance_passed"]
            and decision["grain_ids_unchanged"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inherited_baseline", type=Path)
    parser.add_argument("complete_absorption", type=Path)
    parser.add_argument("partial_one", type=Path)
    parser.add_argument("partial_horizon", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--grain-id", type=int, default=20)
    args = parser.parse_args()
    baseline_result_path = args.inherited_baseline/"result.json"
    baseline_checkpoint_path = args.inherited_baseline/"checkpoint_latest.npz"
    baseline_result = json.loads(baseline_result_path.read_text())
    baseline = _directional_bias(
        baseline_result["after_directional_audit"], args.grain_id)
    baseline_checkpoint_sha256 = _sha(baseline_checkpoint_path)
    cases = {
        "transmission_0": _case(args.complete_absorption, args.grain_id),
        "transmission_0p25_one": _case(args.partial_one, args.grain_id),
        "transmission_0p25_horizon": _case(args.partial_horizon, args.grain_id),
    }
    parents = {case["parent_checkpoint_sha256"] for case in cases.values()}
    common_restart = (
        len(parents) == 1 and parents == {baseline_checkpoint_sha256})
    baseline_bias = baseline["net_growth_minus_shrink_Pa"]
    partial = cases["transmission_0p25_horizon"]
    endpoint = partial["after_bias"]["net_growth_minus_shrink_Pa"]
    payload = {
        "schema": "v67-front-transmission-discriminator-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_inherited_transmission_fraction": 0.55,
        "baseline": {
            "directory": str(args.inherited_baseline.resolve()),
            "result_sha256": _sha(baseline_result_path),
            "checkpoint_sha256": baseline_checkpoint_sha256,
            "source_commit": baseline_result["source_commit"],
            "source_tree_clean": baseline_result["source_tree_clean"],
            "directional_bias": baseline,
        },
        "cases": cases,
        "decision": {
            "same_parent_checkpoint": common_restart,
            "all_hard_audits_passed": all(
                case["hard_audits_passed"] for case in cases.values()),
            "complete_absorption_endpoint_bias_Pa": cases[
                "transmission_0"]["after_bias"]["net_growth_minus_shrink_Pa"],
            "partial_absorption_endpoint_bias_Pa": endpoint,
            "partial_absorption_target_area_change_m2": partial[
                "target_area_change_m2"],
            "partial_absorption_sustains_positive_RX_bias": bool(
                baseline_bias <= 0.0 < endpoint
                and partial["target_area_change_m2"] > 0.0),
            "classification": (
                "FIXED_PARTIAL_ABSORPTION_SUSTAINS_LATE_RX"
                if baseline_bias <= 0.0 < endpoint
                and partial["target_area_change_m2"] > 0.0 else
                "PARTIAL_ABSORPTION_DOES_NOT_SUSTAIN_LATE_RX"),
        },
        "claim_boundary": (
            "Existing-boundary RX discriminator at one lower-rate late state. "
            "The 25% transmission fraction is fixed across the horizon and "
            "does not depend on rate, time, grain label, or outcome. This does "
            "not yet establish spontaneous grain birth or material calibration."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
