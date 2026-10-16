#!/usr/bin/env python3
"""Compare late GB/TJ and matched-random source geography from one state."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from full_model.analysis.analyze_v67_existing_boundary_growth import (
    _directional_bias,
)
from full_model.analysis.run_v58_three_grain_production import _load_checkpoint


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _component_bias(audit):
    rows = [row for row in audit["directional_prices"]
            if row["target_owner_grows"] or row["target_owner_shrinks"]]
    values = {}
    for row in rows:
        sign = 1.0 if row["target_owner_grows"] else -1.0
        volume = float(row["swept_volume_m3"])
        for name, delta in row[
                "structural_energy_component_change_J"].items():
            values[name] = values.get(name, 0.0)+sign*(-float(delta)/volume)
    return values


def _case(directory, grain_id, spacing_m, thickness_m):
    result_path = directory/"result.json"
    checkpoint_path = directory/"checkpoint_latest.npz"
    result = json.loads(result_path.read_text())
    state, runtime, step, gamma, initial, configuration, provenance = (
        _load_checkpoint(checkpoint_path))
    index = state.grain_ids.index(grain_id)
    area = float(np.sum(state.supports[index], dtype=np.longdouble))*spacing_m**2
    initial_area = float(initial[index])/thickness_m
    line = 0.0
    for support, owner in zip(state.supports, state.owners):
        total = sum(np.asarray(getattr(owner, name)) for name in (
            "mobile_plus_m2", "mobile_minus_m2", "forest_plus_m2",
            "forest_minus_m2", "wall_plus_m2", "wall_minus_m2"))
        line += float(np.sum(
            np.asarray(support)[..., None]*total, dtype=np.longdouble))
    decision = result["decision"]
    return {
        "directory": str(directory.resolve()),
        "result_sha256": _sha(result_path),
        "checkpoint_sha256": _sha(checkpoint_path),
        "source_commit": result["source_commit"],
        "source_tree_clean": result["source_tree_clean"],
        "parent_checkpoint_sha256": result["parent_checkpoint_sha256"],
        "configuration": result["configuration"],
        "growth_percent": 100.0*(area/initial_area-1.0),
        "target_area_change_m2": result["owner_area_m2"][str(grain_id)][
            "change"],
        "support_weighted_total_line_cell_sum_m2": line,
        "accepted_source_intervals": decision["accepted_source_intervals"],
        "source_event_count": float(sum(
            row["source_event_count"] for row in result["interval_records"])),
        "directional_bias": _directional_bias(
            result["after_directional_audit"], grain_id),
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
    parser.add_argument("baseline", type=Path)
    parser.add_argument("matched_random", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--grain-id", type=int, default=20)
    parser.add_argument("--spacing-m", type=float, required=True)
    parser.add_argument("--represented-thickness-m", type=float, required=True)
    args = parser.parse_args()
    baseline = _case(
        args.baseline, args.grain_id, args.spacing_m,
        args.represented_thickness_m)
    random = _case(
        args.matched_random, args.grain_id, args.spacing_m,
        args.represented_thickness_m)
    same_parent = (baseline["parent_checkpoint_sha256"]
                   ==random["parent_checkpoint_sha256"])
    config_left = dict(baseline["configuration"])
    config_right = dict(random["configuration"])
    ignored = {"parent_checkpoint", "source_mode"}
    matched = all(
        config_left.get(key) == config_right.get(key)
        for key in set(config_left)|set(config_right) if key not in ignored)
    payload = {
        "schema": "v67-late-source-geography-discriminator-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cases": {"gb_tj_only": baseline, "matched_random": random},
        "decision": {
            "same_parent_checkpoint": same_parent,
            "all_non_source_configuration_matched": matched,
            "all_hard_audits_passed": bool(
                baseline["hard_audits_passed"]
                and random["hard_audits_passed"]),
            "growth_difference_percentage_point": (
                random["growth_percent"]-baseline["growth_percent"]),
            "endpoint_bias_difference_Pa": (
                random["directional_bias"]["net_growth_minus_shrink_Pa"]
                -baseline["directional_bias"]["net_growth_minus_shrink_Pa"]),
            "random_source_reactivates_line_supply": bool(
                random["accepted_source_intervals"]
                >baseline["accepted_source_intervals"]),
            "random_source_restores_positive_growth_bias": bool(
                random["directional_bias"]["net_growth_minus_shrink_Pa"] > 0.0
                and baseline["directional_bias"][
                    "net_growth_minus_shrink_Pa"] <= 0.0),
        },
        "claim_boundary": (
            "Same-law source-geography discriminator; it does not authorize "
            "a source-rate refit or condition-specific mechanism switch."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
