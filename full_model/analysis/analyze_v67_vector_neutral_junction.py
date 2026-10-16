#!/usr/bin/env python3
"""Classify a late vector-neutral GB junction reaction at fixed mechanics."""

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
from full_model.analysis.compare_v67_late_source_geography import (
    _component_bias,
)
from full_model.analysis.run_v58_three_grain_production import _load_checkpoint


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _line_inventory(state):
    return float(sum(np.sum(
        component.boundary_plus_m2+component.boundary_minus_m2,
        dtype=np.longdouble) for component in state.interfaces))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--grain-id", type=int, default=20)
    parser.add_argument("--spacing-m", type=float, required=True)
    parser.add_argument("--represented-thickness-m", type=float, required=True)
    args = parser.parse_args()
    result_path = args.case/"result.json"
    checkpoint_path = args.case/"checkpoint_latest.npz"
    result = json.loads(result_path.read_text())
    state, _, _, _, initial, _, _ = _load_checkpoint(checkpoint_path)
    parent_path = Path(result["configuration"]["parent_checkpoint"])
    parent, *_ = _load_checkpoint(parent_path)
    index = state.grain_ids.index(args.grain_id)
    area = float(np.sum(state.supports[index], dtype=np.longdouble)) * (
        args.spacing_m**2)
    initial_area = float(initial[index])/args.represented_thickness_m
    before_bias = _directional_bias(
        result["before_directional_audit"], args.grain_id)
    after_bias = _directional_bias(
        result["after_directional_audit"], args.grain_id)
    before_components = _component_bias(result["before_directional_audit"])
    after_components = _component_bias(result["after_directional_audit"])
    decision = result["decision"]
    accepted = int(decision["accepted_interface_sink_intervals"])
    hard = bool(
        result["source_tree_clean"]
        and decision["first_law_hard_balance_passed"]
        and decision["mura_hard_balance_passed"]
        and decision["grain_ids_unchanged"])
    before_net = before_bias["net_growth_minus_shrink_Pa"]
    after_net = after_bias["net_growth_minus_shrink_Pa"]
    payload = {
        "schema": "v67-vector-neutral-junction-discriminator-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": result["source_commit"],
        "source_tree_clean": result["source_tree_clean"],
        "result_sha256": _sha(result_path),
        "checkpoint_sha256": _sha(checkpoint_path),
        "parent_checkpoint_sha256": _sha(parent_path),
        "hard_audits_passed": hard,
        "accepted_interface_sink_intervals": accepted,
        "removed_boundary_line_cell_sum_m2": (
            _line_inventory(parent)-_line_inventory(state)),
        "target_growth_percent": 100.0*(area/initial_area-1.0),
        "before_directional_growth_minus_shrink_Pa": before_net,
        "after_directional_growth_minus_shrink_Pa": after_net,
        "directional_bias_change_Pa": after_net-before_net,
        "before_component_growth_minus_shrink_Pa": before_components,
        "after_component_growth_minus_shrink_Pa": after_components,
        "classification": (
            "UNQUALIFIED_HARD_AUDIT" if not hard else
            ("INACTIVE_VECTOR_NEUTRAL_REACTION" if accepted == 0 else
             ("VECTOR_NEUTRAL_REACTION_RESTORES_RX_BIAS"
              if before_net <= 0.0 < after_net else
              "VECTOR_NEUTRAL_REACTION_INSUFFICIENT_FOR_RX_BIAS"))),
        "claim_boundary": (
            "The reaction preserves the explicit Burgers tensor and removes "
            "only scalar line above its exact carrier-capacity lower bound. "
            "It does not test tangential Burgers transfer or disconnection "
            "coupling."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps({key: payload[key] for key in (
        "hard_audits_passed", "accepted_interface_sink_intervals",
        "removed_boundary_line_cell_sum_m2",
        "before_directional_growth_minus_shrink_Pa",
        "after_directional_growth_minus_shrink_Pa", "classification")},
        indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
