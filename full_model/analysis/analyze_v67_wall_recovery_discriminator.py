#!/usr/bin/env python3
"""Classify a matched baseline/redundant-wall-recovery production pair."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from full_model.analysis.analyze_v67_wall_orientation_response import _case
from full_model.analysis.run_v58_three_grain_production import _load_checkpoint


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _wall_inventory(directory):
    state, *_ = _load_checkpoint(Path(directory)/"checkpoint_latest.npz")
    owners = {}
    for grain_id, support, owner in zip(
            state.grain_ids, state.supports, state.owners):
        active = ((np.asarray(support) > .75)
                  &(np.asarray(owner.wall_order) >= .20))
        weight = np.asarray(support)[..., None]
        plus = np.asarray(owner.wall_plus_m2)
        minus = np.asarray(owner.wall_minus_m2)
        total = float(np.sum(
            weight*(plus+minus)*active[..., None], dtype=np.longdouble))
        pair = float(np.sum(
            weight*np.minimum(plus, minus)*active[..., None],
            dtype=np.longdouble))
        signed = float(np.sum(
            weight*np.abs(plus-minus)*active[..., None],
            dtype=np.longdouble))
        owners[str(grain_id)] = {
            "active_cells": int(np.count_nonzero(active)),
            "support_weighted_wall_line_cell_sum_m2": total,
            "support_weighted_signed_wall_cell_sum_m2": signed,
            "support_weighted_neutral_pair_cell_sum_m2": pair,
            "neutral_wall_line_fraction": 0.0 if total == 0.0 else 2.0*pair/total,
            "polarized_wall_line_fraction": 0.0 if total == 0.0 else signed/total,
        }
    return owners


def _result(directory):
    directory = Path(directory)
    payload = json.loads((directory/"result.json").read_text())
    return {
        "result_sha256": _sha(directory/"result.json"),
        "checkpoint_sha256": _sha(directory/"checkpoint_latest.npz"),
        "source_commit": payload["source_commit"],
        "source_tree_clean": payload["source_tree_clean"],
        "parent_checkpoint_sha256": payload["parent_checkpoint_sha256"],
        "configuration": payload["configuration"],
        "decision": payload["decision"],
        "records": payload["interval_records"],
        "orientation": _case(directory),
        "wall_inventory": _wall_inventory(directory),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("recovery", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    baseline = _result(args.baseline)
    recovery = _result(args.recovery)
    records = recovery["records"]
    accepted = [row for row in records
                if row["wall_recovery_classification"].startswith("ACCEPTED")]
    maximum_signed = max(
        (abs(row["wall_recovery_signed_residual_m2"]) for row in records),
        default=0.0)
    maximum_line = max(
        (abs(row["wall_recovery_line_residual_m2"]) for row in records),
        default=0.0)
    maximum_first_law = max(
        (abs(row["wall_recovery_first_law_relative_residual"])
         for row in records), default=0.0)
    payload = {
        "schema": "v67-wall-neutral-recovery-discriminator-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cases": {"baseline": baseline, "recovery": recovery},
        "decision": {
            "same_parent_checkpoint": bool(
                baseline["parent_checkpoint_sha256"]
                ==recovery["parent_checkpoint_sha256"]),
            "both_source_trees_clean": bool(
                baseline["source_tree_clean"] and recovery["source_tree_clean"]),
            "accepted_recovery_intervals": len(accepted),
            "annihilated_pair_cell_sum_m2": float(sum(
                row["wall_recovery_annihilated_pair_cell_sum_m2"]
                for row in accepted)),
            "maximum_signed_content_residual_m2": maximum_signed,
            "maximum_line_balance_residual_m2": maximum_line,
            "maximum_relative_first_law_residual": maximum_first_law,
            "hard_audits_passed": bool(
                maximum_first_law < .05
                and all(not row["wall_recovery_classification"].startswith(
                    "REJECTED") for row in records)),
            "baseline_strict_subgrain": baseline["orientation"][
                "any_strict_subgrain_qualified"],
            "recovery_strict_subgrain": recovery["orientation"][
                "any_strict_subgrain_qualified"],
            "claim_boundary": (
                "A positive result supports conserved redundant-wall recovery "
                "as a precursor-maturation mechanism; it is not material "
                "calibration or spontaneous grain birth."),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
