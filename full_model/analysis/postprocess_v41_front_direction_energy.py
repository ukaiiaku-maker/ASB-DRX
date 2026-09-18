#!/usr/bin/env python3
"""Turn the raw same-state V41 directional matrix into a decision record."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compact-raw-output", type=Path)
    args = parser.parse_args()
    raw = json.loads(args.input.read_text())
    if args.compact_raw_output is not None:
        compact_raw = dict(raw)
        compact_raw["states"] = [{
            **{key: value for key, value in state.items()
               if key != "stage_metadata"},
            "stage_metadata": {key: state.get("stage_metadata", {}).get(key)
                               for key in (
                "source_sha", "stage", "grid", "macro_dt_s",
                "interval_index", "completed_intervals", "physical_time_s")},
        } for state in raw["states"]]
        args.compact_raw_output.write_text(
            json.dumps(compact_raw, indent=2, sort_keys=True)+"\n")
    decisions = []
    for state in raw["states"]:
        trials = state["trials"]
        selected = [row for row in trials
                    if row["deterministic_rate_law"] == "complete_dissipation"]
        legacy = [row for row in trials
                  if row["deterministic_rate_law"] ==
                  "legacy_independent_metropolis"]
        directions = {}
        fractions = sorted({row["proposal_fraction"] for row in selected},
                           reverse=True)
        for fraction in fractions:
            pair = {row["proposal_direction"]: row for row in selected
                    if row["proposal_fraction"] == fraction}
            directions[str(fraction)] = {
                "forward_proposal_a_to_b_event_J": pair[1]["a_to_b_event_J"],
                "reverse_proposal_a_to_b_event_J": pair[-1]["a_to_b_event_J"],
                "forward_proposal_b_to_a_event_J": pair[1]["b_to_a_event_J"],
                "reverse_proposal_b_to_a_event_J": pair[-1]["b_to_a_event_J"],
                "forward_published": pair[1]["state_published"],
                "reverse_published": pair[-1]["state_published"],
            }
        smallest = min(fractions)
        small = [row for row in selected
                 if row["proposal_fraction"] == smallest]
        all_small_endpoints_uphill = all(
            row["a_to_b_event_J"] >= 0.0 and row["b_to_a_event_J"] >= 0.0
            for row in small)
        repaired_conflicts = [row for row in selected
                              if row["kinetic_rule_admitted_trial"]
                              and not row[
                                  "complete_physical_energy_admitted_trial"]]
        legacy_conflicts = [row for row in legacy
                            if row["kinetic_rule_admitted_trial"]
                            and not row[
                                "complete_physical_energy_admitted_trial"]]
        decisions.append({
            "name": state["name"],
            "checkpoint": state["checkpoint"],
            "checkpoint_sha256": state["checkpoint_sha256"],
            "extent_sequence": fractions,
            "matched_proposal_diagnostics": directions,
            "smallest_extent_both_complete_directions_uphill": (
                all_small_endpoints_uphill),
            "complete_dissipation_rate_energy_conflict_count": len(
                repaired_conflicts),
            "legacy_rate_energy_conflict_count": len(legacy_conflicts),
            "physical_stationarity_supported": bool(
                all_small_endpoints_uphill
                and all(row["net_velocity_a_to_b_m_s"] == 0.0
                        for row in small)),
        })
    stationary = next((row for row in decisions
                       if row["physical_stationarity_supported"]), None)
    result = {
        "schema": "asb-drx/v41/front-direction-energy-summary/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "raw_input": {"path": str(args.input.resolve()),
                      "sha256": hashlib.sha256(
                          args.input.read_bytes()).hexdigest()},
        "selected_production_law": "complete_dissipation",
        "retained_comparator": "legacy_independent_metropolis",
        "constitutive_statement": (
            "Each distinct outgoing channel receives a nonnegative EXP-floor "
            "coefficient multiplied by max(1-exp(DeltaF/kT),0); therefore only "
            "its own downhill complete transaction generates deterministic "
            "activity. The complete candidate energy guard remains active."),
        "state_decisions": decisions,
        "physical_pinning_supported": stationary is not None,
        "physical_pinning_checkpoint": (
            None if stationary is None else stationary["checkpoint"]),
        "classification": (
            "COMPLETE_DISSIPATION_REPAIR_PASSED;PHYSICAL_STATIONARITY_SUPPORTED"
            if stationary is not None else
            "COMPLETE_DISSIPATION_REPAIR_PASSED;STATIONARITY_UNRESOLVED"),
        "zero_applied_front_work": True,
        "mobility_multiplier_fitted": False,
        "complete_energy_guard_removed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
