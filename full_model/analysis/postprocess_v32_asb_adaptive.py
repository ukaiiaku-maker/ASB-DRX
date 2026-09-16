#!/usr/bin/env python3
"""Checkpoint-safe ranking of the paired V32 Tier-1 ASB condition screen."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from full_model.analysis.postprocess_v32_asb_anchor import (
    pair_history,
    run_terminal_status,
    summarize_pair,
)
from full_model.hpc3.run_v32_asb_adaptive_case import COORDINATES


def rank_key(record: dict) -> tuple:
    """Pre-registered ordering: conjunction, localization, then thermal response."""
    summary = record["summary"]
    return (
        -int(summary["raw_conjunction"]["qualifying_snapshot_count"] > 0),
        summary["latest_active_fraction"],
        summary["latest_inverse_participation_fraction"],
        summary["latest_entropy_effective_fraction"],
        -summary["latest_temperature_excess_K"],
        -summary["latest_softening_fraction"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-steps", type=int, default=5000)
    args = parser.parse_args()

    results = {}
    for label, temperature, rate in COORDINATES:
        adiabatic_name = f"{label}_seed43_adiabatic"
        control_name = f"{label}_seed43_isothermal"
        adiabatic_dir, control_dir = args.root/adiabatic_name, args.root/control_name
        try:
            history, support, steps, ledgers, parameters, interface, _ = pair_history(
                args.root, adiabatic_name, control_name)
        except ValueError:
            continue
        summary = summarize_pair(
            history, support, steps, interface, args.target_steps,
            float(parameters["dt_strain_step"]))
        runs = {
            "adiabatic": run_terminal_status(adiabatic_dir),
            "control": run_terminal_status(control_dir),
        }
        results[label] = {
            "temperature_K": temperature,
            "strain_rate_s-1": rate,
            "summary": summary,
            "latest_invariants": ledgers[steps[-1]],
            "runs": runs,
            "pair_terminal": all(item["terminal"] for item in runs.values()),
            "pair_successful": all(item["successful"] for item in runs.values()),
        }

    ranked = sorted(results, key=lambda label: rank_key(results[label]))
    expected = len(COORDINATES)
    all_terminal = len(results) == expected and all(
        item["pair_terminal"] for item in results.values())
    all_valid = bool(results) and all(
        item["pair_successful"] and item["latest_invariants"]["passed"]
        for item in results.values())
    strict_candidates = [label for label in ranked if
                         results[label]["summary"]["raw_conjunction"][
                             "qualifying_snapshot_count"] > 0]
    if not all_terminal:
        classification = "TIER1_RUNNING"
    elif not all_valid:
        classification = "TIER1_HARD_INVALID"
    elif strict_candidates:
        classification = "TIER1_STRICT_CANDIDATE_REQUIRES_SEED_AND_GRID"
    else:
        classification = "TIER1_NO_STRICT_LOCALIZATION"
    output = {
        "schema": "asb-drx/v32/asb-adaptive-decision/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": "fb21dbce5e615827060a440759e54b638013f2b2",
        "root": str(args.root),
        "classification": classification,
        "pairs_available": len(results),
        "pairs_expected": expected,
        "all_pairs_terminal": all_terminal,
        "all_pairs_valid": all_valid,
        "ranking": ranked,
        "strict_candidates": strict_candidates,
        "tier2_seed_replication_authorized": bool(
            all_terminal and all_valid and strict_candidates),
        "tier2_selected_coordinate": strict_candidates[0] if (
            all_terminal and all_valid and strict_candidates) else None,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True)+"\n")
    print(classification)


if __name__ == "__main__":
    main()
