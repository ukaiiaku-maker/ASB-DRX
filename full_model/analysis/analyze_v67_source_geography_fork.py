#!/usr/bin/env python3
"""Compare GB/TJ and matched-random Hall--Petch source geography forks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from full_model.analysis.analyze_v67_mobile_supply_collapse import load_case
from full_model.analysis.run_v58_three_grain_production import _load_checkpoint
from full_model.production.arrhenius_dislocation_sources import (
    boundary_tj_site_field, randomized_matched_site_field,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _field_sha(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def classify(parent: dict, boundary: dict, random: dict) -> dict:
    bflow = boundary["flow_exposure_fraction"]
    rflow = random["flow_exposure_fraction"]
    bmobile = boundary["extensive_cell_inventory_m2"]["mobile"]
    rmobile = random["extensive_cell_inventory_m2"]["mobile"]
    same_parent = bool(
        boundary["parent_checkpoint_sha256"]
        == random["parent_checkpoint_sha256"]
        == parent["checkpoint_sha256"])
    hard_valid = bool(
        parent["hard_audits_passed"] and boundary["hard_audits_passed"]
        and random["hard_audits_passed"] and same_parent)
    ratios = {
        "random_to_boundary_flow_exposure": rflow/max(bflow, 1e-300),
        "random_to_boundary_mobile_inventory": rmobile/max(bmobile, 1e-300),
        "random_to_boundary_source_increment": (
            random["source_increment_cell_sum_m2"]
            / max(boundary["source_increment_cell_sum_m2"], 1e-300)),
    }
    if not hard_valid:
        outcome = "SOURCE_GEOGRAPHY_FORK_INVALID"
    elif rflow >= .01 and bflow < .01:
        outcome = "MATCHED_RANDOM_RESTORES_PHYSICAL_FLOW"
    elif (ratios["random_to_boundary_flow_exposure"] >= 2.0
          and ratios["random_to_boundary_mobile_inventory"] >= 1.25):
        outcome = "MATCHED_RANDOM_IMPROVES_SUPPLY_WITHOUT_FLOW_RECOVERY"
    else:
        outcome = "SOURCE_GEOGRAPHY_INSUFFICIENT_TO_PREVENT_FLOW_ARREST"
    return {
        "same_parent_checkpoint": same_parent,
        "hard_valid": hard_valid,
        "ratios": ratios,
        "classification": outcome,
    }


def _child(root: Path, parent_source_total: float) -> dict:
    result = json.loads((root/"result.json").read_text())
    case = load_case(root)
    case.update(
        parent_checkpoint_sha256=result["parent_checkpoint_sha256"],
        configured_source_mode=result["configuration"].get(
            "source_mode", result["decision"].get("source_mode",
                                                  "gb_tj_only")),
        source_increment_cell_sum_m2=(
            case["source_created_line_cell_sum_m2"]-parent_source_total),
    )
    return case


def analyze(parent_root: Path, boundary_root: Path, random_root: Path) -> dict:
    parent_checkpoint = parent_root/"checkpoint_latest.npz"
    parent = load_case(parent_root)
    parent.update(checkpoint_sha256=_sha(parent_checkpoint))
    boundary = _child(
        boundary_root, parent["source_created_line_cell_sum_m2"])
    random = _child(random_root, parent["source_created_line_cell_sum_m2"])

    state, *_ = _load_checkpoint(parent_checkpoint)
    boundary_sites = boundary_tj_site_field(state.supports)
    random_sites = randomized_matched_site_field(boundary_sites, 67)
    site_audit = {
        "seed": 67,
        "shape": list(boundary_sites.shape),
        "boundary_sum": float(np.sum(boundary_sites)),
        "random_sum": float(np.sum(random_sites)),
        "exact_multiset_preserved": bool(np.array_equal(
            np.sort(boundary_sites.ravel()), np.sort(random_sites.ravel()))),
        "spatially_distinct": bool(not np.array_equal(
            boundary_sites, random_sites)),
        "boundary_sha256": _field_sha(boundary_sites),
        "random_sha256": _field_sha(random_sites),
    }
    decision = classify(parent, boundary, random)
    decision["site_measure_exact"] = bool(
        site_audit["exact_multiset_preserved"]
        and site_audit["spatially_distinct"])
    return {
        "schema": "v67-source-geography-fork-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": (
            "Changing only the spatial placement of the exact Hall--Petch "
            "source-opportunity multiset can prevent carrier starvation."),
        "cases": {"parent": parent, "gb_tj_only": boundary,
                  "matched_random": random},
        "source_site_audit_at_parent": site_audit,
        "decision": decision,
        "claim_boundary": (
            "This isolates source geography on one retained high-rate state; "
            "it neither retunes Arrhenius kinetics nor weakens Taylor stress."),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parent", type=Path)
    parser.add_argument("boundary", type=Path)
    parser.add_argument("random", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    payload = analyze(args.parent, args.boundary, args.random)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
