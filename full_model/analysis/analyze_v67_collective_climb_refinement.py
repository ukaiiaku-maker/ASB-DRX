#!/usr/bin/env python3
"""Classify fixed-scale refinement of the collective-climb discriminator."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("n32", type=Path)
    parser.add_argument("n64", type=Path)
    parser.add_argument("n128", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--represented-thickness-m", type=float, required=True)
    parser.add_argument("--relative-tolerance", type=float, default=.05)
    args = parser.parse_args()
    rows = []
    for grid, path in ((32, args.n32), (64, args.n64), (128, args.n128)):
        payload = json.loads(path.read_text())
        best = payload["decision"]["best_joint_candidate"]
        spacing = float(payload["configuration"]["spacing_m"])
        moved = float(best["moved_line_cell_sum_m2"])
        moved_line_m = moved*spacing**2*args.represented_thickness_m
        delta = float(best["joint_delta_helmholtz_J"])
        rows.append({
            "grid": grid,
            "path": str(path.resolve()),
            "sha256": _sha(path),
            "source_commit": payload["source_commit"],
            "source_tree_clean": payload["source_tree_clean"],
            "spacing_m": spacing,
            "candidate_count": payload["decision"]["candidate_count"],
            "downhill_candidate_count": payload["decision"][
                "joint_downhill_count"],
            "best_joint_delta_helmholtz_J": delta,
            "best_moved_line_m": moved_line_m,
            "best_affinity_J_m": (-delta/moved_line_m
                                    if moved_line_m > 0.0 else 0.0),
            "best_candidate": best,
        })
    affinities = [row["best_affinity_J_m"] for row in rows]
    signs_stable = all(value > 0.0 for value in affinities)
    successive = [
        abs(right-left)/max(abs(right), abs(left), 1e-300)
        for left, right in zip(affinities[:-1], affinities[1:])]
    passed = bool(
        signs_stable and all(value <= args.relative_tolerance
                             for value in successive)
        and all(row["source_tree_clean"] for row in rows))
    output = {
        "schema": "v67-collective-climb-fixed-scale-refinement-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "represented_thickness_m": args.represented_thickness_m,
        "relative_tolerance": args.relative_tolerance,
        "cases": rows,
        "decision": {
            "downhill_sign_stable": signs_stable,
            "successive_affinity_relative_changes": successive,
            "provisional_five_percent_refinement_passed": passed,
            "production_promotion_authorized": passed,
            "classification": (
                "GRID_CONVERGED_COLLECTIVE_CLIMB_CANDIDATE" if passed else
                "REJECTED_NONCONVERGED_COLLECTIVE_CLIMB_CANDIDATE"),
        },
        "claim_boundary": (
            "A passing static discriminator would authorize kinetic and "
            "production qualification, not establish DRX by itself."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True)+"\n")
    print(json.dumps(output["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
