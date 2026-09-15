#!/usr/bin/env python3
"""Compare common numerical checkpoint fields across a production-source edit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


V21_ONLY = {
    "P_json", "v21_junction_m2", "v21_channel_exposure_json",
    "v21_balance_ledger_json", "v21_common_parameters_json",
    "v21_topology_json",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--reference-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    differences = []
    tested = []
    with np.load(args.candidate, allow_pickle=True) as candidate, np.load(
            args.reference, allow_pickle=True) as reference:
        for name in sorted((set(candidate.files) & set(reference.files))-V21_ONLY):
            left, right = np.asarray(candidate[name]), np.asarray(reference[name])
            if left.dtype.kind in "OUS" or right.dtype.kind in "OUS":
                continue
            tested.append(name)
            if not np.array_equal(left, right, equal_nan=True):
                differences.append({
                    "field": name,
                    "maximum_absolute_difference": float(np.nanmax(
                        np.abs(left-right))),
                })
    passed = not differences
    result = {
        "schema": "asb-drx/v21-opt-in-legacy-regression/v1",
        "candidate_commit": args.candidate_commit,
        "reference_commit": args.reference_commit,
        "candidate_checkpoint_sha256": digest(args.candidate),
        "reference_checkpoint_sha256": digest(args.reference),
        "common_numeric_fields_tested": len(tested),
        "excluded_schema_only_fields": sorted(V21_ONLY),
        "nonidentical_fields": differences,
        "bitwise_legacy_trajectory_passed": passed,
        "fixture_passed": passed,
        "scientific_gate_passed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"fields": len(tested), "passed": passed}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
