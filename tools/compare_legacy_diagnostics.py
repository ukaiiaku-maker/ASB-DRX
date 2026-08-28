#!/usr/bin/env python3
"""Compare two legacy diagnostic CSVs without assigning physical regimes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def as_float(value: str | None) -> float | None:
    try:
        number = float(value) if value is not None else math.nan
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("reproduction", type=Path)
    parser.add_argument("--control", required=True, choices=("v32", "v33", "v34"))
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--rtol", type=float, default=1e-12)
    args = parser.parse_args()

    ref_fields, ref_rows = read(args.reference)
    new_fields, new_rows = read(args.reproduction)
    common = [field for field in ref_fields if field in new_fields]
    column_differences: dict[str, dict[str, float | int | None]] = {}
    mismatch_count = 0
    numeric_mismatch_count = 0
    nonfinite_or_text_mismatch_count = 0
    for field in common:
        abs_diffs: list[float] = []
        rel_diffs: list[float] = []
        compared = 0
        mismatches = 0
        numeric_mismatches = 0
        nonfinite_or_text_mismatches = 0
        for ref_row, new_row in zip(ref_rows, new_rows):
            ref_value = as_float(ref_row.get(field))
            new_value = as_float(new_row.get(field))
            if ref_value is None or new_value is None:
                if ref_row.get(field) != new_row.get(field):
                    mismatches += 1
                    nonfinite_or_text_mismatches += 1
                continue
            compared += 1
            abs_diff = abs(new_value - ref_value)
            scale = max(abs(ref_value), abs(new_value))
            rel_diff = abs_diff / scale if scale else 0.0
            abs_diffs.append(abs_diff)
            rel_diffs.append(rel_diff)
            if abs_diff > args.atol + args.rtol * abs(ref_value):
                mismatches += 1
                numeric_mismatches += 1
        mismatch_count += mismatches
        numeric_mismatch_count += numeric_mismatches
        nonfinite_or_text_mismatch_count += nonfinite_or_text_mismatches
        column_differences[field] = {
            "compared": compared,
            "mismatches": mismatches,
            "numeric_mismatches": numeric_mismatches,
            "nonfinite_or_text_mismatches": nonfinite_or_text_mismatches,
            "max_abs_difference": max(abs_diffs) if abs_diffs else None,
            "max_relative_difference": max(rel_diffs) if rel_diffs else None,
        }

    report = {
        "schema": "asb-drx-legacy-diagnostics-comparison/v1",
        "control": args.control,
        "reference": {"path": str(args.reference.resolve()), "sha256": digest(args.reference)},
        "reproduction": {"path": str(args.reproduction.resolve()), "sha256": digest(args.reproduction)},
        "comparison": {
            "atol": args.atol,
            "rtol": args.rtol,
            "same_byte_hash": digest(args.reference) == digest(args.reproduction),
            "same_field_order": ref_fields == new_fields,
            "reference_rows": len(ref_rows),
            "reproduction_rows": len(new_rows),
            "common_columns": len(common),
            "total_cell_mismatches": mismatch_count,
            "numeric_tolerance_mismatches": numeric_mismatch_count,
            "nonfinite_or_text_mismatches": nonfinite_or_text_mismatch_count,
            "within_tolerance": (
                ref_fields == new_fields
                and len(ref_rows) == len(new_rows)
                and mismatch_count == 0
            ),
        },
        "column_differences": column_differences,
        "interpretation": "Numerical reproducibility only; this comparison assigns no DRX or ASB status.",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
