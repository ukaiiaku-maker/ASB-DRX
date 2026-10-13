#!/usr/bin/env python3
"""Merge restart-segment histories with explicit gaps and conflicts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge_histories(paths: list[Path]) -> dict:
    by_interval = {}
    sources = []
    duplicate_intervals = []
    for path in paths:
        rows = json.loads(path.read_text())
        sources.append({"path": str(path.resolve()), "sha256": sha256(path),
                        "row_count": len(rows)})
        for row in rows:
            index = int(row["adaptive_interval"])
            if index in by_interval:
                if row != by_interval[index]:
                    raise ValueError(
                        f"conflicting records for adaptive interval {index}")
                duplicate_intervals.append(index)
            else:
                by_interval[index] = row
    ordered = [by_interval[index] for index in sorted(by_interval)]
    gaps = []
    for left, right in zip(ordered, ordered[1:]):
        a = int(left["adaptive_interval"])
        b = int(right["adaptive_interval"])
        if b != a+1:
            gaps.append({"after_interval": a, "before_interval": b,
                         "missing_count": b-a-1})
        if float(right["physical_time_s"]) <= float(left["physical_time_s"]):
            raise ValueError("merged physical time is not strictly increasing")
    return {
        "schema": "asb-drx-v65-merged-adaptive-history-v1",
        "sources": sources,
        "duplicate_intervals": sorted(set(duplicate_intervals)),
        "gaps": gaps,
        "history_complete": not gaps,
        "rows": ordered,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = merge_histories(args.history)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2)+"\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output),
                      "rows": len(payload["rows"]),
                      "history_complete": payload["history_complete"],
                      "gaps": payload["gaps"]}))


if __name__ == "__main__":
    main()
