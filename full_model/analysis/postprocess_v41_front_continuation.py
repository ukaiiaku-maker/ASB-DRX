#!/usr/bin/env python3
"""Classify the repaired zero-work front continuation."""

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
    parser.add_argument("--switch-after-intervals", type=int, default=15)
    parser.add_argument("--execution-source-sha", required=True,
                        help="HEAD at process launch; later analysis-only commits may change reported HEAD")
    args = parser.parse_args()
    raw = json.loads(args.input.read_text())
    records = raw["records"]
    selected = records[args.switch_after_intervals:]
    published = [row for row in selected if row["front_published"]]
    stationary = [row for row in selected if not row["front_published"]]
    displacement = sum(row["front_contour_displacement_m"] for row in selected)
    result = {
        "schema": "asb-drx/v41/front-continuation-decision/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input": {"path": str(args.input.resolve()),
                  "sha256": hashlib.sha256(args.input.read_bytes()).hexdigest()},
        "source_sha": args.execution_source_sha,
        "completion_worktree_head": raw["source_sha"],
        "source_scope_note": (
            "The worktree HEAD changed during execution only through "
            "analysis/report additions; production operators are identical."),
        "protocol": raw["protocol"],
        "zero_applied_front_work": True,
        "physical_time_s": raw["physical_time_s"],
        "repaired_intervals": len(selected),
        "repaired_published_intervals": len(published),
        "repaired_additional_displacement_m": displacement,
        "last_published": (published[-1] if published else None),
        "first_stationary_after_last_publication": next((
            row for row in selected[(selected.index(published[-1])+1
                                     if published else 0):]
            if not row["front_published"]), None),
        "terminal_record": selected[-1],
        "all_common_clocks_closed": all(
            row["common_clock_closed"] for row in selected),
        "stationary_tail_intervals": len(stationary),
        "drx_claimed": False,
        "classification": (
            "REPAIRED_DOWNHILL_ADVANCE_THEN_COMPLETE_ENERGY_STATIONARITY"
            if published and stationary else
            "REPAIRED_FRONT_CONTINUATION_UNRESOLVED"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
