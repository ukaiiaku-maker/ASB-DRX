#!/usr/bin/env python3
"""Publish compact, checksum-verified records for retained V47 local bundles."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_run_checkpoints(rows):
    checked = {}
    for key, row in rows.items():
        path = Path(row["checkpoint"])
        actual = digest(path)
        if actual != row["checkpoint_sha256"]:
            raise ValueError(f"checkpoint checksum mismatch: {path}")
        checked[key] = {
            name: row[name] for name in row if name not in ("records", "diagnostics")
        }
    return checked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--short", type=Path, required=True)
    parser.add_argument("--temporal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    short = json.loads(args.short.read_text())
    temporal = json.loads(args.temporal.read_text())
    payload = {
        "schema": "asb-drx/v47/local-bundle-summary/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "short_refinement": {
            key: short[key] for key in (
                "source_sha", "classification", "scope", "domain_m",
                "representation_length_m", "physical_interval_s",
                "full_62p5us_n256_completed", "full_horizon_exclusion",
                "comparisons")
        },
        "temporal_refinement": {
            key: temporal[key] for key in (
                "source_sha", "classification", "parent_checkpoint",
                "parent_checkpoint_sha256", "parent_physical_time_s",
                "operator_exposure_s", "common_band_half_width",
                "relative_differences")
        },
        "local_bundle_manifests": {
            "short": {"path": str(args.short.resolve()),
                      "sha256": digest(args.short)},
            "temporal": {"path": str(args.temporal.resolve()),
                         "sha256": digest(args.temporal)},
        },
        "checkpoint_records": {
            "short": validate_run_checkpoints(short["runs"]),
            "temporal": validate_run_checkpoints(temporal["runs"]),
        },
        "execution_location": "local",
        "hpc3_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"short": short["classification"],
                      "temporal": temporal["classification"],
                      "sha256": digest(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
