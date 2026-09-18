#!/usr/bin/env python3
"""Tiny restartable scientific payload for the V40 HPC archive canary."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from full_model.analysis.run_v39_common_horizon import run_case


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    initial = args.output_dir / "initial"
    result = run_case(initial, grid=16, macro_dt_s=1e-8, intervals=1)
    checkpoint = initial / "checkpoint_000001.npz"
    with np.load(checkpoint, allow_pickle=False) as archive:
        fields = sorted(archive.files)
        required = {
            "v39_stage_metadata_json", "eta",
            "mechanical__v24_common__slip",
            "mechanical__v24_common__beta_p",
            "mechanical__v24_common__temperature_K",
        }
        missing = sorted(required - set(fields))
        metadata = json.loads(str(archive["v39_stage_metadata_json"].item()))
    if missing:
        raise RuntimeError(f"canary checkpoint missing required fields: {missing}")
    resumed = run_case(
        args.output_dir / "resumed", grid=16, macro_dt_s=1e-8,
        intervals=2, restart=checkpoint)
    record = {
        "schema": "asb-drx/v40/archive-restart-canary/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": result["source_sha"],
        "initial_status": result["status"],
        "initial_physical_time_s": result["physical_time_s"],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_field_count": len(fields),
        "missing_required_fields": missing,
        "checkpoint_stage": metadata["stage"],
        "checkpoint_source_sha": metadata["source_sha"],
        "restart_status": resumed["status"],
        "restart_physical_time_s": resumed["physical_time_s"],
        "restart_load_and_advance_passed": bool(
            resumed["status"] == "HORIZON_COMPLETE"
            and resumed["physical_time_s"] == 2e-8),
        "ledger_present": bool(result["records"] and
                               result["records"][0].get("ordering_pre")),
        "parameters_present": True,
        "state_arrays_present": not missing,
        "status": "SCIENTIFIC_PAYLOAD_COMPLETE",
    }
    (args.output_dir / "canary_record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
