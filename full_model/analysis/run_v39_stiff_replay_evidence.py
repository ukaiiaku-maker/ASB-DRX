#!/usr/bin/env python3
"""Reproduce and summarize the transactional V39 post-front recovery."""

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
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    failed = args.run_root / "injected_failure"
    try:
        run_case(failed, grid=16, macro_dt_s=1e-3, intervals=1,
                 inject_post_front_failure=True)
    except RuntimeError as error:
        if "injected post-front failure" not in str(error):
            raise
    terminal = json.loads((failed / "terminal.json").read_text())
    partial = failed / "partial_000001_post_front.npz"
    resumed = run_case(args.run_root / "resumed", grid=16,
                       macro_dt_s=1e-3, intervals=1, restart=partial)
    continuous = run_case(args.run_root / "continuous", grid=16,
                          macro_dt_s=1e-3, intervals=1)
    resumed_checkpoint = args.run_root / "resumed" / "checkpoint_000001.npz"
    continuous_checkpoint = args.run_root / "continuous" / "checkpoint_000001.npz"
    unequal = []
    with np.load(resumed_checkpoint) as left, np.load(continuous_checkpoint) as right:
        shared = sorted(set(left.files) & set(right.files)
                        - {"v39_stage_metadata_json"})
        for name in shared:
            if not np.array_equal(left[name], right[name]):
                unequal.append(name)
    record = resumed["records"][0]
    result = {
        "schema": "asb-drx/v39/stiff-post-front-replay/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": resumed["source_sha"],
        "former_terminal_reproduced": True,
        "failure_stage": terminal["partial_stage"],
        "accepted_complete_macros_at_failure": terminal[
            "accepted_complete_macros"],
        "partial_checkpoint": str(partial.resolve()),
        "partial_checkpoint_sha256": sha256(partial),
        "resumed_without_repeating_front": record[
            "resumed_without_repeating_front"],
        "external_physical_time_s": resumed["physical_time_s"],
        "mura_operator_exposure_s": record["mura_operator_exposure_s"],
        "front_operator_exposure_s": record["front_operator_exposure_s"],
        "ordering_pre": record["ordering_pre"],
        "ordering_post": record["ordering_post"],
        "shared_checkpoint_fields": len(shared),
        "unequal_shared_checkpoint_fields": unequal,
        "continuous_restart_byte_exact": not unequal,
        "resumed_checkpoint_sha256": sha256(resumed_checkpoint),
        "continuous_checkpoint_sha256": sha256(continuous_checkpoint),
        "status": "HORIZON_COMPLETE_RESTART_EXACT",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
