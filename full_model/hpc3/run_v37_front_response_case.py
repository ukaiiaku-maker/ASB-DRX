#!/usr/bin/env python3
"""Run one preregistered V37 production front trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from full_model.analysis.run_v36_recurrent_physical_response import run_response


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--case-table", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--intervals", type=int, default=40)
    args = parser.parse_args()
    cases = json.loads(args.case_table.read_text())
    if not 0 <= args.case_index < len(cases):
        raise ValueError("case index outside registered V37 front table")
    case = dict(cases[args.case_index])
    case_id = case.pop("id")
    output = args.run_root/case_id
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(output.glob("checkpoint_*.npz"))
    resume = checkpoints[-1] if checkpoints else None
    result = run_response(
        output_dir=output, protocol="continued_deformation",
        grid=args.grid, intervals=args.intervals, dt_s=5.0e-6,
        initial_shear=.01, strain_rate_s=100.0, temperature_K=1100.0,
        child_line_fraction=.35, length_m=3.2e-6,
        interface_width_m=4.0e-7, proposal_fraction=.0625,
        proposal_direction=1, front_enabled=True, mura_enabled=True,
        checkpoint_every=5, resume=resume, **case)
    record = {
        "schema": "asb-drx/v37/front-response-case/v1",
        "case_index": args.case_index,
        "case_id": case_id,
        "parameters": case,
        "case_table_sha256": digest(args.case_table),
        "production_source_commit": os.environ.get("V37_FRONT_SOURCE_SHA"),
        "hpc3_run_id": os.environ.get("HPC3_RUN_ID"),
        "grid": args.grid,
        "requested_intervals": args.intervals,
        "physical_horizon_s": args.intervals*5.0e-6,
        "loading_protocol": "continued_deformation",
        "strain_rate_s": 100.0,
        "result_sha256": result["result_sha256"],
        "completed_intervals": result["completed_intervals"],
        "terminal": result["completed_intervals"] == args.intervals
    }
    (output/"v37_front_run_record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
