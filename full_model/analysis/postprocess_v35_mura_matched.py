#!/usr/bin/env python3
"""Postprocess matched-time V34-boolean and V35-feasible Mura continuations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_case(path):
    path = Path(path)
    status = json.loads((path/"status.json").read_text())
    history = [json.loads(line) for line in (path/"history.jsonl").read_text(
    ).splitlines() if line.strip()]
    checkpoints = sorted(path.glob("checkpoint_step_*.npz"))
    return {
        "path": str(path), "status": status,
        "initial_checkpoint": checkpoints[0].name,
        "initial_checkpoint_sha256": sha256(checkpoints[0]),
        "final_checkpoint": checkpoints[-1].name,
        "final_checkpoint_sha256": sha256(checkpoints[-1]),
        "history_sha256": sha256(path/"history.jsonl"),
        "first_history": history[0], "last_history": history[-1],
        "accepted_intervals": int(status["step"])-2971,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-v34", type=Path, required=True)
    parser.add_argument("--feasible-v35", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old = load_case(args.frozen_v34); new = load_case(args.feasible_v35)
    keys = (
        "maximum_signed_density_m2", "orientation_span_deg",
        "dual_nye_relative_rms", "normalized_line_continuity_residual",
        "structure_factor_peak_fraction",
    )
    differences = {key: new["last_history"][key]-old["last_history"][key]
                   for key in keys}
    common_start = old["initial_checkpoint_sha256"] == new[
        "initial_checkpoint_sha256"]
    matched_time = abs(old["status"]["physical_time_s"]-new["status"][
        "physical_time_s"])
    time_scale = max(abs(old["status"]["physical_time_s"]), 1e-300)
    hard_valid = all(
        case["status"]["status"] == "COMPLETED"
        and case["last_history"]["accepted_step_hard_invariant_passed"]
        and not case["last_history"]["post_step_projection_used"]
        and case["last_history"]["minimum_heat_increment_J_m3"] >= 0.0
        for case in (old, new))
    result = {
        "schema": "asb-drx/v35-mura-matched-time-comparison/v1",
        "analysis_script_sha256": sha256(Path(__file__)),
        "fixture_passed": bool(common_start and hard_valid
                               and matched_time/time_scale < 1e-6),
        "scientific_gate_passed": False,
        "scientific_gate_not_passed_reason": (
            "the short continuation discriminates selection dynamics but does "
            "not establish long-time rate or spatial convergence"),
        "common_initial_checkpoint_exact": common_start,
        "elapsed_time_mismatch_s": matched_time,
        "final_metric_v35_minus_v34": differences,
        "frozen_v34": old, "feasible_v35": new,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"output": str(args.output),
                      "fixture_passed": result["fixture_passed"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
