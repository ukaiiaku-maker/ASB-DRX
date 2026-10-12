#!/usr/bin/env python3
"""Verify exact complete-map equivalence after invariant-energy reuse."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from full_model.analysis.run_v58_three_grain_production import _load_checkpoint
from full_model.analysis.run_v64_adaptive_production import (
    _context, _physical_configuration,
)
from full_model.production.coupled_multigrain_interval import (
    advance_complete_multigrain_interval, make_endpoint,
)
from full_model.production.multigrain_common_state import multigrain_state_digest


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--expected-parent-sha256", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--expected-reference-sha256", required=True)
    parser.add_argument("--accepted-duration", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.parent) != args.expected_parent_sha256:
        raise ValueError("parent checksum mismatch")
    if digest(args.reference) != args.expected_reference_sha256:
        raise ValueError("reference checksum mismatch")
    state, runtime, _, load, *_tail = _load_checkpoint(args.parent)
    reference_state, reference_runtime, _, reference_load, *_ = (
        _load_checkpoint(args.reference))
    configuration = _tail[-2]
    context = _context(_physical_configuration(configuration))
    endpoint = make_endpoint(state, runtime, load, context)
    half = .5*float(args.accepted_duration)
    started = time.perf_counter()
    first = advance_complete_multigrain_interval(endpoint, half, context)
    second = advance_complete_multigrain_interval(first.endpoint, half, context)
    wall = time.perf_counter()-started
    computed_digest = multigrain_state_digest(second.endpoint.state)
    reference_digest = multigrain_state_digest(reference_state)
    result = {
        "schema": "asb-drx-v64-optimization-equivalence-v1",
        "analysis_source_commit": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), text=True).strip(),
        "parent": {"path": str(args.parent.resolve()),
                   "sha256": args.expected_parent_sha256},
        "reference": {"path": str(args.reference.resolve()),
                      "sha256": args.expected_reference_sha256},
        "accepted_path": "two_complete_half_intervals",
        "accepted_duration_s": args.accepted_duration,
        "computed_state_digest": computed_digest,
        "reference_state_digest": reference_digest,
        "state_digest_bitwise_equal": computed_digest == reference_digest,
        "runtime_repr_bitwise_equal": (
            repr(second.endpoint.runtime) == repr(reference_runtime)),
        "load_bitwise_equal": second.endpoint.load == reference_load,
        "load_absolute_difference": abs(second.endpoint.load-reference_load),
        "wall_time_s": wall,
        "map_cost_breakdown_s": {
            "first_half": first.audit["cost_breakdown_s"],
            "second_half": second.audit["cost_breakdown_s"],
        },
        "physical_change": "none; exact invariant-energy reuse only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({key: result[key] for key in (
        "state_digest_bitwise_equal", "runtime_repr_bitwise_equal",
        "load_bitwise_equal", "wall_time_s")}))


if __name__ == "__main__":
    main()
