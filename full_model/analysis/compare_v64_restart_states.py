#!/usr/bin/env python3
"""Compare physical checkpoint state across an adaptive restart boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from full_model.analysis.run_v58_three_grain_production import _load_checkpoint
from full_model.production.multigrain_common_state import multigrain_state_digest


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(left: Path, right: Path) -> dict:
    left_loaded = _load_checkpoint(left)
    right_loaded = _load_checkpoint(right)
    left_state, left_runtime, left_step, left_load = left_loaded[:4]
    right_state, right_runtime, right_step, right_load = right_loaded[:4]
    array_rows = {}
    with np.load(left, allow_pickle=False) as a, np.load(
            right, allow_pickle=False) as b:
        # Diagnostics contain timings and provenance.  Physical state arrays,
        # initial volumes, and the runtime ledger are checked independently.
        physical = sorted(
            key for key in set(a.files) & set(b.files)
            if not key.startswith("diagnostic_")
            and key not in {"metadata_json", "runtime_json"})
        for key in physical:
            left_value = np.asarray(a[key]); right_value = np.asarray(b[key])
            if left_value.shape != right_value.shape:
                array_rows[key] = {"shape_equal": False, "array_equal": False}
                continue
            difference = (np.asarray(left_value, dtype=float)
                          -np.asarray(right_value, dtype=float))
            array_rows[key] = {
                "shape_equal": True,
                "array_equal": bool(np.array_equal(left_value, right_value)),
                "maximum_absolute_difference": float(
                    np.max(np.abs(difference), initial=0.0)),
            }
        runtime_json_equal = bool(np.array_equal(
            np.asarray(a["runtime_json"]), np.asarray(b["runtime_json"])))
    state_digest_left = multigrain_state_digest(left_state)
    state_digest_right = multigrain_state_digest(right_state)
    all_arrays_equal = all(row["array_equal"] for row in array_rows.values())
    return {
        "schema": "asb-drx-v64-restart-state-comparison-v1",
        "left": {"path": str(left.resolve()), "sha256": sha256(left),
                 "state_digest": state_digest_left},
        "right": {"path": str(right.resolve()), "sha256": sha256(right),
                  "state_digest": state_digest_right},
        "physical_state_digest_equal": state_digest_left == state_digest_right,
        "physical_arrays_all_bitwise_equal": all_arrays_equal,
        "runtime_json_bitwise_equal": runtime_json_equal,
        "step_equal": left_step == right_step,
        "load_bitwise_equal": left_load == right_load,
        "runtime_dataclass_repr_equal": repr(left_runtime) == repr(right_runtime),
        "physical_array_comparisons": array_rows,
        "comparison_scope": (
            "physical arrays, initial volumes, clocks, ledgers, load, and state "
            "digest; diagnostic wall times and source provenance are excluded"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.left, args.right)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({key: result[key] for key in (
        "physical_state_digest_equal", "physical_arrays_all_bitwise_equal",
        "runtime_json_bitwise_equal", "step_equal", "load_bitwise_equal")}))


if __name__ == "__main__":
    main()
