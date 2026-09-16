#!/usr/bin/env python3
"""Postprocess the immutable V32 repaired 64/128 matched continuation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from full_model.analysis.postprocess_v31_mura_b1 import checkpoint_metrics


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def history(path):
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def relative(a, b):
    return abs(float(a)-float(b))/max(abs(float(a)), abs(float(b)), 1e-30)


def summarize(case_dir):
    status = json.loads((case_dir/"status.json").read_text())
    records = history(case_dir/"history.jsonl")
    checkpoint = case_dir/status["latest_checkpoint"]
    _, metrics, fields = checkpoint_metrics(
        checkpoint, status["condition"], int(status["grid"]),
        float(status["length_m"]))
    invariant = all(row["accepted_step_hard_invariant_passed"]
                    and not row["post_step_projection_used"] for row in records)
    return {
        "case": case_dir.name, "grid": int(status["grid"]),
        "status": status["status"], "step": int(status["step"]),
        "applied_strain": float(status["applied_strain"]),
        "physical_time_s": float(status["physical_time_s"]),
        "wall_seconds": float(status["wall_seconds_this_invocation"]),
        "source_sha": status["source_sha"],
        "latest_checkpoint": checkpoint.name,
        "latest_checkpoint_sha256": sha256(checkpoint),
        "history_records": len(records),
        "all_recorded_hard_invariants_passed": invariant,
        "maximum_dual_nye_relative_rms": max(
            row["dual_nye_relative_rms"] for row in records),
        "maximum_line_continuity_residual": max(
            row["normalized_line_continuity_residual"] for row in records),
        "maximum_first_law_relative": max(
            row["energy_balance_relative"] for row in records),
        "minimum_recorded_event_scale": min(
            row["mura_event_scale"] for row in records),
        "recorded_full_stall_count": sum(
            bool(row["mura_physical_stall"]) for row in records),
        "final_metrics": metrics,
        "cumulative_ledger": status["cumulative_ledger"],
        "_records": records, "_fields": fields,
    }


def plot(cases, path):
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    for case in cases:
        records = case["_records"]
        x = [row["applied_strain"] for row in records]
        label = str(case["grid"])
        axes[0, 0].plot(x, [row["dual_nye_relative_rms"] for row in records],
                        label=label)
        axes[0, 1].plot(x, [row["maximum_signed_density_m2"] for row in records],
                        label=label)
        axes[0, 2].plot(x, [row["orientation_span_deg"] for row in records],
                        label=label)
        axes[0, 3].plot(x, [row["mura_event_scale"] for row in records],
                        label=label)
    for column, (case, field, title) in enumerate((
            (cases[0], "signed_norm_m2", "signed density"),
            (cases[0], "alpha_norm_m1", "Nye magnitude"),
            (cases[1], "signed_norm_m2", "signed density"),
            (cases[1], "alpha_norm_m1", "Nye magnitude"))):
        axes[1, column].imshow(case["_fields"][field].T, origin="lower",
                               cmap="magma", interpolation="nearest")
        axes[1, column].set_title(f"{title}, {case['grid']}²")
        axes[1, column].set_xticks([]); axes[1, column].set_yticks([])
    axes[0, 0].set_ylabel("dual Nye relative RMS")
    axes[0, 1].set_ylabel("maximum signed density (m$^{-2}$)")
    axes[0, 2].set_ylabel("orientation span (deg)")
    axes[0, 3].set_ylabel("accepted scalar event scale")
    for axis in axes[0]:
        axis.set_xlabel("applied strain"); axis.legend(title="grid")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--expected-source-sha", required=True,
                        help="immutable production commit used by both cases")
    args = parser.parse_args()
    case_dirs = sorted((args.run_root/"cases").glob("mechanical_heterogeneity_*"))
    if len(case_dirs) != 2:
        raise RuntimeError("V32 matched decision requires exactly two cases")
    cases = [summarize(path) for path in case_dirs]
    expected_source = args.expected_source_sha
    source_ok = all(case["source_sha"] == expected_source for case in cases)
    complete = all(case["status"] == "COMPLETED"
                   and case["applied_strain"] >= .05 for case in cases)
    metric_names = ("dual_nye_relative_rms", "alpha_rms_m1",
                    "maximum_signed_density_m2", "mean_total_density_m2",
                    "orientation_span_deg", "slip_rms",
                    "structure_factor_peak_fraction")
    pair = {name: relative(cases[0]["final_metrics"][name],
                           cases[1]["final_metrics"][name])
            for name in metric_names}
    convergence = all(value < .05 for value in pair.values())
    invariants = all(case["all_recorded_hard_invariants_passed"]
                     and case["maximum_first_law_relative"] < 2e-11
                     and min(row["minimum_heat_increment_J_m3"]
                             for row in case["_records"]) >= 0.0
                     for case in cases)
    qualified = bool(source_ok and complete and convergence and invariants)
    classification = ("V32_REPAIRED_MATCHED_CONTINUATION_QUALIFIED" if qualified
                      else "V32_REPAIR_VALID_BUT_SPATIAL_CONVERGENCE_FAILED"
                      if source_ok and complete and invariants
                      else "V32_MATCHED_CONTINUATION_INCOMPLETE_OR_INVALID")
    serial = []
    for case in cases:
        item = {key: value for key, value in case.items()
                if not key.startswith("_")}
        serial.append(item)
    result = {
        "schema": "asb-drx/v32-mura-matched-decision/v1",
        "immutable_source_sha": expected_source,
        "source_verified": source_ok,
        "cases": serial,
        "final_grid_pair_relative_differences": pair,
        "all_declared_observables_below_5_percent": convergence,
        "hard_invariants_passed": invariants,
        "scientific_gate_passed": qualified,
        "classification": classification,
        "broad_b2_authorized": qualified,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    fields = ("case", "grid", "status", "step", "applied_strain",
              "wall_seconds", "latest_checkpoint", "latest_checkpoint_sha256",
              "maximum_dual_nye_relative_rms", "maximum_first_law_relative",
              "minimum_recorded_event_scale", "recorded_full_stall_count")
    with args.index.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for case in serial:
            writer.writerow({name: case.get(name) for name in fields})
    plot(cases, args.figure)
    print(json.dumps({"classification": classification,
                      "broad_b2_authorized": qualified}, sort_keys=True))


if __name__ == "__main__":
    main()
