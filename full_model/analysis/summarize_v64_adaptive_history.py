#!/usr/bin/env python3
"""Reduce V64 adaptive histories without losing error-source semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(run: Path) -> dict:
    history_path = run/"history.json"
    result_path = run/"result.json"
    history = json.loads(history_path.read_text())
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    attempts = [attempt for row in history
                for attempt in row["accuracy_decision"]["attempts"]]
    metrics = sorted({name for attempt in attempts
                      for name in attempt.get("metric_errors", {})})
    metric_maxima = {}
    for name in metrics:
        rows = [attempt["metric_errors"][name] for attempt in attempts
                if name in attempt.get("metric_errors", {})]
        limiting = max(rows, key=lambda item: item["normalized_error"])
        metric_maxima[name] = {
            "source": limiting["source"],
            "maximum_normalized_error": limiting["normalized_error"],
            "tolerance": limiting["relative_tolerance"],
            "maximum_tolerance_ratio": (
                limiting["normalized_error"]/limiting["relative_tolerance"]),
        }
    cost = {name: 0.0 for name in (
        "mechanical_map", "front_map", "endpoint_observables",
        "accounted_total")}
    complete_maps = 0
    for attempt in attempts:
        for trial in attempt.get("trial_cost_breakdown_s", {}).values():
            if trial is None:
                continue
            complete_maps += 1
            for name in cost:
                cost[name] += float(trial[name])
    accounted = max(cost["accounted_total"], 1e-300)
    published_leaves = []
    def collect(audit):
        if "first_half" in audit and "second_half" in audit:
            collect(audit["first_half"]); collect(audit["second_half"])
        else:
            published_leaves.append(audit)
    for row in history:
        collect(row["published_interval_audit"])
    front_work = {
        name: sum(sum(leaf.get(name, [])) for leaf in published_leaves)
        for name in (
            "front_force_rate_iterations",
            "front_complete_energy_evaluations",
            "front_independent_directional_prices",
            "front_joint_capacity_solves",
        )
    }
    durations = [float(row["accepted_duration_s"]) for row in history]
    median_duration = statistics.median(durations)
    ordinary = [row for row in history
                if float(row["accepted_duration_s"]) >= .5*median_duration]
    runner_wall = None if result is None else float(result["wall_time_s"])
    return {
        "run": str(run.resolve()),
        "history_sha256": digest(history_path),
        "result_sha256": digest(result_path) if result_path.exists() else None,
        "accepted_intervals": len(history),
        "accuracy_rejected_attempts": sum(
            row.get("classification") == "REJECTED_ACCURACY"
            for row in attempts),
        "admissibility_failed_attempts": sum(
            row.get("classification") == "ADMISSIBILITY_FAILURE"
            for row in attempts),
        "accepted_duration_s": sum(row["accepted_duration_s"] for row in history),
        "start_time_s": (history[0]["physical_time_s"]
                         -history[0]["accepted_duration_s"]),
        "final_time_s": history[-1]["physical_time_s"],
        "maximum_accepted_tolerance_ratio": max(
            row["maximum_tolerance_ratio"] for row in history),
        "metric_maxima_over_all_accuracy_attempts": metric_maxima,
        "complete_map_evaluations_with_cost_data": complete_maps,
        "complete_map_cost_s": cost,
        "complete_map_cost_fractions": {
            name: cost[name]/accounted for name in (
                "mechanical_map", "front_map", "endpoint_observables")},
        "published_fine_path_front_work": front_work,
        "median_accepted_duration_s": median_duration,
        "ordinary_interval_count": len(ordinary),
        "ordinary_median_wall_time_s": statistics.median(
            float(row["wall_time_s"]) for row in ordinary),
        "ordinary_median_wall_seconds_per_physical_ns": statistics.median(
            float(row["wall_time_s"])/float(row["accepted_duration_s"])*1e-9
            for row in ordinary),
        "runner_wall_time_s": runner_wall,
        "runner_wall_seconds_per_physical_ns": (
            None if runner_wall is None else
            runner_wall/max(sum(durations)*1e9, 1e-300)),
        "controller_serialization_checkpoint_io_overhead_s": (
            None if runner_wall is None else
            max(0.0, runner_wall-cost["accounted_total"])),
        "execution_status": None if result is None else result["execution_status"],
        "inherited_prefix_uncontrolled_by_v64": (
            None if result is None else
            result["inherited_prefix_uncontrolled_by_v64"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "schema": "asb-drx-v64-adaptive-history-summary-v1",
        "runs": [summarize(run) for run in args.run],
        "cost_scope": (
            "instrumented complete maps only; runner wall time additionally "
            "contains controller, serialization, checkpoint compression, and I/O"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"output": str(args.output), "sha256": digest(args.output)}))


if __name__ == "__main__":
    main()
