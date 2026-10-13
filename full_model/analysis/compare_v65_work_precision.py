#!/usr/bin/env python3
"""Compare two controlled suffixes on one parent and physical window."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from full_model.analysis.run_v58_three_grain_production import _load_checkpoint
from full_model.analysis.run_v64_adaptive_production import (
    _context, _metrics, _physical_configuration,
)
from full_model.production.coupled_multigrain_interval import make_endpoint


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(value) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=float).ravel()))


def endpoint_errors(parent, reference, candidate, metrics) -> dict:
    result = {}
    for metric in metrics:
        if metric.source == "interval_integral":
            continue
        name = metric.observable_name or metric.name
        left = np.asarray(reference.observables[name], dtype=float)
        right = np.asarray(candidate.observables[name], dtype=float)
        origin = np.asarray(parent.observables[name], dtype=float)
        if metric.source == "endpoint_increment":
            scale = max(norm(left-origin), norm(right-origin),
                        metric.absolute_scale)
        else:
            scale = max(norm(left), norm(right), metric.absolute_scale)
        relative = norm(left-right)/max(scale, np.finfo(float).tiny)
        result[metric.name] = {
            "source": metric.source, "observable_name": name,
            "absolute_l2": norm(left-right), "normalization_l2": scale,
            "relative_difference": relative,
            "working_tolerance": metric.relative_tolerance,
            "passed": bool(relative <= metric.relative_tolerance),
        }
    return result


def ledger_increment_errors(parent, reference, candidate) -> dict:
    fields = (
        "cumulative_mechanical_external_work_J",
        "cumulative_mechanical_internal_energy_change_J",
        "cumulative_mechanical_first_law_residual_J",
        "generated_heat_J", "transformed_volume_m3",
        "cumulative_front_first_law_residual_J",
    )
    result = {}
    for name in fields:
        origin = float(getattr(parent.runtime.ledger, name))
        left = float(getattr(reference.runtime.ledger, name))-origin
        right = float(getattr(candidate.runtime.ledger, name))-origin
        floor = 4096*np.finfo(float).eps*max(abs(origin), 1e-300)
        scale = max(abs(left), abs(right), floor, 1e-300)
        result[name] = {
            "reference_increment": left, "candidate_increment": right,
            "relative_difference": abs(left-right)/scale,
        }
    return result


def load_run(run: Path):
    result_path = run/"result.json"
    result = json.loads(result_path.read_text())
    checkpoint = Path(result["checkpoint"])
    loaded = _load_checkpoint(checkpoint)
    return result, checkpoint, loaded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parent_loaded = _load_checkpoint(args.parent)
    reference_result, reference_path, reference_loaded = load_run(
        args.reference_run)
    candidate_result, candidate_path, candidate_loaded = load_run(
        args.candidate_run)
    pstate, pruntime, _, pload, _, pconfig, _ = parent_loaded
    rstate, rruntime, _, rload, *_ = reference_loaded
    cstate, cruntime, _, cload, *_ = candidate_loaded
    physical = _physical_configuration(pconfig or {})
    context = _context(physical)
    parent = make_endpoint(pstate, pruntime, pload, context)
    reference = make_endpoint(rstate, rruntime, rload, context)
    candidate = make_endpoint(cstate, cruntime, cload, context)
    tolerance = float(reference_result["accuracy_configuration"][
        "state_relative_tolerance"])
    metrics = _metrics(parent, tolerance, tolerance, tolerance)
    errors = endpoint_errors(parent, reference, candidate, metrics)
    ledger = ledger_increment_errors(parent, reference, candidate)
    reference_wall = float(reference_result["wall_time_s"])
    candidate_wall = float(candidate_result["wall_time_s"])
    final_time_difference = abs(
        rruntime.ledger.physical_time_s-cruntime.ledger.physical_time_s)
    support_difference = norm(rstate.supports-cstate.supports)/max(
        norm(rstate.supports), norm(cstate.supports), 1e-300)
    volumes_reference = np.sum(rstate.supports, axis=(1, 2))
    volumes_candidate = np.sum(cstate.supports, axis=(1, 2))
    grain_volume_relative = norm(volumes_reference-volumes_candidate)/max(
        norm(volumes_reference), norm(volumes_candidate), 1e-300)
    payload = {
        "schema": "asb-drx-v65-work-precision-comparison-v1",
        "parent": {"path": str(args.parent.resolve()),
                   "sha256": sha256(args.parent)},
        "reference": {"run": str(args.reference_run.resolve()),
                      "checkpoint": str(reference_path.resolve()),
                      "checkpoint_sha256": sha256(reference_path),
                      "source_commit": reference_result["source_commit"],
                      "wall_time_s": reference_wall},
        "candidate": {"run": str(args.candidate_run.resolve()),
                      "checkpoint": str(candidate_path.resolve()),
                      "checkpoint_sha256": sha256(candidate_path),
                      "source_commit": candidate_result["source_commit"],
                      "wall_time_s": candidate_wall},
        "common_final_time_absolute_difference_s": final_time_difference,
        "endpoint_metric_differences": errors,
        "ledger_increment_differences": ledger,
        "support_signal_relative_l2": support_difference,
        "grain_volume_signal_relative_l2": grain_volume_relative,
        "wall_time_speedup_reference_over_candidate": (
            reference_wall/candidate_wall),
        "candidate_agrees_with_reference_at_working_tolerance": bool(
            final_time_difference <= 128*np.finfo(float).eps*max(
                abs(rruntime.ledger.physical_time_s), 1e-300)
            and all(row["passed"] for row in errors.values())),
        "candidate_is_faster": bool(candidate_wall < reference_wall),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps({
        "output": str(args.output), "sha256": sha256(args.output),
        "agrees": payload[
            "candidate_agrees_with_reference_at_working_tolerance"],
        "speedup": payload["wall_time_speedup_reference_over_candidate"],
    }))


if __name__ == "__main__":
    main()
