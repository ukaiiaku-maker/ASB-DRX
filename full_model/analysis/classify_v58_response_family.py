#!/usr/bin/env python3
"""Classify the matched V58 front/flow-temperature response family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(root, common_step=None, common_time_s=None):
    root = Path(root)
    result_path = root/"result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    class_path = (root/"classification.json" if result_path.exists()
                  else root/"classification_partial.json")
    classification = json.loads(class_path.read_text())
    history = json.loads((root/"history.json").read_text())
    if common_time_s is not None:
        scale = max(abs(float(common_time_s)), 1e-300)
        matches = [item for item in history
                   if abs(float(item["time_s"])-float(common_time_s))/scale
                   <= 1e-10]
        if not matches:
            raise ValueError(
                f"{root} does not reach common physical time {common_time_s}")
        row = matches[-1]
    elif common_step is None:
        row = history[-1]
    else:
        matches = [item for item in history if int(item["step"]) == common_step]
        if not matches:
            raise ValueError(f"{root} does not reach common step {common_step}")
        row = matches[-1]
    source = (result["source_commit"] if result is not None
              else classification["source_commit"])
    return {"result": result, "classification": classification,
            "row": row, "source": source,
            "history": [item for item in history
                        if int(item["step"]) <= int(row["step"])],
            "reached_common_horizon": (
                abs(float(row["time_s"])-float(common_time_s))
                / max(abs(float(common_time_s)), 1e-300) <= 1e-10
                if common_time_s is not None else
                int(row["step"]) == common_step
                if common_step is not None else result is not None)}


def observables(case):
    final = case["row"]
    return {
        "final_shear_stress_Pa": float(final["shear_stress_Pa"]),
        "temperature_mean_K": float(final["temperature_mean_K"]),
        "temperature_contrast_K": float(final["temperature_contrast_K"]),
        "favored_grain_volume_change_m3": float(
            final["grain_volume_change_m3"][1]),
    }


def subtract(left, right):
    return {key: left[key]-right[key] for key in left}


def refinement_error(reference, candidate):
    return {key: abs(candidate[key]-value)/max(abs(value), 1e-300)
            for key, value in reference.items()}


def persistent_candidate(history):
    peak = max(item["shear_stress_Pa"] for item in history)
    run = longest = 0; first = None
    for item in history:
        spatial = item.get("localization", {})
        plastic = spatial.get("plastic_power", {})
        heat = spatial.get("irreversible_heat_rate", {})
        candidate = bool(
            plastic.get("band_like", False)
            and heat.get("band_like", False)
            and plastic.get("maximum_to_mean", 0.0) >= 5.0
            and item["mechanical_energy"]["relative_first_law_residual"] <= .05)
        if candidate:
            first = int(item["step"]) if first is None else first
            run += 1; longest = max(longest, run)
        else:
            run = 0
    thermal = any(item["temperature_contrast_K"] >= 100.0
                  for item in history)
    softened = (first is not None and any(
        item["shear_stress_Pa"] < peak for item in history
        if int(item["step"]) >= first))
    return bool(longest >= 3 and thermal and softened), longest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--front-full", required=True)
    parser.add_argument("--front-frozen", required=True)
    parser.add_argument("--no-front-full", required=True)
    parser.add_argument("--no-front-frozen", required=True)
    parser.add_argument("--time-refined")
    parser.add_argument("--space-refined")
    parser.add_argument("--common-step", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    names = {
        "front_full": args.front_full,
        "front_frozen": args.front_frozen,
        "no_front_full": args.no_front_full,
        "no_front_frozen": args.no_front_frozen,
    }
    raw = {name: load(path, args.common_step)
           for name, path in names.items()}
    common_time_s = float(raw["front_full"]["row"]["time_s"])
    obs = {name: observables(value) for name, value in raw.items()}
    front_effect_full = subtract(obs["front_full"], obs["no_front_full"])
    front_effect_frozen = subtract(
        obs["front_frozen"], obs["no_front_frozen"])
    flow_feedback_front = subtract(
        obs["front_full"], obs["front_frozen"])
    flow_feedback_no_front = subtract(
        obs["no_front_full"], obs["no_front_frozen"])
    difference_of_differences = subtract(
        front_effect_full, front_effect_frozen)
    sources = {value["source"] for value in raw.values()}
    refinements = {}
    for label, path in (("time", args.time_refined),
                        ("space", args.space_refined)):
        if path:
            # A temporal or spatial refinement generally reaches the same
            # physical horizon at a different integer step.  Match it by the
            # authoritative baseline clock, not by an unrelated step number.
            candidate = load(path, common_time_s=common_time_s)
            candidate_obs = observables(candidate)
            errors = refinement_error(obs["front_full"], candidate_obs)
            refinements[label] = {
                "path": str(path), "observables": candidate_obs,
                "relative_error": errors,
                "passed_5_percent": max(errors.values()) <= .05,
            }
    baseline_class = raw["front_full"]["classification"]
    baseline_persistent, baseline_longest = persistent_candidate(
        raw["front_full"]["history"])
    controls_valid = bool(
        len(sources) == 1
        and all(value["reached_common_horizon"] for value in raw.values())
        and all(value["classification"]["owner_nye_consistent"]
                for value in raw.values()))
    all_case_local_dissipation = all(
        value["classification"]["local_dissipation_nonnegative"]
        for value in raw.values())
    refinement_passed = bool(
        {"time", "space"}.issubset(refinements)
        and all(value["passed_5_percent"] for value in refinements.values()))
    strict = bool(
        controls_valid and refinement_passed
        and baseline_persistent
        and baseline_class["substantial_existing_boundary_drx"]
        and baseline_class["local_dissipation_nonnegative"]
        and baseline_class["owner_nye_consistent"])
    report = {
        "schema": "asb-drx-v58-response-family-v1",
        "common_step": args.common_step,
        "common_time_s": common_time_s,
        "cases": names, "source_commits": sorted(sources),
        "observables": obs,
        "front_effect_at_full_feedback": front_effect_full,
        "front_effect_at_frozen_flow": front_effect_frozen,
        "flow_temperature_effect_with_front": flow_feedback_front,
        "flow_temperature_effect_without_front": flow_feedback_no_front,
        "front_flow_difference_of_differences": difference_of_differences,
        "matched_controls_valid": controls_valid,
        "all_case_local_dissipation_nonnegative": all_case_local_dissipation,
        "refinements": refinements,
        "selected_refinement_passed": refinement_passed,
        "substantial_existing_boundary_drx": baseline_class[
            "substantial_existing_boundary_drx"],
        "persistent_asb_trajectory_candidate": baseline_persistent,
        "maximum_consecutive_asb_candidate_steps": baseline_longest,
        "strict_coupled_response_qualified": strict,
        "spontaneous_grain_birth": False,
    }
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
