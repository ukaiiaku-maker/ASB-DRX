#!/usr/bin/env python3
"""Classify the matched V58 front/flow-temperature response family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(root):
    root = Path(root)
    result = json.loads((root/"result.json").read_text())
    classification = json.loads((root/"classification.json").read_text())
    return result, classification


def observables(result, classification):
    final = result["final"]
    return {
        "final_shear_stress_Pa": float(final["shear_stress_Pa"]),
        "temperature_mean_K": float(final["temperature_mean_K"]),
        "temperature_contrast_K": float(final["temperature_contrast_K"]),
        "favored_grain_volume_change_m3": float(
            final["grain_volume_change_m3"][1]),
        "gross_transformed_volume_m3": float(
            classification["gross_transformed_volume_m3"]),
    }


def subtract(left, right):
    return {key: left[key]-right[key] for key in left}


def refinement_error(reference, candidate):
    return {key: abs(candidate[key]-value)/max(abs(value), 1e-300)
            for key, value in reference.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--front-full", required=True)
    parser.add_argument("--front-frozen", required=True)
    parser.add_argument("--no-front-full", required=True)
    parser.add_argument("--no-front-frozen", required=True)
    parser.add_argument("--time-refined")
    parser.add_argument("--space-refined")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    names = {
        "front_full": args.front_full,
        "front_frozen": args.front_frozen,
        "no_front_full": args.no_front_full,
        "no_front_frozen": args.no_front_frozen,
    }
    raw = {name: load(path) for name, path in names.items()}
    obs = {name: observables(*value) for name, value in raw.items()}
    front_effect_full = subtract(obs["front_full"], obs["no_front_full"])
    front_effect_frozen = subtract(
        obs["front_frozen"], obs["no_front_frozen"])
    flow_feedback_front = subtract(
        obs["front_full"], obs["front_frozen"])
    flow_feedback_no_front = subtract(
        obs["no_front_full"], obs["no_front_frozen"])
    difference_of_differences = subtract(
        front_effect_full, front_effect_frozen)
    sources = {value[0]["source_commit"] for value in raw.values()}
    refinements = {}
    for label, path in (("time", args.time_refined),
                        ("space", args.space_refined)):
        if path:
            candidate = load(path)
            candidate_obs = observables(*candidate)
            errors = refinement_error(obs["front_full"], candidate_obs)
            refinements[label] = {
                "path": str(path), "observables": candidate_obs,
                "relative_error": errors,
                "passed_5_percent": max(errors.values()) <= .05,
            }
    baseline_class = raw["front_full"][1]
    controls_valid = bool(
        len(sources) == 1
        and all(value[1]["trajectory_complete"] for value in raw.values())
        and all(value[1]["owner_nye_consistent"] for value in raw.values()))
    all_case_local_dissipation = all(
        value[1]["local_dissipation_nonnegative"] for value in raw.values())
    refinement_passed = bool(
        {"time", "space"}.issubset(refinements)
        and all(value["passed_5_percent"] for value in refinements.values()))
    strict = bool(
        controls_valid and refinement_passed
        and baseline_class["persistent_asb_trajectory_candidate"]
        and baseline_class["substantial_existing_boundary_drx"]
        and baseline_class["local_dissipation_nonnegative"]
        and baseline_class["owner_nye_consistent"])
    report = {
        "schema": "asb-drx-v58-response-family-v1",
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
        "persistent_asb_trajectory_candidate": baseline_class[
            "persistent_asb_trajectory_candidate"],
        "strict_coupled_response_qualified": strict,
        "spontaneous_grain_birth": False,
    }
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
