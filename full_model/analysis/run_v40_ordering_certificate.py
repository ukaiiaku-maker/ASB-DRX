#!/usr/bin/env python3
"""Finite-time overlap certificate for the V40 ordering dispatch."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np

from full_model.analysis.run_v24_mechanical_supply import build_case
from full_model.production.extensive_wall import (
    _attempt_rate_s, accepted_ordering_step,
)


FIELDS = (
    "wall_tangle_plus_m2", "wall_tangle_minus_m2",
    "wall_ordered_plus_m2", "wall_ordered_minus_m2",
)


def build_variant(name: str):
    state, _, _, _, systems, topologies, _, parameters, _, _ = build_case(16)
    shape = state.density.wall_tangle_plus_m2.shape
    plus = np.zeros(shape); minus = np.zeros(shape)
    scale = 1.0 if name != "near_empty_mixed" else 1e-9
    plus[7, 7, :] = scale * 1.2e13
    if name != "fully_polarized_plus":
        minus[7, 7, :] = scale * 9.0e12
    density = replace(
        state.density,
        wall_tangle_plus_m2=plus, wall_tangle_minus_m2=minus,
        wall_ordered_plus_m2=np.zeros(shape),
        wall_ordered_minus_m2=np.zeros(shape))
    target = np.zeros(state.common.orientation_rad.shape + (3, 3))
    stress = np.full(state.common.slip.shape, 7e8)
    attempt = float(np.max(_attempt_rate_s(
        np.max(np.abs(stress), axis=2), state.common.temperature_K,
        parameters)))
    return state, density, systems, topologies, parameters, target, stress, attempt


def run(case, method, exposure, rate_multiplier=1.0):
    state, density, systems, topologies, parameters, target, stress, attempt = case
    parameters = replace(
        parameters, ordering_integration_method=method,
        ordering_attempt_frequency_s=(
            rate_multiplier * parameters.ordering_attempt_frequency_s))
    duration = exposure / (attempt * rate_multiplier)
    updated, ledger, _ = accepted_ordering_step(
        density, systems, topologies, state.common.orientation_rad, target,
        stress, state.common.temperature_K, parameters, duration)
    return updated, ledger, duration


def compare(left, right, initial):
    capacity = max(sum(float(np.sum(getattr(
        initial, f"wall_tangle_{sign}_m2") + getattr(
            initial, f"wall_ordered_{sign}_m2"), dtype=np.longdouble))
                       for sign in ("plus", "minus")), 1e-300)
    absolute = max(float(np.max(np.abs(
        getattr(left, name) - getattr(right, name)))) for name in FIELDS)
    lordered = sum(float(np.sum(getattr(
        left, f"wall_ordered_{sign}_m2"), dtype=np.longdouble))
                   for sign in ("plus", "minus"))
    rordered = sum(float(np.sum(getattr(
        right, f"wall_ordered_{sign}_m2"), dtype=np.longdouble))
                   for sign in ("plus", "minus"))
    return {
        "maximum_absolute_density_difference_m2": absolute,
        "maximum_difference_over_total_capacity": absolute / capacity,
        "ordered_inventory_left_m2_cells": lordered,
        "ordered_inventory_right_m2_cells": rordered,
        "ordered_inventory_relative_difference": abs(lordered-rordered) /
        max(abs(rordered), 1e-300),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    variants = {}
    for name in ("mixed_sign", "fully_polarized_plus", "near_empty_mixed"):
        case = build_variant(name)
        finite, finite_ledger, duration = run(case, "finite_time_bdf", 50.0)
        selected, selected_ledger, _ = run(
            case, "implicit_backward_euler", 50.0)
        variants[name] = {
            "attempt_exposure": 50.0, "duration_s": duration,
            "finite_method": finite_ledger["integration_method"],
            "selected_method": selected_ledger["integration_method"],
            "selected_endpoint_diagnostic_semantics": selected_ledger[
                "asymptotic_endpoint_diagnostic_semantics"],
            "comparison": compare(selected, finite, case[1]),
        }
    case = build_variant("mixed_sign")
    baseline, baseline_ledger, duration = run(case, "finite_time_bdf", 20.0)
    scaled, scaled_ledger, scaled_duration = run(
        case, "finite_time_bdf", 20.0, rate_multiplier=4.0)
    high_finite, high_finite_ledger, high_duration = run(
        case, "finite_time_bdf", 100.0)
    high_selected, high_selected_ledger, _ = run(
        case, "implicit_backward_euler", 100.0)
    switch_pass = all(
        row["comparison"]["maximum_difference_over_total_capacity"] <= 2e-6
        and row["comparison"]["ordered_inventory_relative_difference"] <= 2e-6
        for row in variants.values())
    result = {
        "schema": "asb-drx/v40/ordering-finite-time-certificate/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "endpoint_diagnostic_interpretation": (
            "H times endpoint rate divided by a local inventory floor is a "
            "stationary endpoint drift diagnostic, not by itself a bound on "
            "finite-time kinetic error"),
        "finite_time_fallback": "bounded finite-time scipy BDF integration",
        "switch_exposure": 50.0,
        "switch_variants": variants,
        "rate_time_rescaling": {
            "rate_multiplier": 4.0,
            "baseline_duration_s": duration,
            "scaled_duration_s": scaled_duration,
            "baseline_method": baseline_ledger["integration_method"],
            "scaled_method": scaled_ledger["integration_method"],
            "comparison": compare(scaled, baseline, case[1]),
        },
        "high_exposure_overlap": {
            "attempt_exposure": 100.0, "duration_s": high_duration,
            "finite_method": high_finite_ledger["integration_method"],
            "selected_method": high_selected_ledger["integration_method"],
            "comparison": compare(high_selected, high_finite, case[1]),
        },
        "switch_overlap_passed": bool(switch_pass),
        "rate_time_rescaling_passed": bool(compare(
            scaled, baseline, case[1])[
                "maximum_difference_over_total_capacity"] <= 2e-7),
        "high_exposure_overlap_passed": bool(compare(
            high_selected, high_finite, case[1])[
                "maximum_difference_over_total_capacity"] <= 2e-6),
        "claim_boundary": (
            "fixed-field ordering-channel certificate; coupled coefficient "
            "freezing remains controlled by common macro refinement"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
