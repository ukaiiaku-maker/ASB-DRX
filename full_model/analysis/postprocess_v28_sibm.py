#!/usr/bin/env python3
"""Classify projection-free SIBM symmetry and first-passage coupling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def by_case(record):
    return {row["case"].split("-", 1)[-1]: row for row in record["records"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--near-equal", type=Path, required=True)
    parser.add_argument("--phase-controls", type=Path, required=True)
    parser.add_argument("--full-controls", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    near = json.loads(args.near_equal.read_text())
    phase = json.loads(args.phase_controls.read_text())
    full = json.loads(args.full_controls.read_text())
    registry = json.loads(args.registry.read_text())
    pcase, fcase = by_case(phase), by_case(full)
    odd = all(row["opposite_pressure_signs"] and row["opposite_velocity_signs"]
              and row["pressure_odd_relative_residual"] <= .05
              and row["velocity_odd_relative_residual"] <= .05
              for row in near["antisymmetric_pairs"])
    phase_signs = bool(
        pcase["favorable"]["response_sign"] > 0
        and pcase["reversed"]["response_sign"] < 0
        and pcase["label_swapped"]["response_sign"] < 0
        and pcase["mobility_off"]["response_sign"] == 0)
    full_signs = bool(
        fcase["favorable"]["response_sign"] > 0
        and fcase["reversed"]["response_sign"] < 0
        and fcase["label_swapped"]["response_sign"] < 0
        and fcase["mobility_off"]["response_sign"] == 0)
    projection_free = all(row["equal_state_projection_activations"] == 0
                          for row in near["records"])
    phase_equal_stationary = abs(float(
        pcase["equal"]["displacement_interface_widths"])) <= 1e-3
    full_equal_stationary = bool(
        abs(float(fcase["equal"]["displacement_interface_widths"])) <= 1e-3
        and fcase["equal"]["response_sign"] == 0
        and fcase["equal"]["front_ledger"]["swept_volume_m3"] == 0.0)
    phase_qualified = bool(
        projection_free and odd and phase_signs and phase_equal_stationary)
    if phase_qualified and not full_equal_stationary:
        classification = "SIBM_FRONT_FIRST_PASSAGE_ASYMMETRY"
        first_biased = "contour_first_passage_and_irreversible_front_processing"
    elif not phase_qualified:
        classification = "SIBM_PHASE_RESIDUAL_EXCHANGE_SYMMETRY_FAILURE"
        first_biased = "phase_operator_or_discrete_initial_state"
    elif not full_signs:
        classification = "SIBM_COMPLETE_COUPLING_DIRECTIONALITY_FAILURE"
        first_biased = "complete_coupling"
    else:
        classification = "PROJECTION_FREE_DISCRETE_SIBM_SYMMETRY_QUALIFIED"
        first_biased = None
    result = {
        "schema": "asb-drx/v28-projection-free-sibm-decision/v1",
        "grid": near["grid"], "steps": near["steps"],
        "projection_free": projection_free,
        "near_equal_odd_response": odd,
        "phase_equal_stationary": phase_equal_stationary,
        "phase_label_and_contrast_signs_passed": phase_signs,
        "complete_label_and_contrast_signs_passed": full_signs,
        "complete_equal_state_stationary": full_equal_stationary,
        "first_biased_suboperator": first_biased,
        "registry": registry,
        "phase_equal_record": pcase["equal"],
        "complete_equal_record": fcase["equal"],
        "classification": classification,
        "fixture_passed": True,
        "scientific_gate_passed": bool(
            classification == "PROJECTION_FREE_DISCRETE_SIBM_SYMMETRY_QUALIFIED"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(classification)


if __name__ == "__main__":
    main()
