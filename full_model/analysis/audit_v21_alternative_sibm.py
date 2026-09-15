#!/usr/bin/env python3
"""Classify the V21 fallback existing-HAGB sign-discrimination pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_v14_pinned_criticality import classify_case, common_record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--subcritical", type=Path, required=True)
    parser.add_argument("--supercritical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    common = common_record(args.source)
    subcritical = classify_case(
        "subcritical", args.subcritical, common, expected_sign=-1)
    supercritical = classify_case(
        "supercritical", args.supercritical, common, expected_sign=1)
    passed = subcritical["passed"] and supercritical["passed"]
    result = {
        "schema": "asb-drx/v21-existing-hagb-fallback/v1",
        "classification": ("V21_EXISTING_HAGB_SIBM_SIGN_DISCRIMINATION_PASSED"
                           if passed else
                           "V21_EXISTING_HAGB_SIBM_SIGN_DISCRIMINATION_FAILED"),
        "purpose": (
            "Outcome-C/D fallback after the common intragranular operator "
            "selected only uniform order and no robust finite signed-wall mode"),
        "source": {key: value for key, value in common.items()
                   if key != "psi_gv"},
        "cases": [subcritical, supercritical],
        "claim_limit": (
            "short generic pinned-HAGB sign discrimination; not converged "
            "production SIBM kinetics or material calibration"),
        "hpc3_job_launched": False,
        "fixture_passed": bool(passed),
        "scientific_gate_passed": bool(passed),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({
        "classification": result["classification"],
        "amplitude_change_cells": {
            case["name"]: case["amplitude_change_cells"]
            for case in result["cases"]},
    }, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
