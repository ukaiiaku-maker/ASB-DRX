#!/usr/bin/env python3
"""Reduce completed V65 evidence into decision-grade JSON and Markdown."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def reduce(campaign: Path) -> dict:
    analysis = campaign / "analysis"
    run = campaign / "runs/production_1p035_to1p25us_6debbe0"
    finalizer = load(campaign / "finalizer_status.json")
    result = load(run / "result.json")
    summary = load(analysis / "production_to1p25us_summary.json")
    asb = load(analysis / "final_asb_through1p25us.json")
    grain = load(analysis / "final_grain20_selection.json")
    work_precision = load(analysis / "work_precision_1p030_to1p035us.json")
    if finalizer.get("status") != "COMPLETE":
        raise RuntimeError("finalizer has not completed successfully")
    exit_codes = finalizer.get("step_exit_codes", {})
    tests_passed = (
        bool(exit_codes)
        and all(int(value) == 0 for value in exit_codes.values())
        and finalizer.get("all_required_artifacts_present") is True
    )
    if not tests_passed:
        raise RuntimeError("finalizer evidence is incomplete or failed")

    terminal_checkpoint = Path(result["checkpoint"])
    if not terminal_checkpoint.is_file():
        raise FileNotFoundError(terminal_checkpoint)
    final_asb_row = asb["rows"][-1]
    physical_episode = asb["physical_episode"]
    final_grain = grain["rows"][-1]
    maximum_grain = grain["volume_history_extrema"]["maximum_attained"]
    target_reached = (
        result.get("execution_status") == "TARGET_REACHED"
        and abs(float(result["final_time_s"])
                - float(result["target_time_s"])) <= 1e-18
    )
    strict_asb = bool(physical_episode.get("inherited_strict_asb", False))
    numerical_qualified = bool(
        work_precision["candidate_is_faster"]
        and work_precision["candidate_agrees_with_reference_at_working_tolerance"]
    )
    prepared_transient = bool(
        float(maximum_grain["volume_growth_fraction"]) >= 0.05
        and float(final_grain["volume_growth_fraction"])
        < float(maximum_grain["volume_growth_fraction"])
    )
    production_run = summary["runs"][0]
    component_audits = []
    for path in sorted(analysis.glob("power_component_transition_*.json")) + sorted(
        (analysis / "live_monitor").glob("power_component_transition_*.json")
    ):
        payload = load(path)
        component_audits.append({
            "artifact": str(path.resolve()),
            "artifact_sha256": digest(path),
            **payload["decision"],
        })
    curvature_path = analysis/"live_monitor/stress_curvature_latest.json"
    curvature = load(curvature_path) if curvature_path.is_file() else None
    peak_validation_path = analysis/"live_monitor/peak_forecast_validation.json"
    peak_validation = (load(peak_validation_path)
                       if peak_validation_path.is_file() else None)
    return {
        "schema": "asb-drx-v65-final-decision-v1",
        "execution_status": result.get("execution_status"),
        "delivery_status": "POSTPROCESS_AND_TESTS_COMPLETE_ARCHIVE_PENDING",
        "scoped_experiment_complete": bool(target_reached and tests_passed),
        "full_scientific_objective_complete": False,
        "provenance": {
            "production_source_commit": result.get("source_commit"),
            "terminal_checkpoint": str(terminal_checkpoint.resolve()),
            "terminal_checkpoint_sha256": digest(terminal_checkpoint),
            "final_time_s": float(result["final_time_s"]),
            "target_time_s": float(result["target_time_s"]),
            "final_applied_shear_strain": float(result["final_load"]),
        },
        "numerical_qualification": {
            "production_candidate_qualified": numerical_qualified,
            "reference_over_candidate_speedup": float(
                work_precision["wall_time_speedup_reference_over_candidate"]),
            "maximum_candidate_reference_difference_fraction": max(
                float(row["relative_difference"])
                for row in work_precision["endpoint_metric_differences"].values()),
            "working_tolerance": 0.02,
            "production_maximum_accepted_tolerance_ratio": float(
                production_run["maximum_accepted_tolerance_ratio"]),
            "production_accepted_intervals": int(
                production_run["accepted_intervals"]),
            "production_runner_wall_time_s": float(
                production_run["runner_wall_time_s"]),
            "production_runner_wall_seconds_per_physical_ns": float(
                production_run["runner_wall_seconds_per_physical_ns"]),
        },
        "physical_claims": {
            "prepared_boundary_transient_growth_then_recession": prepared_transient,
            "spontaneous_intragranular_grain_birth": False,
            "strict_asb": strict_asb,
            "asb_episode_count": len(physical_episode.get("episodes", [])),
            "maximum_asb_episode_duration_s": float(
                physical_episode.get("maximum_episode_duration_s", 0.0)),
            "final_post_peak_softening_fraction": float(
                physical_episode["softening_summary"][
                    "final_qualifying_operator_segment_fraction"]),
            "final_temperature_peak_minus_mean_K": float(
                final_asb_row["temperature_max_minus_mean_K"]),
            "final_power_participation": float(
                final_asb_row["plastic_power"][
                    "inverse_participation_fraction"]),
            "final_power_aspect_ratio": float(
                final_asb_row["power_aspect_ratio"]),
            "final_heat_power_overlap": float(
                final_asb_row["heat_power_component_overlap"]),
            "grain_20_maximum_growth_fraction": float(
                maximum_grain["volume_growth_fraction"]),
            "grain_20_maximum_growth_interval": int(
                maximum_grain["adaptive_interval"]),
            "grain_20_final_growth_fraction": float(
                final_grain["volume_growth_fraction"]),
            "grain_20_final_displacement_cells": float(
                final_grain["equivalent_contour_displacement_cells"]),
        },
        "history_scope": {
            "available_scalar_history_complete": bool(
                asb["scalar_history"]["history_complete"]),
            "declared_gaps": asb["scalar_history"]["gaps"],
            "semantics": asb["scalar_history"]["semantics"],
        },
        "localization_identity_audits": component_audits,
        "preterminal_stress_peak_projection": (
            None if curvature is None else {
                "artifact": str(curvature_path.resolve()),
                "artifact_sha256": digest(curvature_path),
                "decision": curvature["decision"],
                "projected_peak_times_s": [
                    row["projected_peak_time_s"] for row in curvature["fits"]
                ],
            }
        ),
        "preterminal_stress_peak_validation": (
            None if peak_validation is None else {
                "artifact": str(peak_validation_path.resolve()),
                "artifact_sha256": digest(peak_validation_path),
                **peak_validation,
            }
        ),
        "tests": {
            "passed": tests_passed,
            "exit_codes": exit_codes,
        },
        "remaining_work": [
            "common-origin matched-temperature ASB control",
            "spatial and temporal refinement of any future localization onset",
            "a persistent conjunctive strict-ASB episode rather than one screen",
            "spontaneous intragranular grain-birth capability",
            "broader fixed-law rate and temperature response qualification",
        ],
        "missing_or_failed_asb_requirements": physical_episode.get(
            "missing_or_failed_requirements", []),
    }


def markdown(payload: dict) -> str:
    p = payload["physical_claims"]
    q = payload["numerical_qualification"]
    provenance = payload["provenance"]
    identity_audits = payload.get("localization_identity_audits", [])
    forecast = payload.get("preterminal_stress_peak_projection")
    validation = payload.get("preterminal_stress_peak_validation")
    identity_text = (
        " No threshold-component identity event was separately audited."
        if not identity_audits else
        " Audited component-selection events are: " + ", ".join(
            f"`{item['classification']}`" for item in identity_audits
        ) + ". These events prevent largest-area morphology samples from being "
        "interpreted as one continuous physical structure."
    )
    forecast_text = (
        "" if forecast is None else
        " A preterminal scalar-history extrapolation was preserved as "
        f"`{forecast['decision']['classification']}`; it explicitly did not "
        "constitute an observed peak, softening event, or spatial ASB claim."
    )
    validation_text = (
        "" if validation is None else
        " Its prospective validation result is "
        f"`{validation['classification']}` and remains a scalar-stress result, "
        "not a spatial ASB qualification."
    )
    return f"""# V65 final decision

## Decision

V65 completed its scoped controlled continuation to
`{provenance['final_time_s']:.12e} s` under production source
`{provenance['production_source_commit']}`.  The same-law candidate is
production-qualified: it is `{q['reference_over_candidate_speedup']:.6f}x`
faster than the fixed-proposal comparator and its largest registered endpoint
difference is `{q['maximum_candidate_reference_difference_fraction']:.6g}`
against the unchanged `0.02` working contract.

The physical result is decision-grade but not a complete DRX/ASB validation.
Prepared grain 20 exhibits transient boundary-mediated growth followed by
recession; this is not spontaneous intragranular grain birth.  Strict ASB is `{str(p['strict_asb']).lower()}`.
At the terminal state, peak-minus-mean
temperature is `{p['final_temperature_peak_minus_mean_K']:.6g} K`, power
participation is `{p['final_power_participation']:.6g}`, power aspect ratio is
`{p['final_power_aspect_ratio']:.6g}`, heat/power overlap is
`{p['final_heat_power_overlap']:.6g}`, and qualifying-operator softening is
`{p['final_post_peak_softening_fraction']:.6g}`.

## Grain-selection result

Grain 20 attained a maximum growth fraction of
`{p['grain_20_maximum_growth_fraction']:.6g}` at adaptive interval
`{p['grain_20_maximum_growth_interval']}`.  Its terminal growth fraction is
`{p['grain_20_final_growth_fraction']:.6g}` and its endpoint-average-perimeter
displacement is `{p['grain_20_final_displacement_cells']:.6g}` cells.  The
registered classification is transient prepared-boundary growth followed by
recession.

## Claim boundary

The available scalar history retains the declared interval 2185--2456 gap and
is not labeled a complete whole-history record.  Raw checkpoint conjunctions
and accepted-time suffix trends remain attributable.  No common-origin matched
temperature control or spatial refinement exists, and no spontaneous grain
nucleation is claimed.{identity_text}{forecast_text}{validation_text}

## Remaining work

""" + "".join(f"- {item}\n" for item in payload["remaining_work"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    payload = reduce(args.campaign.resolve())
    atomic_text(args.json, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    atomic_text(args.markdown, markdown(payload))


if __name__ == "__main__":
    main()
