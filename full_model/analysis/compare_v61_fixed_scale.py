"""Create a source-bound fixed-physical-scale V61 refinement certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from full_model.analysis.postprocess_v59_physical_asb import analyze
from full_model.analysis.run_v58_three_grain_production import _load_checkpoint


TOLERANCE = 0.05


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(left, right, floor=1e-300):
    return abs(float(left)-float(right))/max(
        abs(float(left)), abs(float(right)), float(floor))


def _row_by_time(rows):
    return {round(float(row["physical_time_s"]), 18): row for row in rows}


def _physical_configuration(configuration):
    excluded = {
        "n", "flow_temperature_mode", "recovery_temperature_mode",
        "front_temperature_mode",
    }
    return {key: value for key, value in configuration.items()
            if key not in excluded}


def _state_metrics(checkpoint: Path) -> dict:
    state, runtime, step, _gamma, initial_volume, configuration, provenance = (
        _load_checkpoint(checkpoint))
    spacing = float(configuration["length_m"])/int(configuration["n"])
    thickness = 2.0*2.48e-10
    domain_volume = float(configuration["length_m"])**2*thickness
    final_volume = np.sum(state.supports, axis=(1, 2))*spacing**2*thickness
    return {
        "step": step,
        "physical_time_s": runtime.ledger.physical_time_s,
        "source_commit": (provenance or {}).get("source_commit"),
        "configuration": configuration,
        "fresh_sweep_fraction": float(
            state.ledger.fresh_sweep_fraction*spacing**2*thickness
            /domain_volume),
        "revisit_sweep_fraction": float(
            state.ledger.revisit_sweep_fraction*spacing**2*thickness
            /domain_volume),
        "net_grain_volume_fraction_change": (
            (final_volume-initial_volume)/domain_volume).tolist(),
    }


def compare(coarse_physical: Path, coarse_control: Path,
            fine_physical: Path, fine_control: Path) -> dict:
    coarse = analyze([coarse_physical], [coarse_control])
    fine = analyze([fine_physical], [fine_control])
    coarse_rows = _row_by_time(coarse["rows"])
    fine_rows = _row_by_time(fine["rows"])
    common_times = sorted(set(coarse_rows) & set(fine_rows))
    if not common_times:
        raise ValueError("refinement pair has no common saved physical time")
    common_time = common_times[-1]
    left = coarse_rows[common_time]; right = fine_rows[common_time]
    coarse_checkpoint = Path(left["checkpoint"])
    fine_checkpoint = Path(right["checkpoint"])
    coarse_state = _state_metrics(coarse_checkpoint)
    fine_state = _state_metrics(fine_checkpoint)
    physical_scales = (
        _physical_configuration(coarse_state["configuration"])
        == _physical_configuration(fine_state["configuration"])
        and coarse_state["configuration"]["n"]
        < fine_state["configuration"]["n"])
    same_source = bool(
        coarse_state["source_commit"]
        and coarse_state["source_commit"] == fine_state["source_commit"]
        and coarse["temperature_intervention_certificate"]["passed"]
        and fine["temperature_intervention_certificate"]["passed"])
    observables = {
        "post_front_stress": _relative(
            left["post_front_equilibrated_stress_Pa"],
            right["post_front_equilibrated_stress_Pa"]),
        "temperature_peak_minus_mean": _relative(
            left["temperature_max_minus_mean_K"],
            right["temperature_max_minus_mean_K"], floor=1.0),
        "matched_causal_temperature_excess": _relative(
            left.get("matched_temperature_excess_K") or 0.0,
            right.get("matched_temperature_excess_K") or 0.0, floor=1.0),
        "power_participation": _relative(
            left["plastic_power"]["inverse_participation_fraction"],
            right["plastic_power"]["inverse_participation_fraction"],
            floor=1.0/int(coarse_state["configuration"]["n"])**2),
        "fresh_sweep_fraction": _relative(
            coarse_state["fresh_sweep_fraction"],
            fine_state["fresh_sweep_fraction"], floor=1e-12),
    }
    widths_available = bool(
        left["power_width_minor_m"] is not None
        and right["power_width_minor_m"] is not None)
    if widths_available:
        observables["power_width_minor"] = _relative(
            left["power_width_minor_m"], right["power_width_minor_m"],
            floor=float(coarse_state["configuration"]["length_m"])
            /int(fine_state["configuration"]["n"]))
    localized = bool(
        max(left["temperature_max_minus_mean_K"],
            right["temperature_max_minus_mean_K"]) >= 50.0)
    maximum_error = max(observables.values())
    endpoint_passed = bool(
        maximum_error <= TOLERANCE and (not localized or widths_available))

    def onset(rows):
        candidates = [row for row in rows
                      if row["temperature_max_minus_mean_K"] >= 50.0]
        return None if not candidates else float(candidates[0]["physical_time_s"])

    coarse_onset = onset(coarse["rows"]); fine_onset = onset(fine["rows"])
    saved_cadences = [
        float(item["declared_sampling_cadence_s"])
        for item in (*coarse["sampling_segments"].values(),
                     *fine["sampling_segments"].values())
        if item.get("declared_sampling_cadence_s") is not None]
    onset_agreement = bool(
        coarse_onset is None and fine_onset is None
        or coarse_onset is not None and fine_onset is not None
        and abs(coarse_onset-fine_onset) <= max(
            saved_cadences, default=0.0))
    record = {
        "common_physical_time_s": common_time,
        "coarse_grid": int(coarse_state["configuration"]["n"]),
        "fine_grid": int(fine_state["configuration"]["n"]),
        "coarse_checkpoint": str(coarse_checkpoint.resolve()),
        "fine_checkpoint": str(fine_checkpoint.resolve()),
        "coarse_checkpoint_sha256": _digest(coarse_checkpoint),
        "fine_checkpoint_sha256": _digest(fine_checkpoint),
        "source_commit": coarse_state["source_commit"],
        "relative_errors": observables,
        "maximum_selected_relative_error": maximum_error,
        "widths_available": widths_available,
        "localized_state_requires_width": localized,
        "coarse_temperature_onset_s": coarse_onset,
        "fine_temperature_onset_s": fine_onset,
        "onset_agreement_within_saved_cadence": onset_agreement,
        "coarse_physical_episode_duration_s": coarse[
            "physical_episode"]["maximum_episode_duration_s"],
        "fine_physical_episode_duration_s": fine[
            "physical_episode"]["maximum_episode_duration_s"],
    }
    passed = bool(
        physical_scales and same_source and endpoint_passed and onset_agreement)
    return {
        "schema": "asb-drx-v61-fixed-scale-refinement-certificate-v1",
        "passed": passed,
        "provisional_relative_tolerance": TOLERANCE,
        "fixed_physical_scales_verified": physical_scales,
        "same_source_commit_verified": same_source,
        "endpoint_observables_passed": endpoint_passed,
        "onset_agreement_passed": onset_agreement,
        "comparison_records": [record],
        "claim_limit": (
            "This certificate covers the listed source-bound common-time "
            "comparison only; it does not infer persistence beyond the actual "
            "paired horizon or qualify spontaneous grain birth."),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse-physical", required=True, type=Path)
    parser.add_argument("--coarse-control", required=True, type=Path)
    parser.add_argument("--fine-physical", required=True, type=Path)
    parser.add_argument("--fine-control", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.coarse_physical, args.coarse_control,
                     args.fine_physical, args.fine_control)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
