#!/usr/bin/env python3
"""Compare copied-state macro-step refinements at one physical endpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _checkpoint(path):
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]).copy() for name in data.files
                  if name not in {"metadata_json", "runtime_json"}}
        runtime = json.loads(str(data["runtime_json"]))
    return arrays, runtime


def _common(arrays, name, trailing):
    supports = arrays["supports"]
    return sum(
        supports[index][(...,)+(None,)*trailing]
        *arrays[f"owner__{index}__{name}"]
        for index in range(supports.shape[0]))


def _signed_density(arrays):
    supports = arrays["supports"]
    result = 0.0
    for index in range(supports.shape[0]):
        signed = sum(
            arrays[f"owner__{index}__{plus}"]
            -arrays[f"owner__{index}__{minus}"]
            for plus, minus in (
                ("mobile_plus_m2", "mobile_minus_m2"),
                ("forest_plus_m2", "forest_minus_m2"),
                ("wall_plus_m2", "wall_minus_m2")))
        result = result+supports[index][..., None]*signed
    return result


def _error(candidate, reference, parent):
    difference = np.asarray(candidate)-np.asarray(reference)
    reference = np.asarray(reference)
    increment = reference-np.asarray(parent)
    norm = float(np.linalg.norm(difference.ravel()))
    return {
        "maximum_absolute": float(np.max(np.abs(difference))),
        "rms_absolute": float(np.sqrt(np.mean(difference*difference))),
        "signal_relative_l2": norm/max(
            float(np.linalg.norm(reference.ravel())), 1e-300),
        "increment_relative_l2": norm/max(
            float(np.linalg.norm(increment.ravel())), 1e-300),
    }


def _last_history(path):
    values = json.loads(Path(path).read_text())
    if not values:
        raise ValueError("history is empty")
    return values[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate-history", required=True)
    parser.add_argument("--reference-history", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--reference-label", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    parent, parent_runtime = _checkpoint(args.parent)
    candidate, candidate_runtime = _checkpoint(args.candidate)
    reference, reference_runtime = _checkpoint(args.reference)
    fields = {
        "supports": (candidate["supports"], reference["supports"],
                     parent["supports"]),
        "common_temperature_K": (
            _common(candidate, "temperature_K", 0),
            _common(reference, "temperature_K", 0),
            _common(parent, "temperature_K", 0)),
        "common_signed_slip": (
            _common(candidate, "slip", 1),
            _common(reference, "slip", 1),
            _common(parent, "slip", 1)),
        "common_plastic_distortion": (
            _common(candidate, "beta_p", 2),
            _common(reference, "beta_p", 2),
            _common(parent, "beta_p", 2)),
        "common_signed_density_m2": (
            _signed_density(candidate), _signed_density(reference),
            _signed_density(parent)),
        "common_family_nye_m1": (
            _common(candidate, "family_nye_m1", 3),
            _common(reference, "family_nye_m1", 3),
            _common(parent, "family_nye_m1", 3)),
    }
    field_errors = {
        name: _error(value[0], value[1], value[2])
        for name, value in fields.items()}
    candidate_history = _last_history(args.candidate_history)
    reference_history = _last_history(args.reference_history)
    endpoint_scalars = {}
    for name in (
            "shear_stress_Pa", "helmholtz_J", "thermal_internal_J",
            "temperature_mean_K", "temperature_contrast_K"):
        first = float(candidate_history[name])
        second = float(reference_history[name])
        endpoint_scalars[name] = {
            args.candidate_label: first, args.reference_label: second,
            "absolute_error": abs(first-second),
            "signal_relative_error": abs(first-second)/max(abs(second), 1e-300),
        }
    ledger_increments = {}
    parent_ledger = parent_runtime["ledger"]
    for name in (
            "transformed_volume_m3", "generated_heat_J",
            "cumulative_mechanical_external_work_J",
            "cumulative_mechanical_internal_energy_change_J",
            "cumulative_mechanical_first_law_residual_J",
            "cumulative_front_first_law_residual_J"):
        first = float(candidate_runtime["ledger"][name]-parent_ledger[name])
        second = float(reference_runtime["ledger"][name]-parent_ledger[name])
        ledger_increments[name] = {
            args.candidate_label: first, args.reference_label: second,
            "absolute_error": abs(first-second),
            "reference_relative_error": abs(first-second)/max(abs(second), 1e-300),
        }
    localization = {}
    for channel in ("plastic_power", "irreversible_heat_rate"):
        localization[channel] = {}
        for metric in (
                "maximum_to_mean", "largest_component_fraction",
                "component_span_fraction", "band_like"):
            localization[channel][metric] = {
                args.candidate_label:
                    candidate_history["localization"][channel][metric],
                args.reference_label:
                    reference_history["localization"][channel][metric],
            }
    exact_clock = {
        "physical_time_absolute_difference_s": abs(
            candidate_runtime["ledger"]["physical_time_s"]
            -reference_runtime["ledger"]["physical_time_s"]),
        "applied_shear_absolute_difference": abs(
            candidate_runtime["ledger"]["applied_shear_strain"]
            -reference_runtime["ledger"]["applied_shear_strain"]),
    }
    result = {
        "schema": "asb-drx-v62-copied-state-temporal-refinement-v1",
        "parent": str(Path(args.parent).resolve()),
        "candidate": str(Path(args.candidate).resolve()),
        "reference": str(Path(args.reference).resolve()),
        "candidate_label": args.candidate_label,
        "reference_label": args.reference_label,
        "clock_and_load": exact_clock,
        "field_errors": field_errors,
        "endpoint_scalars": endpoint_scalars,
        "ledger_increment_errors": ledger_increments,
        "endpoint_localization": localization,
        "scope": (
            "one copied-state physical interval; this does not certify the "
            "accumulated trajectory or spatial convergence"),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
