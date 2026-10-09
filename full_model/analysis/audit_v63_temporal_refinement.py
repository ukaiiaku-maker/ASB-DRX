#!/usr/bin/env python3
"""Fresh-state caller-step accuracy including state, rates, and energy clocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from full_model.analysis.audit_v63_phase_sensitive_fields import (
    checkpoint_fields, digest, norm,
)
from full_model.analysis.run_v58_three_grain_production import _load_checkpoint


INTERVAL_DIAGNOSTICS = {
    "accepted_interval_front_heat_J_by_cell",
    "accepted_interval_mechanical_heat_J_m3",
    "accepted_interval_mechanical_work_J_m3",
    "accepted_interval_conduction_J_m3",
}


def scaled_error(candidate, reference, parent) -> dict:
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    parent = np.asarray(parent, dtype=float)
    difference = norm(candidate-reference)
    signal = max(norm(candidate), norm(reference), 1e-300)
    candidate_increment = candidate-parent
    reference_increment = reference-parent
    increment_scale = max(norm(candidate_increment), norm(reference_increment),
                          1e-300)
    return {
        "absolute_l2": difference,
        "signal_relative_l2": difference/signal,
        "increment_relative_l2": difference/increment_scale,
        "candidate_increment_l2": norm(candidate_increment),
        "reference_increment_l2": norm(reference_increment),
        "increment_effectively_vanishing": bool(
            increment_scale < 1e-12*signal),
    }


def interval_history(path: Path) -> dict:
    rows = json.loads(Path(path).read_text())
    if not rows:
        raise ValueError("temporal-refinement history is empty")
    consumed = float(sum(
        row["mechanical"]["consumed_interval_s"] for row in rows))
    requested = float(rows[-1]["time_s"]-(
        rows[0]["time_s"]-rows[0]["mechanical"]["consumed_interval_s"]))
    return {
        "row_count": len(rows),
        "requested_physical_time_s": requested,
        "mechanical_consumed_physical_time_s": consumed,
        "mechanical_irreversible_heat_J": float(sum(
            row["mechanical"]["irreversible_heat_J"] for row in rows)),
        "mechanical_external_plastic_work_J": float(sum(
            row["mechanical"]["external_plastic_work_J"] for row in rows)),
        "complete_mechanical_external_work_J": float(sum(
            row["mechanical_energy"]["external_work_J"] for row in rows)),
        "complete_mechanical_internal_energy_change_J": float(sum(
            row["mechanical_energy"]["internal_energy_change_J"] for row in rows)),
        "front_generated_heat_J": float(sum(
            row["front"]["generated_heat_J"] for row in rows)),
        "front_internal_subintervals": int(sum(
            row["front"]["subintervals"] for row in rows)),
        "mechanical_internal_substeps": int(sum(
            row["mechanical"]["total_internal_substeps"] for row in rows)),
        "maximum_front_contour_fraction": float(max(
            row["front"]["maximum_unclipped_fraction"] for row in rows)),
        "all_front_contour_clocks_closed": bool(all(
            row["front"]["contour_cfl_satisfied"] for row in rows)),
        "all_mechanical_clocks_closed": bool(np.isclose(
            consumed, requested, rtol=0.0, atol=1e-18)),
    }


def ledger_increment(parent_path: Path, endpoint_path: Path) -> dict:
    _, parent_runtime, _, parent_gamma, _, _, _ = _load_checkpoint(parent_path)
    _, runtime, _, gamma, _, _, _ = _load_checkpoint(endpoint_path)
    fields = (
        "physical_time_s", "applied_shear_strain", "transformed_volume_m3",
        "generated_heat_J", "cumulative_mechanical_external_work_J",
        "cumulative_mechanical_internal_energy_change_J",
        "cumulative_mechanical_first_law_residual_J",
        "cumulative_front_first_law_residual_J", "intervals",
        "accepted_events", "rejected_events")
    values = {name: float(getattr(runtime.ledger, name)
                          -getattr(parent_runtime.ledger, name))
              for name in fields}
    values["gamma_checkpoint_increment"] = float(gamma-parent_gamma)
    return values


def run(parent: Path, candidate: Path, reference: Path,
        candidate_history: Path, reference_history: Path,
        output: Path) -> dict:
    parent_fields, parent_meta = checkpoint_fields(parent)
    candidate_fields, candidate_meta = checkpoint_fields(candidate)
    reference_fields, reference_meta = checkpoint_fields(reference)
    if not np.isclose(candidate_meta["physical_time_s"],
                      reference_meta["physical_time_s"], rtol=0.0, atol=1e-18):
        raise ValueError("candidate and reference physical times differ")
    if not np.isclose(candidate_meta["gamma"], reference_meta["gamma"],
                      rtol=0.0, atol=1e-15):
        raise ValueError("candidate and reference loads differ")
    names = sorted(set(parent_fields)&set(candidate_fields)&set(reference_fields)
                   -INTERVAL_DIAGNOSTICS)
    candidate_ledger = ledger_increment(parent, candidate)
    reference_ledger = ledger_increment(parent, reference)
    ledger_errors = {}
    for name in candidate_ledger:
        a, b = candidate_ledger[name], reference_ledger[name]
        ledger_errors[name] = {
            "candidate": a, "reference": b, "absolute_error": abs(a-b),
            "relative_error": abs(a-b)/max(abs(a), abs(b), 1e-300),
            "both_zero": bool(a == 0.0 and b == 0.0),
        }
    candidate_interval = interval_history(candidate_history)
    reference_interval = interval_history(reference_history)
    interval_errors = {}
    for name in candidate_interval:
        a, b = candidate_interval[name], reference_interval[name]
        interval_errors[name] = {"candidate": a, "reference": b}
        if isinstance(a, (int, float)) and not isinstance(a, bool):
            interval_errors[name].update({
                "absolute_error": abs(a-b),
                "relative_error": abs(a-b)/max(abs(a), abs(b), 1e-300),
            })
    result = {
        "schema": "asb-drx-v63-fresh-state-temporal-refinement-v1",
        "parent": parent_meta, "candidate": candidate_meta,
        "reference": reference_meta,
        "endpoint_clock": {
            "physical_time_absolute_difference_s": abs(
                candidate_meta["physical_time_s"]-reference_meta["physical_time_s"]),
            "gamma_absolute_difference": abs(
                candidate_meta["gamma"]-reference_meta["gamma"]),
        },
        "state_and_instantaneous_rate_errors": {
            name: scaled_error(candidate_fields[name], reference_fields[name],
                               parent_fields[name]) for name in names},
        "ledger_increment_errors": ledger_errors,
        "accepted_interval_integrals_and_cost": interval_errors,
        "scope": (
            "one copied fresh-history state interval; endpoint instantaneous "
            "rates are distinct from accepted-interval integrated work/heat"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate-history", type=Path, required=True)
    parser.add_argument("--reference-history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.parent, args.candidate, args.reference,
                 args.candidate_history, args.reference_history, args.output)
    errors = result["state_and_instantaneous_rate_errors"]
    print(json.dumps({
        "sha256": digest(args.output),
        "temperature_increment_error": errors["common_temperature_K"],
        "signed_density_increment_error": errors["signed_total_density_m2"],
        "nye_increment_error": errors["exact_reconstructed_nye_m1"],
        "power_signal_error": errors["instantaneous_plastic_power_W_m3"][
            "signal_relative_l2"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
