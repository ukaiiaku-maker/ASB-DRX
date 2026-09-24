#!/usr/bin/env python3
"""Outcome-neutral, physical-clock analysis of the V53 ASB causal pair."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from full_model.analysis.postprocess_v32_asb_anchor import (
    relative_ledger_invariants, valid_checkpoints,
)
from full_model.analysis.run_v37_conduction_localization import weighted_width


SCHEMA = "asb-drx/v53/asb-mechanism-decision/v1"
CANDIDATE_MAX_POWER_IPR = 0.25
CANDIDATE_MIN_T_CONTRAST_K = 50.0
PERSISTENCE_S = 1.0e-6
MIN_COMPONENT_OVERLAP = 0.25


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def periodic_components(mask: np.ndarray) -> list[np.ndarray]:
    """Return four-neighbour components on a doubly periodic grid."""
    active = np.asarray(mask, dtype=bool)
    if active.ndim != 2:
        raise ValueError("component mask must be two-dimensional")
    seen = np.zeros_like(active); components = []
    nx, ny = active.shape
    for start in zip(*np.nonzero(active)):
        if seen[start]:
            continue
        stack = [start]; seen[start] = True; cells = []
        while stack:
            i, j = stack.pop(); cells.append((i, j))
            for ni, nj in (((i-1) % nx, j), ((i+1) % nx, j),
                           (i, (j-1) % ny), (i, (j+1) % ny)):
                if active[ni, nj] and not seen[ni, nj]:
                    seen[ni, nj] = True; stack.append((ni, nj))
        component = np.zeros_like(active); component[tuple(zip(*cells))] = True
        components.append(component)
    return components


def field_metrics(field: np.ndarray, spacing_m: float) -> tuple[dict, np.ndarray]:
    value = np.asarray(field, dtype=float)
    if value.ndim != 2 or not np.all(np.isfinite(value)):
        raise ValueError("mechanism field must be finite and two-dimensional")
    nonnegative = np.maximum(value, 0.0)
    total = float(nonnegative.sum())
    ipr = (1.0 if total == 0.0 else
           total**2/(value.size*float(np.sum(nonnegative**2))))
    threshold = float(value.mean()+value.std())
    components = periodic_components(value > threshold)
    largest = (max(components, key=np.count_nonzero) if components else
               np.zeros(value.shape, dtype=bool))
    shifted = np.maximum(value-float(value.min()), 0.0)
    widths = weighted_width(shifted, spacing_m) if np.any(shifted) else None
    return {
        "minimum": float(value.min()), "mean": float(value.mean()),
        "maximum": float(value.max()), "standard_deviation": float(value.std()),
        "inverse_participation_fraction": float(ipr),
        "threshold": threshold, "periodic_component_count": len(components),
        "largest_periodic_component_area_fraction": float(largest.mean()),
        "second_moment_widths": widths,
        "undefined_localized_width": widths is None,
    }, largest


def overlap(left: np.ndarray, right: np.ndarray) -> float:
    union = np.count_nonzero(left | right)
    return 0.0 if union == 0 else float(np.count_nonzero(left & right)/union)


def checkpoint_record(path: Path) -> tuple[dict, np.ndarray, dict]:
    with np.load(path, allow_pickle=True) as raw:
        parameters = json.loads(str(raw["P_json"].item()))
        spacing = float(parameters["L_phys"])/int(parameters["Nx"])
        power, component = field_metrics(
            np.asarray(raw["asb_last_plastic_power_W_m3"]), spacing)
        heat, _ = field_metrics(
            np.asarray(raw["asb_last_heat_production_W_m3"]), spacing)
        temperature = np.asarray(raw["T"], dtype=float)
        activity, _ = field_metrics(np.asarray(raw["asb_last_gdot_abs"]), spacing)
        step = int(raw["step"]); time_s = float(raw["sim_time"])
        strain = float((step+1)*parameters["dt_strain_step"])
        ledger = (json.loads(str(raw["v30_asb_last_step_json"].item()))
                  if "v30_asb_last_step_json" in raw else {})
        invariants = relative_ledger_invariants(raw)
        result = {
            "checkpoint": str(path.resolve()), "checkpoint_sha256": digest(path),
            "step": step, "physical_time_s": time_s,
            "applied_strain": strain,
            "applied_strain_semantics": "(step+1)*dt_strain_step",
            "stress_Pa": float(raw["sigma_bar"])*1.0e6,
            "temperature_mean_K": float(temperature.mean()),
            "temperature_peak_K": float(temperature.max()),
            "temperature_peak_minus_mean_K": float(
                temperature.max()-temperature.mean()),
            "work_conjugate_plastic_power": power,
            "independent_irreversible_heat": heat,
            "absolute_shear_rate": activity,
            "saved_field_semantics": "last accepted production step",
            "last_step_ledger": ledger,
            "hard_invariants": invariants,
            "candidate_snapshot": bool(
                power["inverse_participation_fraction"]
                <= CANDIDATE_MAX_POWER_IPR
                and temperature.max()-temperature.mean()
                >= CANDIDATE_MIN_T_CONTRAST_K),
        }
    return result, component, parameters


def physical_parameter_audit(left: dict, right: dict) -> dict:
    allowed = {"causal_temperature_ablation", "restart_file", "nSteps"}
    keys = set(left) | set(right)
    differences = {key: {"baseline": left.get(key), "control": right.get(key)}
                   for key in sorted(keys) if left.get(key) != right.get(key)}
    unintended = {key: value for key, value in differences.items()
                  if key not in allowed and not key.startswith("_")}
    return {
        "declared_allowed_differences": sorted(allowed),
        "all_differences": differences,
        "unintended_physical_differences": unintended,
        "only_declared_intervention_differs": not unintended,
        "baseline_intervention": left.get("causal_temperature_ablation", "none"),
        "control_intervention": right.get("causal_temperature_ablation"),
        "recovery_temperature_routing_declared": True,
        "heat_conduction_and_export_retained": True,
    }


def terminal_record(directory: Path) -> dict:
    path = directory/"v37_conduction_run_record.json"
    record = json.loads(path.read_text())
    return {"path": str(path.resolve()), "sha256": digest(path), **record}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directories = {"baseline": args.baseline_dir.resolve(),
                   "control": args.control_dir.resolve()}
    records = {name: terminal_record(path) for name, path in directories.items()}
    checkpoints = {name: valid_checkpoints(path) for name, path in directories.items()}
    common_steps = sorted(set(checkpoints["baseline"]) & set(checkpoints["control"]))
    trajectories = {"baseline": [], "control": []}; components = {}
    parameters = {}
    for name in trajectories:
        components[name] = {}
        for step in common_steps:
            row, component, parameters[name] = checkpoint_record(
                checkpoints[name][step])
            trajectories[name].append(row); components[name][step] = component
    clock_rows = []
    for left, right in zip(trajectories["baseline"], trajectories["control"]):
        clock_rows.append({
            "step": left["step"],
            "physical_time_exact": left["physical_time_s"] == right["physical_time_s"],
            "applied_strain_exact": left["applied_strain"] == right["applied_strain"],
            "baseline_time_s": left["physical_time_s"],
            "control_time_s": right["physical_time_s"],
        })
    common_parent = (
        records["baseline"].get("intervention_start_checkpoint_sha256")
        == records["control"].get("intervention_start_checkpoint_sha256")
        and records["baseline"].get("intervention_start_checkpoint_sha256")
        is not None)
    parameter_audit = (physical_parameter_audit(
        parameters["baseline"], parameters["control"])
        if common_steps else {"only_declared_intervention_differs": False})
    causal_comparable = bool(
        common_steps and common_parent
        and all(row["physical_time_exact"] and row["applied_strain_exact"]
                for row in clock_rows)
        and parameter_audit["only_declared_intervention_differs"])

    candidate_start = None; persistent = False; overlap_rows = []
    last_component = None
    for row in trajectories["baseline"]:
        current = components["baseline"][row["step"]]
        current_overlap = None if last_component is None else overlap(last_component, current)
        overlap_rows.append({"step": row["step"], "overlap_with_prior": current_overlap})
        if not row["candidate_snapshot"]:
            candidate_start = None
        elif candidate_start is None or (current_overlap is not None
                                         and current_overlap < MIN_COMPONENT_OVERLAP):
            candidate_start = row["physical_time_s"]
        elif row["physical_time_s"]-candidate_start >= PERSISTENCE_S:
            persistent = True
        last_component = current

    hard_valid = bool(common_steps and all(
        row["hard_invariants"]["passed"]
        for values in trajectories.values() for row in values))
    validity_limited = any(
        "VALIDITY_BOUNDARY" in str(record.get("terminal_reason", ""))
        for record in records.values())
    computational_complete = all(record.get("terminal", False)
                                 for record in records.values())
    if not hard_valid:
        classification = "HARD_INVALID_AFFECTED_RESULT"
    elif not causal_comparable:
        classification = "INCOMPLETE_OR_CAUSALLY_UNMATCHED"
    elif validity_limited:
        classification = "VALIDITY_LIMITED_ACCEPTED_COMMON_PREFIX"
    elif persistent:
        classification = "LOCALIZED_CANDIDATE_REQUIRES_GRID_AND_TIMESTEP_REFINEMENT"
    elif computational_complete:
        classification = "VALID_BROAD_OR_NONPERSISTENT_THERMAL_FEEDBACK"
    else:
        classification = "INCOMPLETE_MECHANISM_TEST"
    effect = None
    if common_steps:
        b = trajectories["baseline"][-1]; c = trajectories["control"][-1]
        effect = {
            "common_step": common_steps[-1],
            "temperature_peak_K_control_minus_baseline": (
                c["temperature_peak_K"]-b["temperature_peak_K"]),
            "temperature_contrast_K_control_minus_baseline": (
                c["temperature_peak_minus_mean_K"]
                -b["temperature_peak_minus_mean_K"]),
            "plastic_power_ipr_control_minus_baseline": (
                c["work_conjugate_plastic_power"]["inverse_participation_fraction"]
                -b["work_conjugate_plastic_power"]["inverse_participation_fraction"]),
        }
    result = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "classification": classification,
        "execution": {"computational_complete": computational_complete,
                      "terminal_records": records},
        "evidence_validity": {"hard_invariants_passed": hard_valid,
                              "validity_limited": validity_limited},
        "causal_comparability": {
            "passed": causal_comparable,
            "common_intervention_start_checkpoint": common_parent,
            "common_steps": common_steps, "clock_audit": clock_rows,
            "parameter_audit": parameter_audit,
        },
        "candidate_localization": {
            "screening_candidate": persistent,
            "minimum_duration_s": PERSISTENCE_S,
            "minimum_same_component_overlap": MIN_COMPONENT_OVERLAP,
            "periodic_connectivity": True,
            "overlap_history": overlap_rows,
        },
        "refinement": {"grid_passed": False, "timestep_passed": False,
                       "strict_asb_claimed": False},
        "matched_feedback_effect": effect,
        "trajectories": trajectories,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"classification": classification,
                      "output": str(args.output), "sha256": digest(args.output)}))


if __name__ == "__main__":
    main()
