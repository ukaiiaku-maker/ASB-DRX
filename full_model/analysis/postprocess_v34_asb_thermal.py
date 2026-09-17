#!/usr/bin/env python3
"""Decision output for the checkpoint-shared V34 thermal causal matrix."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from full_model.analysis.postprocess_v32_asb_anchor import effective_support
from full_model.production.asb_classifier import localization_geometry
from full_model.hpc3.run_v34_asb_thermal_case import CASES, case_definition


INVALID_ROUTING_SOURCE_COMMITS = {
    "0c45d036306d56d649c53d69d321d59739930698",
}

EFFECT_UNITS = {
    "delta_stress_Pa": "Pa",
    "delta_active_fraction": "1",
    "delta_temperature_max_K": "K",
    "delta_softening_fraction": "1",
    "delta_effective_width_m": "m",
}


def checkpoints(directory: Path) -> dict[int, Path]:
    result = {}
    for path in directory.glob("drx_v25_restart_*.npz"):
        try:
            with np.load(path, allow_pickle=True) as data:
                result[int(data["step"])] = path
        except (OSError, ValueError, KeyError, EOFError):
            continue
    return result


def actual_semantics(parameters: dict) -> str:
    declared = str(parameters.get("thermal_control_semantics", "auto")).lower()
    if declared == "exact_prescribed_temperature":
        return "EXACT_PRESCRIBED_TEMPERATURE_WITH_THERMOSTAT_EXPORT"
    if float(parameters.get("T_bath_coupling", 0.0)) > 0.0:
        return "FINITE_BATH"
    if float(parameters.get("k_thermal", 0.0)) > 0.0:
        return "FINITE_CONDUCTION_PERIODIC_INSULATED"
    return "NO_CONDUCTION_LOCAL_ADIABATIC"


def terminal_reason(directory: Path) -> str:
    text = "\n".join(path.read_text(errors="replace")[-20000:]
                     for path in sorted(directory.glob("run-from-*.log")))
    if "THERMAL VALIDITY STOP" in text:
        return "THERMAL_MODEL_VALIDITY_BOUNDARY"
    if "MECHANICAL VALIDITY STOP" in text:
        return "MECHANICAL_MODEL_VALIDITY_BOUNDARY"
    if (directory/"v34_thermal_run_record.json").exists():
        record = json.loads((directory/"v34_thermal_run_record.json").read_text())
        return "REQUESTED_HORIZON" if int(record["exit_code"]) == 0 else "DRIVER_FAILURE"
    return "RUNNING"


def diagnostic_at_or_before(directory: Path, step: int) -> dict[str, str]:
    files = sorted(directory.glob("*_asb_diagnostics.csv"))
    if not files:
        return {}
    with files[-1].open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    eligible = [row for row in rows if int(float(row.get("step", -1))) <= step]
    return eligible[-1] if eligible else {}


def read_json_if_present(path: Path) -> dict:
    """Read a small provenance record without making partial runs fatal."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return {}


def process_records(path: Path) -> list[dict[str, object]]:
    """Return launch process identities recorded by the immutable manager."""
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        fields = line.split("\t")
        if len(fields) == 3:
            records.append({"case_id": int(fields[0]), "pid": int(fields[1]),
                            "launched_utc": fields[2]})
    return records


def case_passes_hard_gates(item: dict) -> bool:
    """Apply the unchanged per-case conservation and routing gates."""
    return bool(
        item.get("terminal", False)
        and float(item.get("relative_first_law_residual", float("inf"))) < 0.05
        and float(item.get("maximum_relative_burgers_residual", float("inf"))) < 1e-10
        and float(item.get("maximum_relative_line_residual", float("inf"))) < 1e-10
        and float(item.get("maximum_relative_energy_residual", float("inf"))) < 1e-10
        and item.get("terminal_reason") != "DRIVER_FAILURE"
        and item.get("causal_interpretation_valid", False))


def classify_selected_matrix(available: dict, effects: dict,
                             corrected_cases: dict,
                             corrected_effects: dict,
                             invalid_legacy_cases: list[str]) -> dict:
    """Classify production selections while retaining legacy quarantines."""
    expected = {str(case_definition(index)["case_name"])
                for index in range(len(CASES))}
    selected_cases = dict(available)
    selected_effects = dict(effects)
    selected_cases.update(corrected_cases)
    selected_effects.update(corrected_effects)
    complete = (expected <= set(selected_cases)
                and all(selected_cases[name].get("terminal", False)
                        for name in expected))
    valid = bool(complete and all(
        case_passes_hard_gates(selected_cases[name]) for name in expected))
    reference = "full_law_local_adiabatic"
    comparisons_valid = bool(complete and all(
        selected_effects.get(name, {}).get("status") == "MATCHED_EXACT"
        for name in expected-{reference}))
    corrected_quarantines = sorted(
        name for name in invalid_legacy_cases
        if name in corrected_cases
        and case_passes_hard_gates(corrected_cases[name])
        and corrected_effects.get(name, {}).get("status") == "MATCHED_EXACT")
    if not complete:
        classification = "V34_THERMAL_CAUSAL_RUNNING"
    elif not (valid and comparisons_valid):
        classification = "V34_THERMAL_CAUSAL_HARD_INVALID"
    elif corrected_quarantines:
        classification = (
            "V36_THERMAL_VALID_MATRIX_COMPLETE_WITH_QUARANTINED_LEGACY")
    else:
        classification = "V34_THERMAL_CAUSAL_COMPLETE"
    return {
        "classification": classification,
        "complete": complete,
        "valid": valid,
        "comparisons_valid": comparisons_valid,
        "selected_case_sources": {
            name: ("corrected_selective_relaunch"
                   if name in corrected_cases else "original_matrix")
            for name in sorted(expected)},
        "quarantined_legacy_cases_with_valid_replacements": corrected_quarantines,
    }


def matched_causal_effect(subject: dict, reference: dict,
                          subject_record: dict, reference_record: dict) -> dict:
    """Return a causal contrast only for exact, attributable common states.

    No temporal interpolation is currently admitted.  Every rejected contrast
    therefore retains the same typed keys with null numeric values and an
    explicit reason, preventing consumers from subtracting unmatched states.
    """
    time_tolerance_s = 1e-15
    strain_tolerance = 1e-14
    subject_time = float(subject["sim_time_s"])
    reference_time = float(reference["sim_time_s"])
    subject_strain = float(subject["nominal_strain"])
    reference_strain = float(reference["nominal_strain"])
    time_difference = abs(subject_time-reference_time)
    strain_difference = abs(subject_strain-reference_strain)
    matched_time = time_difference <= time_tolerance_s
    matched_strain = strain_difference <= strain_tolerance
    subject_source = subject_record.get("source_commit")
    reference_source = reference_record.get("source_commit")
    subject_prefix = subject_record.get("shared_checkpoint_sha256")
    reference_prefix = reference_record.get("shared_checkpoint_sha256")
    attributable = bool(subject_source and reference_source
                        and subject_prefix and reference_prefix)
    common_prefix = bool(attributable and subject_prefix == reference_prefix)
    routing_valid = bool(subject.get("causal_interpretation_valid", False)
                         and reference.get("causal_interpretation_valid", False))
    if not routing_valid:
        status = "INVALID_CAUSAL_ROUTING"
    elif not attributable:
        status = "PROVENANCE_UNATTRIBUTABLE"
    elif not common_prefix:
        status = "COMMON_PREFIX_MISMATCH"
    elif not (matched_time and matched_strain):
        status = "UNMATCHED_TIME_OR_STRAIN"
    else:
        status = "MATCHED_EXACT"
    values = {name: None for name in EFFECT_UNITS}
    if status == "MATCHED_EXACT":
        values = {
            "delta_stress_Pa": subject["stress_Pa"]-reference["stress_Pa"],
            "delta_active_fraction": (subject["active_fraction"]
                                      -reference["active_fraction"]),
            "delta_temperature_max_K": (subject["temperature_max_K"]
                                         -reference["temperature_max_K"]),
            "delta_softening_fraction": (subject["softening_fraction"]
                                          -reference["softening_fraction"]),
            "delta_effective_width_m": (subject["effective_width_m"]
                                         -reference["effective_width_m"]),
        }
    return {
        "status": status,
        "numeric_effects": values,
        "units": EFFECT_UNITS,
        "matching": {
            "subject_step": int(subject["step"]),
            "reference_step": int(reference["step"]),
            "subject_time_s": subject_time,
            "reference_time_s": reference_time,
            "absolute_time_difference_s": time_difference,
            "time_tolerance_s": time_tolerance_s,
            "matched_time": matched_time,
            "subject_nominal_strain": subject_strain,
            "reference_nominal_strain": reference_strain,
            "absolute_nominal_strain_difference": strain_difference,
            "nominal_strain_tolerance": strain_tolerance,
            "matched_nominal_strain": matched_strain,
        },
        "interpolation": {
            "used": False,
            "method": "none",
            "status": ("NOT_REQUIRED_EXACT_MATCH" if status == "MATCHED_EXACT"
                       else "NOT_PERMITTED_FOR_THIS_EVIDENCE"),
        },
        "provenance": {
            "attributable": attributable,
            "common_prefix": common_prefix,
            "subject_source_commit": subject_source,
            "reference_source_commit": reference_source,
            "subject_shared_checkpoint_sha256": subject_prefix,
            "reference_shared_checkpoint_sha256": reference_prefix,
        },
    }


def summarize_checkpoint(path: Path, peak_stress: float) -> dict[str, object]:
    with np.load(path, allow_pickle=True) as data:
        parameters = json.loads(str(data["P_json"].item()))
        ledger = json.loads(str(data["v30_asb_cumulative_json"].item()))
        mura = json.loads(str(data["v21_balance_ledger_json"].item()))
        rate = np.asarray(data["asb_last_gdot_abs"], dtype=float)
        temperature = np.asarray(data["T"], dtype=float)
        stress = float(data["sigma_bar"])
        dx = float(parameters["L_phys"])/int(parameters["Nx"])
        active, width = localization_geometry(rate, dx, dx)
        scale = max(abs(float(ledger["external_work_J_m3"])),
                    abs(float(ledger["physical_stored_change_J_m3"]))
                    +abs(float(ledger["thermal_change_J_m3"]))
                    +abs(float(ledger["exported_heat_J_m3"])), 1.0)
        support = effective_support(rate)
        return {
            "step": int(data["step"]), "sim_time_s": float(data["sim_time"]),
            "nominal_strain": (int(data["step"])+1)*float(parameters["dt_strain_step"]),
            "active_fraction": active, "effective_width_m": width,
            **support,
            "softening_fraction": (peak_stress-abs(stress))/max(peak_stress, 1e-300),
            "stress_Pa": stress,
            "temperature_min_K": float(temperature.min()),
            "temperature_mean_K": float(temperature.mean()),
            "temperature_max_K": float(temperature.max()),
            "thermal_semantics": actual_semantics(parameters),
            "causal_temperature_ablation": parameters.get(
                "causal_temperature_ablation", "none"),
            "authoritative_common_temperature_routing": bool(parameters.get(
                "v34_authoritative_common_temperature_routing", False)),
            "physical_heat_deposited_J_m3": float(ledger["deposited_heat_J_m3"]),
            "heat_exported_J_m3": float(ledger["exported_heat_J_m3"]),
            "thermostat_export_J_m3": float(ledger.get(
                "v34_thermostat_export_J_m3", 0.0)),
            "thermal_change_J_m3": float(ledger["thermal_change_J_m3"]),
            "relative_first_law_residual": abs(float(
                ledger["first_law_residual_J_m3"]))/scale,
            "maximum_relative_burgers_residual": float(
                mura["maximum_relative_burgers_rate_residual"]),
            "maximum_relative_line_residual": float(
                mura["maximum_relative_line_balance_residual"]),
            "maximum_relative_energy_residual": float(
                mura["maximum_relative_energy_balance_residual"]),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-step", type=int, default=2500)
    parser.add_argument("--corrected-selective-root", type=Path)
    args = parser.parse_args()
    available = {}
    histories = {}
    for case_id in range(len(CASES)):
        case = case_definition(case_id); name = str(case["case_name"])
        found = checkpoints(args.root/name)
        if not found:
            continue
        histories[name] = found
    common_steps = sorted(set.intersection(*(set(item) for item in histories.values()))) \
        if len(histories) == len(CASES) else []
    common_step = common_steps[-1] if common_steps else None
    for case_id in range(len(CASES)):
        case = case_definition(case_id); name = str(case["case_name"])
        found = histories.get(name, {})
        if not found:
            continue
        stresses = []
        for step in sorted(found):
            with np.load(found[step], allow_pickle=True) as data:
                stresses.append(abs(float(data["sigma_bar"])))
        selected_step = common_step if common_step is not None else max(found)
        summary = summarize_checkpoint(found[selected_step], max(stresses))
        diag = diagnostic_at_or_before(args.root/name, selected_step)
        summary.update(
            terminal_reason=terminal_reason(args.root/name),
            terminal=(args.root/name/"v34_thermal_run_record.json").exists(),
            latest_available_step=max(found),
            flow_operator_T_mean_K=float(diag.get("flow_operator_T_mean_K", "nan")),
            recovery_operator_T_mean_K=float(diag.get(
                "recovery_operator_T_mean_K", "nan")))
        selective = summary["causal_temperature_ablation"] in (
            "freeze_flow", "freeze_recovery", "freeze_flow_and_recovery")
        summary["causal_interpretation_valid"] = bool(
            not selective or summary["authoritative_common_temperature_routing"])
        summary["causal_classification"] = (
            "VALID_CAUSAL_ROUTING" if summary["causal_interpretation_valid"]
            else "INVALID_CAUSAL_ABLATION_ROUTING")
        available[name] = summary
    all_terminal = len(available) == len(CASES) and all(
        item["terminal"] for item in available.values())
    all_valid = len(available) == len(CASES) and all(
        item["relative_first_law_residual"] < 0.05
        and item["maximum_relative_burgers_residual"] < 1e-10
        and item["maximum_relative_line_residual"] < 1e-10
        and item["maximum_relative_energy_residual"] < 1e-10
        and item["terminal_reason"] != "DRIVER_FAILURE"
        and item["causal_interpretation_valid"]
        for item in available.values())
    invalid_causal_cases = sorted(
        name for name, item in available.items()
        if not item["causal_interpretation_valid"])
    effects = {}
    reference = available.get("full_law_local_adiabatic")
    if reference:
        reference_record = read_json_if_present(
            args.root/"full_law_local_adiabatic"/"v34_thermal_run_record.json")
        for name, item in available.items():
            subject_record = read_json_if_present(
                args.root/name/"v34_thermal_run_record.json")
            effects[name] = matched_causal_effect(
                item, reference, subject_record, reference_record)
    source_commits = set()
    for case_id in range(len(CASES)):
        path = args.root/str(case_definition(case_id)["case_name"])/"v34_thermal_run_record.json"
        if path.exists():
            source_commits.add(json.loads(path.read_text())["source_commit"])
    launch_status = read_json_if_present(args.root/"launch_status.json")
    shared_record = read_json_if_present(
        args.root/"shared_prefix_T0900_R3e4_seed43"/"v34_shared_prefix_record.json")
    for record in (launch_status, shared_record):
        if record.get("source_sha"):
            source_commits.add(record["source_sha"])
        if record.get("source_commit"):
            source_commits.add(record["source_commit"])
    if launch_status.get("source_sha") in INVALID_ROUTING_SOURCE_COMMITS:
        invalid_causal_cases = sorted(set(invalid_causal_cases) | {
            str(case_definition(index)["case_name"])
            for index in (1, 2)})
    corrected_relaunch = None
    corrected_cases = {}
    corrected_effects = {}
    if args.corrected_selective_root is not None:
        corrected_status = read_json_if_present(
            args.corrected_selective_root/"launch_status.json")
        corrected_relaunch = {
            "root": str(args.corrected_selective_root),
            "launch_status": corrected_status,
            "launch_processes": process_records(
                args.corrected_selective_root/"processes.tsv"),
            "preflight_classification": "AUTHORITATIVE_ROUTING_PREFLIGHT_PASSED",
        }
        full_history = histories.get("full_law_local_adiabatic", {})
        for case_id in (1, 2):
            case = case_definition(case_id)
            name = str(case["case_name"])
            found = checkpoints(args.corrected_selective_root/name)
            if not found:
                continue
            selected = min(max(found), args.target_step)
            stresses = []
            for path in found.values():
                with np.load(path, allow_pickle=True) as data:
                    stresses.append(abs(float(data["sigma_bar"])))
            summary = summarize_checkpoint(found[selected], max(stresses))
            diag = diagnostic_at_or_before(
                args.corrected_selective_root/name, selected)
            summary.update(
                terminal_reason=terminal_reason(
                    args.corrected_selective_root/name),
                terminal=(args.corrected_selective_root/name/
                          "v34_thermal_run_record.json").exists(),
                latest_available_step=max(found),
                flow_operator_T_mean_K=float(diag.get(
                    "flow_operator_T_mean_K", "nan")),
                recovery_operator_T_mean_K=float(diag.get(
                    "recovery_operator_T_mean_K", "nan")),
                causal_interpretation_valid=bool(
                    summary["authoritative_common_temperature_routing"]),
                causal_classification=(
                    "VALID_CAUSAL_ROUTING" if summary[
                        "authoritative_common_temperature_routing"]
                    else "INVALID_CAUSAL_ABLATION_ROUTING"))
            corrected_cases[name] = summary
            if selected in full_history:
                full_stresses = []
                for path in full_history.values():
                    with np.load(path, allow_pickle=True) as data:
                        full_stresses.append(abs(float(data["sigma_bar"])))
                matched_full = summarize_checkpoint(
                    full_history[selected], max(full_stresses))
                matched_full["causal_interpretation_valid"] = True
                corrected_effects[name] = matched_causal_effect(
                    summary, matched_full,
                    read_json_if_present(args.corrected_selective_root/name/
                                         "v34_thermal_run_record.json"),
                    read_json_if_present(args.root/"full_law_local_adiabatic"/
                                         "v34_thermal_run_record.json"))
        corrected_relaunch["cases"] = corrected_cases
        corrected_relaunch[
            "causal_effects_relative_to_full_law_at_matched_step"] = corrected_effects
        if corrected_status.get("source_sha"):
            source_commits.add(corrected_status["source_sha"])
    selected_matrix = classify_selected_matrix(
        available, effects, corrected_cases, corrected_effects,
        invalid_causal_cases)
    classification = selected_matrix["classification"]
    result = {
        "schema": "asb-drx/v36/thermal-matched-causality/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_commits": sorted(source_commits),
        "shared_checkpoint_step": 100,
        "shared_checkpoint_sha256": shared_record.get("checkpoint_sha256"),
        "launch_status": launch_status,
        "launch_processes": process_records(args.root/"processes.tsv"),
        "corrected_selective_relaunch": corrected_relaunch,
        "target_common_step": args.target_step,
        "latest_common_step": common_step,
        "strict_asb_thresholds_changed": False,
        "strict_asb_claimed_from_causal_matrix": False,
        "invalid_causal_cases": invalid_causal_cases,
        "classification": classification,
        "all_cases_terminal": all_terminal, "all_cases_valid": all_valid,
        "selected_production_matrix": selected_matrix,
        "cases": available, "causal_effects_relative_to_full_law": effects,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(classification)


if __name__ == "__main__":
    main()
