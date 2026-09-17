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
        for name, item in available.items():
            effects[name] = {
                "delta_active_fraction": item["active_fraction"]-reference["active_fraction"],
                "delta_temperature_max_K": item["temperature_max_K"]-reference["temperature_max_K"],
                "delta_softening_fraction": item["softening_fraction"]-reference["softening_fraction"],
                "delta_effective_width_m": item["effective_width_m"]-reference["effective_width_m"],
            }
    if not all_terminal:
        classification = "V34_THERMAL_CAUSAL_RUNNING"
    elif not all_valid:
        classification = "V34_THERMAL_CAUSAL_HARD_INVALID"
    else:
        classification = "V34_THERMAL_CAUSAL_COMPLETE"
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
        corrected_cases = {}
        corrected_effects = {}
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
                corrected_effects[name] = {
                    "matched_step": selected,
                    "delta_stress_Pa": summary["stress_Pa"]-matched_full["stress_Pa"],
                    "delta_active_fraction": (summary["active_fraction"]
                                              -matched_full["active_fraction"]),
                    "delta_temperature_max_K": (summary["temperature_max_K"]
                                                 -matched_full["temperature_max_K"]),
                    "delta_softening_fraction": (summary["softening_fraction"]
                                                  -matched_full["softening_fraction"]),
                    "delta_effective_width_m": (summary["effective_width_m"]
                                                 -matched_full["effective_width_m"]),
                }
        corrected_relaunch["cases"] = corrected_cases
        corrected_relaunch[
            "causal_effects_relative_to_full_law_at_matched_step"] = corrected_effects
        if corrected_status.get("source_sha"):
            source_commits.add(corrected_status["source_sha"])
    result = {
        "schema": "asb-drx/v34/thermal-causal-comparison/v1",
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
        "cases": available, "causal_effects_relative_to_full_law": effects,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(classification)


if __name__ == "__main__":
    main()
