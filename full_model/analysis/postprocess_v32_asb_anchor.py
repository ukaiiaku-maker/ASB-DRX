#!/usr/bin/env python3
"""Checkpoint-safe V32 classification of the four common-Mura ASB anchors."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from full_model.production.asb_classifier import (
    ASBCriteria, classify, matched_history,
)


CRITERIA = ASBCriteria(0.25, 50.0, 0.20, 2.0, 1.0e-6, 0.05)
PAIR_NAMES = {
    "heterogeneous": ("heterogeneous_adiabatic", "heterogeneous_isothermal"),
    "homogeneous": ("homogeneous_adiabatic", "homogeneous_isothermal"),
}


def run_terminal_status(directory: Path) -> dict[str, object]:
    """Read the runner-owned terminal record without consulting fragile PIDs."""
    candidates = (
        directory/"v32_adaptive_run_record.json",
        directory/"v31_anchor_run_record.json",
    )
    record_path = next((path for path in candidates if path.exists()), candidates[-1])
    if not record_path.exists():
        return {"terminal": False, "successful": False, "reason": "RUNNING"}
    try:
        record = json.loads(record_path.read_text())
    except (OSError, ValueError):
        return {"terminal": False, "successful": False,
                "reason": "TERMINAL_RECORD_NOT_READABLE"}
    exit_code = int(record.get("exit_code", -1))
    logs = sorted(directory.glob("run-from-*.log"))
    tail = ""
    if logs:
        try:
            tail = logs[-1].read_text(errors="replace")[-20000:]
        except OSError:
            pass
    if "THERMAL VALIDITY STOP" in tail:
        reason = "THERMAL_MODEL_VALIDITY_BOUNDARY"
    elif "MECHANICAL VALIDITY STOP" in tail:
        reason = "MECHANICAL_MODEL_VALIDITY_BOUNDARY"
    elif exit_code == 0:
        reason = "REQUESTED_HORIZON_OR_CLEAN_DRIVER_STOP"
    else:
        reason = "DRIVER_FAILURE"
    return {"terminal": True, "successful": exit_code == 0,
            "exit_code": exit_code, "reason": reason, "record": record}


def step_from_path(path: Path) -> int:
    return int(re.search(r"(\d+)$", path.stem).group(1))


def valid_checkpoints(directory: Path) -> dict[int, Path]:
    result = {}
    for path in directory.glob("drx_v25_restart_*.npz"):
        try:
            with np.load(path, allow_pickle=True) as data:
                step = int(data["step"])
                if not {"P_json", "T", "asb_last_gdot_abs", "sigma_bar",
                        "sim_time", "v30_asb_cumulative_json",
                        "v21_balance_ledger_json"}.issubset(data.files):
                    continue
            result[step] = path
        except (OSError, ValueError, EOFError, KeyError):
            # A running producer may have a not-yet-atomically-visible file.
            continue
    return result


def effective_support(rate: np.ndarray) -> dict[str, float]:
    value = np.abs(np.asarray(rate, dtype=float)).ravel()
    total = float(np.sum(value))
    if total <= 0.0:
        return {"inverse_participation_fraction": 1.0,
                "entropy_effective_fraction": 1.0}
    probability = value/total
    ipr_fraction = 1.0/(value.size*float(np.sum(probability*probability)))
    positive = probability > 0.0
    entropy_fraction = float(np.exp(
        -np.sum(probability[positive]*np.log(probability[positive])))/value.size)
    return {"inverse_participation_fraction": ipr_fraction,
            "entropy_effective_fraction": entropy_fraction}


def relative_ledger_invariants(data) -> dict[str, float | bool]:
    ledger = json.loads(str(data["v30_asb_cumulative_json"].item()))
    mura = json.loads(str(data["v21_balance_ledger_json"].item()))
    scale = max(abs(float(ledger["external_work_J_m3"])),
                abs(float(ledger["physical_stored_change_J_m3"]))
                +abs(float(ledger["thermal_change_J_m3"]))
                +abs(float(ledger["exported_heat_J_m3"])), 1.0)
    heat_scale = max(abs(float(ledger["deposited_heat_J_m3"])),
                     sum(abs(float(ledger[f"{name}_J_m3"])) for name in (
                         "plastic_drag", "mobile_forest_recovery",
                         "neutral_pair_annihilation", "junction_relaxation",
                         "boundary_recovery")), 1.0)
    result = {
        "relative_first_law_residual": abs(float(
            ledger["first_law_residual_J_m3"]))/scale,
        "relative_dissipation_heat_residual": abs(float(
            ledger["dissipation_heat_residual_J_m3"]))/heat_scale,
        "minimum_physical_channel_W_m3": float(
            ledger["v31_minimum_physical_channel_W_m3"]),
        "maximum_channel_closure_W_m3": float(
            ledger["v31_maximum_channel_closure_W_m3"]),
        "maximum_relative_burgers_residual": float(
            mura["maximum_relative_burgers_rate_residual"]),
        "maximum_relative_line_residual": float(
            mura["maximum_relative_line_balance_residual"]),
        "maximum_relative_energy_residual": float(
            mura["maximum_relative_energy_balance_residual"]),
    }
    result["passed"] = bool(
        result["relative_first_law_residual"] < 0.05
        and result["relative_dissipation_heat_residual"] < 0.05
        and result["minimum_physical_channel_W_m3"] >= -1e-10
        and result["maximum_relative_burgers_residual"] < 1e-10
        and result["maximum_relative_line_residual"] < 1e-10
        and result["maximum_relative_energy_residual"] < 1e-10)
    return result


def pair_history(root: Path, adiabatic_name: str, control_name: str):
    adiabatic = valid_checkpoints(root/adiabatic_name)
    control = valid_checkpoints(root/control_name)
    steps = sorted(set(adiabatic)&set(control))
    if len(steps) < 2:
        raise ValueError(f"fewer than two matched checkpoints for {adiabatic_name}")
    rates, hot, cold, stress, times, support = [], [], [], [], [], []
    ledgers = {}
    final_fields = {}
    for step in steps:
        with np.load(adiabatic[step], allow_pickle=True) as a, np.load(
                control[step], allow_pickle=True) as c:
            rate = np.asarray(a["asb_last_gdot_abs"], dtype=float)
            rates.append(rate); hot.append(np.asarray(a["T"], dtype=float))
            cold.append(np.asarray(c["T"], dtype=float))
            stress.append(float(a["sigma_bar"])); times.append(float(a["sim_time"]))
            support.append(effective_support(rate))
            ledgers[step] = relative_ledger_invariants(a)
            if step == steps[-1]:
                parameters = json.loads(str(a["P_json"].item()))
                final_fields = {
                    "rate": rate, "temperature": np.asarray(a["T"], dtype=float),
                    "control_temperature": np.asarray(c["T"], dtype=float),
                    "rho": np.asarray(a["rho"], dtype=float),
                }
    dx = float(parameters["L_phys"])/int(parameters["Nx"])
    interface = math.sqrt(float(parameters["kappa_eta"])/float(parameters["W_eta"]))
    history = matched_history(
        np.asarray(rates), np.asarray(hot), np.asarray(cold), np.asarray(stress),
        np.asarray(times), dx, dx)
    return history, support, steps, ledgers, parameters, interface, final_fields


def raw_conjunction(history, interface_width):
    flags = []
    for item in history:
        failed = []
        if item.active_fraction > CRITERIA.maximum_active_fraction:
            failed.append("localized_plastic_rate_or_work")
        if item.temperature_excess_K < CRITERIA.minimum_temperature_excess_K:
            failed.append("matched_temperature_excess")
        if item.softening_fraction < CRITERIA.minimum_softening_fraction:
            failed.append("post_peak_softening")
        if item.effective_width_m < CRITERIA.minimum_width_to_interface*interface_width:
            failed.append("resolved_finite_width")
        flags.append((not failed, failed))
    longest = 0.0; start = None; qualifying = 0
    for index, (passed, _) in enumerate(flags):
        if passed:
            qualifying += 1
            if start is None: start = history[index].time_s
            longest = max(longest, history[index].time_s-start)
        else:
            start = None
    return {
        "qualifying_snapshot_count": qualifying,
        "longest_conjunctive_persistence_s": longest,
        "last_failed_criteria": flags[-1][1],
    }


def trend(values, count=6):
    value = np.asarray(values[-count:], dtype=float)
    if value.size < 2: return 0.0
    return float(np.polyfit(np.arange(value.size), value, 1)[0])


def summarize_pair(history, support, steps, interface, target_steps, strain_increment):
    raw = raw_conjunction(history, interface)
    complete = steps[-1] >= target_steps-1
    provisional = classify(history, interface, CRITERIA, refinement_passed=False)
    active = [item.active_fraction for item in history]
    heating = [item.temperature_excess_K for item in history]
    softening = [item.softening_fraction for item in history]
    if not complete:
        diagnosis = "ANCHOR_INCOMPLETE_UNDEREXPOSURE_NOT_EXCLUDED"
    elif raw["qualifying_snapshot_count"]:
        diagnosis = "LOCALIZATION_REQUIRES_REFINEMENT_CONFIRMATION"
    elif min(active[1:], default=1.0) <= CRITERIA.maximum_active_fraction:
        diagnosis = "TRANSIENT_LOCALIZATION_STRICT_CONJUNCTION_NOT_MET"
    else:
        diagnosis = "BROAD_RESPONSE_NO_STRICT_LOCALIZATION_AT_ANCHOR"
    return {
        "latest_step": steps[-1], "target_steps": target_steps,
        "progress_fraction": min((steps[-1]+1)/target_steps, 1.0),
        "latest_nominal_strain": float((steps[-1]+1)*strain_increment),
        "complete": complete, "diagnosis": diagnosis,
        "strict_decision_without_refinement": asdict(provisional),
        "raw_conjunction": raw,
        "minimum_active_fraction_after_initial": min(active[1:], default=active[0]),
        "latest_active_fraction": active[-1],
        "latest_effective_width_m": history[-1].effective_width_m,
        "maximum_temperature_excess_K": max(heating),
        "latest_temperature_excess_K": heating[-1],
        "maximum_softening_fraction": max(softening),
        "latest_softening_fraction": softening[-1],
        "latest_inverse_participation_fraction": support[-1][
            "inverse_participation_fraction"],
        "latest_entropy_effective_fraction": support[-1][
            "entropy_effective_fraction"],
        "recent_active_fraction_slope_per_checkpoint": trend(active),
        "recent_heating_slope_K_per_checkpoint": trend(heating),
        "recent_softening_slope_per_checkpoint": trend(softening),
        "matched_steps": steps,
        "history": [asdict(item) | support[index]
                    for index, item in enumerate(history)],
    }


def plot(result, fields, output: Path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    colors = {"heterogeneous": "#2468a2", "homogeneous": "#e18d2d"}
    for name, record in result["pairs"].items():
        h = record["history"]
        strain = ((np.array(record["matched_steps"]) + 1)
                  * record["strain_increment"])
        axes[0, 0].plot(strain, [x["active_fraction"] for x in h], "o-",
                        ms=3, color=colors[name], label=name)
        axes[0, 1].plot(strain, [x["temperature_excess_K"] for x in h], "o-",
                        ms=3, color=colors[name], label=name)
        axes[0, 2].plot(strain, [x["softening_fraction"] for x in h], "o-",
                        ms=3, color=colors[name], label=name)
    axes[0, 0].axhline(CRITERIA.maximum_active_fraction, color="k", ls="--")
    axes[0, 1].axhline(CRITERIA.minimum_temperature_excess_K, color="k", ls="--")
    axes[0, 2].axhline(CRITERIA.minimum_softening_fraction, color="k", ls="--")
    axes[0, 0].set_ylabel("Active fraction"); axes[0, 1].set_ylabel("Matched ΔT (K)")
    axes[0, 2].set_ylabel("Post-peak softening fraction")
    for ax in axes[0]: ax.set_xlabel("Nominal strain"); ax.grid(alpha=.2); ax.legend()
    hetero = fields["heterogeneous"]
    length_um = 1.0e6*float(hetero["domain_length_m"])
    extent = (0, length_um, 0, length_um)
    for ax, key, title in zip(axes[1], ("rate", "temperature", "rho"),
                              ("|plastic rate| (s⁻¹)", "Temperature (K)", "Density (m⁻²)")):
        image = ax.imshow(hetero[key].T, origin="lower", extent=extent, cmap="magma")
        ax.set_title(title); ax.set_xlabel("x (μm)"); ax.set_ylabel("y (μm)")
        fig.colorbar(image, ax=ax, shrink=.8)
    fig.suptitle("V32 common-Mura ASB anchor checkpoint classification")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    parser.add_argument("--target-steps", type=int, default=5000)
    args = parser.parse_args()
    pairs = {}; invariants = {}; fields = {}; runs = {}
    for name, case_names in PAIR_NAMES.items():
        history, support, steps, ledger, parameters, interface, final = pair_history(
            args.root, *case_names)
        strain_increment = float(parameters["dt_strain_step"])
        pairs[name] = summarize_pair(
            history, support, steps, interface, args.target_steps,
            strain_increment)
        pairs[name]["strain_increment"] = strain_increment
        invariants[name] = ledger[steps[-1]]
        fields[name] = final | {"domain_length_m": float(parameters["L_phys"])}
        runs[name] = {
            "adiabatic": run_terminal_status(args.root/case_names[0]),
            "control": run_terminal_status(args.root/case_names[1]),
        }
    all_complete = all(pair["complete"] for pair in pairs.values())
    all_terminal = all(run["terminal"] for pair in runs.values()
                       for run in pair.values())
    all_successful = all(run["successful"] for pair in runs.values()
                         for run in pair.values())
    validity_limited = any("VALIDITY_BOUNDARY" in str(run["reason"])
                           for pair in runs.values() for run in pair.values())
    all_invariants = all(item["passed"] for item in invariants.values())
    if not all_invariants or (all_terminal and not all_successful):
        classification = "HARD_INVALID_ASB_ANCHOR"
    elif not all_terminal:
        classification = "RUNNING_LONG_ASB_ANCHOR"
    elif pairs["heterogeneous"]["raw_conjunction"]["qualifying_snapshot_count"]:
        classification = "ASB_LOCALIZATION_CANDIDATE_REQUIRES_REFINEMENT"
    elif validity_limited:
        classification = "ASB_ANCHOR_VALIDITY_LIMITED_MECHANISTIC_NEGATIVE"
    else:
        classification = "ASB_ANCHOR_MECHANISTIC_NEGATIVE"
    result = {
        "schema": "asb-drx/v32/asb-anchor-decision/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": "c643afe00b4ea6f7f30e021675d7648ecdb6422c",
        "root": str(args.root), "criteria": asdict(CRITERIA),
        "pairs": pairs, "runs": runs, "latest_invariants": invariants,
        "all_cases_complete": all_complete,
        "all_cases_terminal": all_terminal,
        "all_cases_successful": all_successful,
        "validity_limited": validity_limited,
        "all_invariants_passed": all_invariants,
        "classification": classification,
        "adaptive_screen_authorized": bool(
            all_terminal and all_successful and all_invariants
            and classification != "HARD_INVALID_ASB_ANCHOR"),
        "strict_asb_claimed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    if args.figure:
        plot(result, fields, args.figure)
    print(classification)


if __name__ == "__main__":
    main()
