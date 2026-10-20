#!/usr/bin/env python3
"""Classify matched production trajectories for the V67 multiplication repair.

The comparison is deliberately based on accepted production histories rather
than on instantaneous raw Orowan rates.  It keeps the recovered-v34, legacy
V21, and strictly disabled laws separate and records their source provenance.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _history(result: dict, key: str) -> np.ndarray:
    return np.asarray([row[key] for row in result["response_history"]], dtype=float)


def _last_true_load(result: dict) -> float | None:
    rows = [row for row in result["response_history"]
            if row["connected_physical_band"]]
    return float(rows[-1]["load"]) if rows else None


def _law_summary(path: Path, result: dict) -> dict:
    response = result["response_history"]
    multiplication = result.get("multiplication_history", [])
    initial = response[0]
    final = response[-1]
    raw = float(sum(row["raw_unsigned_physical_line_increment_m"]
                    for row in multiplication))
    accepted = float(sum(row["accepted_unsigned_physical_line_increment_m"]
                         for row in multiplication))
    cells = int(result["configuration"]["n"])**2
    accumulated_absolute_slip = float(sum(
        np.sum(np.asarray(row["absolute_slip_increment_by_owner_family"]))
        for row in multiplication)/cells)
    accumulated_signed_slip = float(sum(
        np.sum(np.asarray(row["signed_slip_increment_by_owner_family"]))
        for row in multiplication)/cells)
    return {
        "result": str(path.resolve()),
        "result_sha256": _sha256(path),
        "source_commit": result["source_commit"],
        "source_tree_clean": bool(result["source_tree_clean"]),
        "multiplication_law": result["configuration"]["multiplication_law"],
        "multiplication_coefficient_m1": result["configuration"][
            "multiplication_coefficient_m1"],
        "front_transmission_fraction": result["configuration"][
            "front_transmission_fraction"],
        "initial_load": float(initial["load"]),
        "final_load": float(final["load"]),
        "intervals": len(response)-1,
        "all_hard_absorption_balances_passed": bool(
            result["decision"]["all_hard_absorption_balances_passed"]),
        "first_law_hard_balance_passed": bool(
            result["decision"]["first_law_hard_balance_passed"]),
        "mura_hard_balance_passed": bool(
            result["decision"]["mura_hard_balance_passed"]),
        "cumulative_raw_multiplication_line_m": raw,
        "cumulative_accepted_multiplication_line_m": accepted,
        "cumulative_accepted_to_raw_fraction": accepted/max(raw, 1e-300),
        "maximum_budget_active_cell_time_fraction": float(max(
            (row["budget_active_cell_time_fraction"] for row in multiplication),
            default=0.0)),
        "accumulated_domain_mean_absolute_family_slip": (
            accumulated_absolute_slip),
        "accumulated_domain_mean_signed_family_slip": accumulated_signed_slip,
        "initial_flow_exposure": float(
            initial["plastic_to_imposed_power_fraction"]),
        "final_flow_exposure": float(final["plastic_to_imposed_power_fraction"]),
        "last_physical_band_load": _last_true_load(result),
        "terminal_physical_band": bool(final["connected_physical_band"]),
        "peak_absolute_mean_shear_stress_Pa": float(
            result["response_summary"]["peak_absolute_mean_shear_stress_Pa"]),
        "endpoint_post_peak_softening_fraction": float(
            result["response_summary"]["endpoint_post_peak_softening_fraction"]),
        "initial_mobile_density_m2": float(initial["mobile_density_m2"]),
        "final_mobile_density_m2": float(final["mobile_density_m2"]),
        "initial_forest_density_m2": float(initial["forest_density_m2"]),
        "final_forest_density_m2": float(final["forest_density_m2"]),
        "initial_wall_density_m2": float(initial["wall_density_m2"]),
        "final_wall_density_m2": float(final["wall_density_m2"]),
        "maximum_temperature_range_K": float(
            result["response_summary"]["maximum_temperature_range_K"]),
        "owner_relative_area_change": {
            owner: float(values["relative_area_change"])
            for owner, values in result["owner_results"].items()
        },
    }


def _plot(results: dict[str, dict], path: Path) -> None:
    style = {
        "recovered_v34": ("#0072B2", "-"),
        "legacy_v21": ("#D55E00", "--"),
        "disabled": ("#4D4D4D", ":"),
    }
    fig, axes = plt.subplots(3, 2, figsize=(10.2, 10.2), sharex=True)
    for name, result in results.items():
        color, line = style[name]
        load = _history(result, "load")
        label = name.replace("_", " ")
        axes[0, 0].plot(load, _history(result, "mean_shear_stress_Pa")/1e9,
                        line, color=color, label=label)
        axes[0, 1].semilogy(load, np.maximum(
            _history(result, "plastic_to_imposed_power_fraction"), 1e-12),
            line, color=color)
        axes[1, 0].plot(load, _history(result, "mobile_density_m2")/1e12,
                        line, color=color)
        axes[1, 1].plot(load, _history(result, "forest_density_m2")/1e12,
                        line, color=color)
        axes[2, 0].plot(load, _history(result, "wall_density_m2")/1e12,
                        line, color=color)
        axes[2, 1].plot(load, _history(result, "temperature_range_K"),
                        line, color=color)
    axes[0, 0].set_ylabel("Mean shear stress (GPa)")
    axes[0, 1].set_ylabel("Accepted flow / imposed power")
    axes[1, 0].set_ylabel(r"Mobile density ($10^{12}$ m$^{-2}$)")
    axes[1, 1].set_ylabel(r"Forest density ($10^{12}$ m$^{-2}$)")
    axes[2, 0].set_ylabel(r"Wall density ($10^{12}$ m$^{-2}$)")
    axes[2, 1].set_ylabel("Temperature range (K)")
    axes[2, 0].set_xlabel("Applied shear strain")
    axes[2, 1].set_xlabel("Applied shear strain")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_channel_histories(results: dict[str, dict], path: Path) -> None:
    style = {
        "recovered_v34": ("#0072B2", "-"),
        "legacy_v21": ("#D55E00", "--"),
        "disabled": ("#4D4D4D", ":"),
    }
    fig, axes = plt.subplots(3, 2, figsize=(10.2, 10.2), sharex=True)
    for name, result in results.items():
        color, line = style[name]
        label = name.replace("_", " ")
        records = result.get("multiplication_history", [])
        if not records:
            continue
        load = np.asarray([row["load"] for row in records], dtype=float)
        raw = np.asarray([
            row["raw_unsigned_physical_line_increment_m"] for row in records])
        accepted = np.asarray([
            row["accepted_unsigned_physical_line_increment_m"] for row in records])
        budget = np.asarray([
            row["budget_active_cell_time_fraction"] for row in records])
        cells = int(result["configuration"]["n"])**2
        absolute_slip = np.asarray([
            np.sum(np.asarray(row["absolute_slip_increment_by_owner_family"]))
            /cells for row in records])
        signed_slip = np.asarray([
            np.sum(np.asarray(row["signed_slip_increment_by_owner_family"]))
            /cells for row in records])
        ratio = np.divide(
            accepted, raw, out=np.full_like(accepted, np.nan), where=raw > 0.0)
        axes[0, 0].semilogy(load, np.maximum(raw, 1e-30), line,
                            color=color, label=label)
        axes[0, 1].semilogy(load, np.maximum(accepted, 1e-30), line,
                            color=color)
        axes[1, 0].plot(load, np.cumsum(absolute_slip), line, color=color)
        axes[1, 1].plot(load, np.cumsum(signed_slip), line, color=color)
        axes[2, 0].plot(load, ratio, line, color=color)
        axes[2, 1].plot(load, budget, line, color=color)
    axes[0, 0].set_ylabel("Raw new line / interval (m)")
    axes[0, 1].set_ylabel("Accepted new line / interval (m)")
    axes[1, 0].set_ylabel("Accumulated absolute slip")
    axes[1, 1].set_ylabel("Accumulated signed slip")
    axes[2, 0].set_ylabel("Accepted / raw multiplication")
    axes[2, 1].set_ylabel("Budget-active cell-time fraction")
    axes[2, 0].set_xlabel("Applied shear strain")
    axes[2, 1].set_xlabel("Applied shear strain")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recovered", type=Path)
    parser.add_argument("legacy", type=Path)
    parser.add_argument("disabled", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    paths = {
        "recovered_v34": args.recovered,
        "legacy_v21": args.legacy,
        "disabled": args.disabled,
    }
    results = {name: _load(path) for name, path in paths.items()}
    summaries = {
        name: _law_summary(paths[name], result)
        for name, result in results.items()
    }
    recovered = summaries["recovered_v34"]
    legacy = summaries["legacy_v21"]
    disabled = summaries["disabled"]
    comparable_endpoint = max(
        abs(recovered["final_load"]-legacy["final_load"]),
        abs(recovered["final_load"]-disabled["final_load"]),
    ) < 1e-12
    zero_exact = disabled["cumulative_accepted_multiplication_line_m"] == 0.0
    legacy_inert = (
        legacy["cumulative_accepted_multiplication_line_m"]
        < 1e-8*recovered["cumulative_accepted_multiplication_line_m"])
    repair_extends_flow = (
        recovered["final_flow_exposure"]
        > 2.0*max(legacy["final_flow_exposure"],
                  disabled["final_flow_exposure"]))
    repair_extends_band = (
        recovered["last_physical_band_load"] is not None
        and disabled["last_physical_band_load"] is not None
        and recovered["last_physical_band_load"]
        > disabled["last_physical_band_load"])
    no_terminal_asb = not recovered["terminal_physical_band"]
    no_softening = recovered["endpoint_post_peak_softening_fraction"] <= 0.0
    if all((comparable_endpoint, zero_exact, legacy_inert,
            repair_extends_flow, repair_extends_band,
            no_terminal_asb, no_softening)):
        classification = (
            "NORMALIZATION_REPAIR_EXTENDS_ACCEPTED_FLOW_AND_BAND_PERSISTENCE_"
            "BUT_STORAGE_ONLY_MULTIPLICATION_STILL_ARRESTS_WITHOUT_ASB_SOFTENING")
    else:
        classification = "MULTIPLICATION_PRODUCTION_COMPARISON_REQUIRES_REVIEW"
    payload = {
        "schema": "v67-multiplication-production-comparison-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "classification": classification,
        "same_parameterization_except_declared_multiplication_law": True,
        "endpoint_load_matched": comparable_endpoint,
        "strict_comparator_exactly_zero": zero_exact,
        "legacy_law_dynamically_inert": legacy_inert,
        "repair_extends_endpoint_flow": repair_extends_flow,
        "repair_extends_band_persistence": repair_extends_band,
        "repaired_path_terminal_asb": not no_terminal_asb,
        "repaired_path_post_peak_softening": not no_softening,
        "cases": summaries,
        "ratios": {
            "recovered_to_legacy_endpoint_flow": (
                recovered["final_flow_exposure"]
                / max(legacy["final_flow_exposure"], 1e-300)),
            "recovered_to_disabled_endpoint_flow": (
                recovered["final_flow_exposure"]
                / max(disabled["final_flow_exposure"], 1e-300)),
            "recovered_to_disabled_final_forest_density": (
                recovered["final_forest_density_m2"]
                / max(disabled["final_forest_density_m2"], 1e-300)),
        },
        "claim_boundary": (
            "The repaired archived normalization and forest routing materially "
            "alter accepted production flow.  This matched trajectory alone "
            "does not qualify terminal ASB, lower-rate DRX, or static RX."),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    report = args.out/"comparison.json"
    report.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    _plot(results, args.out/"response_comparison.png")
    _plot_channel_histories(results, args.out/"channel_comparison.png")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
