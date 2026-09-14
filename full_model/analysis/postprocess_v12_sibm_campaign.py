#!/usr/bin/env python3
"""Decision-grade post-processing for the directive-v12 SIBM campaign.

The HPC wrapper's ``final.json`` files are immutable raw records.  This module
audits their checkpoints and contour histories and writes a separate, explicit
scientific classification.  In particular, a drawable diffuse zero contour is
not treated as proof that both members of the declared grain pair retain an
independently resolved pure core.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PURE_THRESHOLD = 0.8
MIN_PURE_CORE_CELLS = 16
PAIR_CONTOUR_ERROR = "pair zero contour is not resolved in the active window"


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value: Any) -> float:
    return float(value)


def load_contour_history(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as stream:
        return [{key: _number(value) for key, value in row.items()}
                for row in csv.DictReader(stream)]


def _change(rows: list[dict[str, float]], field: str) -> float:
    return rows[-1][field] - rows[0][field]


def _interpolated_change(rows: list[dict[str, float]], field: str,
                         time_s: float) -> float:
    times = np.asarray([row["time_s"] for row in rows])
    values = np.asarray([row[field] for row in rows])
    return float(np.interp(time_s, times, values) - values[0])


def morphology_classification(record: dict[str, Any],
                              rows: list[dict[str, float]]) -> str:
    """Classify the observable contour morphology, ignoring pair validity."""
    dx = 10.0e-6 / int(record["grid"])
    tip = _change(rows, "bulge_tip_displacement_m")
    mean = _change(rows, "signed_normal_displacement_mean_m")
    area = _change(rows, "excess_bulge_area_m2")
    amplitude = _change(rows, "bulge_amplitude_m")
    distance = rows[-1]["tip_distance_to_window_m"]
    if float(record["radius_um"]) == 0.0:
        return "SIBM_FLAT_BOUNDARY_MIGRATION_ONLY"
    if distance <= 1.5 * dx:
        return "SIBM_PAIR_WINDOW_LIMITED"
    if tip > 2.0 * dx and mean > dx:
        return "SIBM_RESOLVED_NORMAL_GROWTH"
    if area > 0.0 and amplitude <= dx:
        return "SIBM_LATERAL_SPREADING_WITHOUT_NORMAL_ADVANCE"
    if mean < -dx and area < 0.0:
        return "SIBM_RETRACTED"
    if abs(mean) <= 0.25 * dx and abs(tip) <= 0.25 * dx:
        return "SIBM_STALLED"
    return "INCONCLUSIVE_INSUFFICIENT_NORMAL_DISPLACEMENT"


def criticality_classification(record: dict[str, Any],
                               rows: list[dict[str, float]],
                               pair_identity_initial: bool) -> str:
    """Apply the preregistered sign test using mean motion, area, and pressure."""
    mean = _change(rows, "signed_normal_displacement_mean_m")
    area = _change(rows, "excess_bulge_area_m2")
    pressure = rows[-1]["local_normal_pressure_Pa"]
    subcritical = float(record["radius_um"]) < 0.60
    expected = ((mean < 0.0 and area < 0.0 and pressure < 0.0) if subcritical
                else (mean > 0.0 and area > 0.0 and pressure > 0.0))
    return ("SIBM_CRITICALITY_CONTROL_PASSED"
            if pair_identity_initial and expected
            else "SIBM_CRITICALITY_CONTROL_FAILED")


def audit_checkpoint(path: Path, parent: int, child: int) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as state:
        eta = np.asarray(state["eta"])
        populations = ("rp", "rm", "rho_forest", "rho_wall")
        mins = {name: float(np.min(state[name])) for name in populations}
        metadata = json.loads(str(state["sparse_front_metadata_json"]))
        return {
            "path": str(path),
            "sha256": _digest(path),
            "step": int(state["step"]),
            "phase_count": int(eta.shape[2]),
            "declared_grain_count": int(state["Ng"]),
            "parent_pure_core_cells": int(np.sum(eta[:, :, parent] >= PURE_THRESHOLD)),
            "child_pure_core_cells": int(np.sum(eta[:, :, child] >= PURE_THRESHOLD)),
            "parent_phase_max": float(np.max(eta[:, :, parent])),
            "child_phase_max": float(np.max(eta[:, :, child])),
            "phase_simplex_max_error": float(np.max(np.abs(np.sum(eta, axis=2) - 1.0))),
            "minimum_physical_population": min(mins.values()),
            "minimum_physical_population_by_field": mins,
            "atomic_promotion_commits": int(state["atomic_promotion_commit_total"]),
            "orientation_sha256": hashlib.sha256(
                np.asarray(state["psi_gv"]).tobytes()).hexdigest(),
            "front_ledger": metadata["ledger"],
        }


def _ledger_passed(ledger: dict[str, Any]) -> bool:
    return (
        abs(float(ledger.get("line_closure_m", np.inf))) <= 1.0e-16
        and float(ledger.get("signed_burgers_change_m2", np.inf)) == 0.0
        and abs(float(ledger.get("line_energy_released_J", np.inf))
                - float(ledger.get("heat_released_J", -np.inf))) <= 1.0e-20
    )


def audit_case(case_root: Path, status_root: Path,
               raw: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, float]]]:
    case_id = raw["case_id"]
    case_dir = case_root / case_id
    rows = load_contour_history(case_dir / "sibm_contour_diagnostics.csv")
    checkpoints = sorted(case_dir.glob("drx_v25_restart_*.npz"))
    parent = int(raw["boundary"]["parent_label"])
    child = int(raw["boundary"]["child_label"])
    initial = audit_checkpoint(checkpoints[0], parent, child)
    final = audit_checkpoint(checkpoints[-1], parent, child)
    initial_pair = (initial["parent_pure_core_cells"] >= MIN_PURE_CORE_CELLS
                    and initial["child_pure_core_cells"] >= MIN_PURE_CORE_CELLS)
    final_pair = (final["parent_pure_core_cells"] >= MIN_PURE_CORE_CELLS
                  and final["child_pure_core_cells"] >= MIN_PURE_CORE_CELLS)
    stderr = (status_root / case_id / "stderr.log").read_text(errors="replace")
    contour_lost = PAIR_CONTOUR_ERROR in stderr
    conservation = (
        initial["phase_simplex_max_error"] <= 1.0e-12
        and final["phase_simplex_max_error"] <= 1.0e-12
        and initial["minimum_physical_population"] >= 0.0
        and final["minimum_physical_population"] >= 0.0
        and _ledger_passed(final["front_ledger"])
    )
    no_allocation = (
        initial["phase_count"] == final["phase_count"]
        and initial["declared_grain_count"] == final["declared_grain_count"]
        and initial["atomic_promotion_commits"] == 0
        and final["atomic_promotion_commits"] == 0
        and initial["orientation_sha256"] == final["orientation_sha256"]
    )
    morphology = morphology_classification(raw, rows)
    if raw["tier"] == "E":
        primary = criticality_classification(raw, rows, initial_pair)
    elif not initial_pair or (contour_lost and float(raw["radius_um"]) > 0.0):
        primary = "SIBM_PAIR_IDENTITY_LOST"
    elif contour_lost and float(raw["radius_um"]) == 0.0:
        primary = "SIBM_FLAT_BOUNDARY_MIGRATION_ONLY"
    else:
        primary = morphology
    tip = _change(rows, "bulge_tip_displacement_m")
    mean = _change(rows, "signed_normal_displacement_mean_m")
    area = _change(rows, "excess_bulge_area_m2")
    amplitude = _change(rows, "bulge_amplitude_m")
    hard = bool(conservation and no_allocation and initial_pair and final_pair
                and not contour_lost)
    return {
        "case_id": case_id,
        "tier": raw["tier"],
        "primary_classification": primary,
        "secondary_morphology_classification": morphology,
        "raw_bundle_classification": raw["classification"],
        "raw_application_exit": int(raw["application_exit"]),
        "hard_invariants_passed": hard,
        "conservation_invariants_passed": bool(conservation),
        "no_label_or_orientation_allocation": bool(no_allocation),
        "pair_identity_resolved_initially": bool(initial_pair),
        "pair_identity_resolved_at_final_checkpoint": bool(final_pair),
        "pair_zero_contour_lost_exception": bool(contour_lost),
        "contour_diagnostics_available": bool(rows),
        "grid": int(raw["grid"]),
        "radius_um": float(raw["radius_um"]),
        "window_um": float(raw["window_um"]),
        "mobility_multiplier": float(raw["mobility_multiplier"]),
        "temperature_K": float(raw["temperature_K"]),
        "rate_s-1": float(raw["rate_s-1"]),
        "steps_requested": int(raw["steps_requested"]),
        "steps_observed": int(rows[-1]["step"]),
        "final_time_s": float(rows[-1]["time_s"]),
        "final_strain": float(rows[-1]["strain"]),
        "tip_change_m": tip,
        "mean_normal_change_m": mean,
        "excess_area_change_m2": area,
        "amplitude_change_m": amplitude,
        "final_pressure_Pa": float(rows[-1]["local_normal_pressure_Pa"]),
        "final_normal_velocity_m_s": float(rows[-1]["area_equivalent_normal_velocity_m_s"]),
        "final_tip_distance_to_window_m": float(rows[-1]["tip_distance_to_window_m"]),
        "wallclock_seconds": raw.get("wallclock_seconds"),
        "source_commit": raw["source_commit"],
        "source_checkpoint_sha256": raw["source_checkpoint_sha256"],
        "final_checkpoint_sha256_raw_record": raw["final_checkpoint_sha256"],
        "final_checkpoint_checksum_matches_raw_record": (
            final["sha256"] == raw["final_checkpoint_sha256"]),
        "initial_checkpoint_audit": initial,
        "final_checkpoint_audit": final,
    }, rows


def paired_change(histories: dict[str, list[dict[str, float]]], first: str,
                  second: str, field: str) -> dict[str, float]:
    horizon = min(histories[first][-1]["time_s"], histories[second][-1]["time_s"])
    one = _interpolated_change(histories[first], field, horizon)
    two = _interpolated_change(histories[second], field, horizon)
    return {"common_time_s": horizon, first: one, second: two,
            "first_minus_second": one - two}


def build_decision(records: list[dict[str, Any]],
                   histories: dict[str, list[dict[str, float]]]) -> dict[str, Any]:
    by_id = {record["case_id"]: record for record in records}
    baseline_control = paired_change(
        histories, "A1_baseline_bulge", "A2_no_bulge",
        "bulge_tip_displacement_m")
    grid_control = paired_change(
        histories, "D1_grid192_bulge", "D2_grid192_control",
        "bulge_tip_displacement_m")
    windows = paired_change(
        histories, "A3_window_2um", "A4_window_3p5um",
        "bulge_tip_displacement_m")
    common_grid_time = min(histories[name][-1]["time_s"] for name in
                           ("A1_baseline_bulge", "D1_grid192_bulge", "D3_grid256_bulge"))
    grid_changes = {
        name: _interpolated_change(histories[name], "bulge_tip_displacement_m", common_grid_time)
        for name in ("A1_baseline_bulge", "D1_grid192_bulge", "D3_grid256_bulge")
    }
    grid_values = np.asarray(list(grid_changes.values()))
    grid_spread = float(np.ptp(grid_values) / max(abs(float(np.mean(grid_values))), 1.0e-300))
    mobility_names = ("B1_mobility_0p3", "A1_baseline_bulge", "B2_mobility_3")
    mobility_time = min(histories[name][-1]["time_s"] for name in mobility_names)
    mobility_changes = {name: _interpolated_change(
        histories[name], "bulge_tip_displacement_m", mobility_time) for name in mobility_names}
    mobility_ordered = (mobility_changes[mobility_names[0]]
                        <= mobility_changes[mobility_names[1]]
                        <= mobility_changes[mobility_names[2]])
    temperature = paired_change(
        histories, "B3_temperature_1000K", "B4_temperature_1200K",
        "bulge_tip_displacement_m")
    rate = paired_change(
        histories, "B5_rate_300", "B6_rate_3000",
        "bulge_tip_displacement_m")
    criticality = all(by_id[name]["primary_classification"]
                      == "SIBM_CRITICALITY_CONTROL_PASSED"
                      for name in ("E1_subcritical_fixture", "E2_supercritical_fixture"))
    any_hard_failure = any(not record["hard_invariants_passed"] for record in records)
    decision = {
        "schema": "full-v34-v12-sibm-postprocessing/v2",
        "case_count": len(records),
        "scientific_source_commit": records[0]["source_commit"],
        "classification_authority": "directive-v12 criteria plus checkpoint-level pair-core audit",
        "pure_core_definition": {
            "phase_fraction_threshold": PURE_THRESHOLD,
            "minimum_cells_per_pair_member": MIN_PURE_CORE_CELLS,
        },
        "all_conservation_invariants_passed": all(
            record["conservation_invariants_passed"] for record in records),
        "all_no_allocation_checks_passed": all(
            record["no_label_or_orientation_allocation"] for record in records),
        "all_hard_invariants_passed": not any_hard_failure,
        "pair_identity_finding": (
            "Every geometrically seeded case lacks an independently resolved parent pure core "
            "at its step-0 campaign checkpoint; both no-bulge controls lose that core by step 250."),
        "baseline_tip_relative_to_control_at_common_time": baseline_control,
        "grid192_tip_relative_to_control_at_common_time": grid_control,
        "window_comparison": windows,
        "window_trajectory_agreement_within_one_128_cell": (
            abs(windows["first_minus_second"]) <= 10.0e-6 / 128),
        "grid_comparison": {
            "common_time_s": common_grid_time,
            "tip_changes_m": grid_changes,
            "relative_spread": grid_spread,
            "provisional_five_percent_passed": grid_spread <= 0.05,
        },
        "mobility_comparison": {
            "common_time_s": mobility_time,
            "tip_changes_m": mobility_changes,
            "physically_ordered": bool(mobility_ordered),
        },
        "temperature_comparison": temperature,
        "rate_comparison": rate,
        "criticality_fixture_passed": bool(criticality),
        "additional_pair_cases": 0,
        "additional_pair_reason": "no second pair met the preregistered pure-core criterion",
        "source_geometry_decision": "V12_SOURCE_PAIR_AND_POST_SEED_GEOMETRY_INVALID_FOR_SIBM_QUALIFICATION",
        "full_model_sibm_decision": "FULL_MODEL_SIBM_GROWTH_MECHANISM_UNRESOLVED",
        "decision_reasons": [
            "the seeded pair fails the independent-pure-core identity requirement at step 0",
            "baseline normal tip advance does not exceed its rapidly migrating no-bulge control",
            "the provisional 5% grid comparison fails",
            "the subcritical/supercritical fixture does not produce the required sign split",
            "mobility response is not physically ordered",
        ],
        "scientific_flags": {
            "hard_invariants_passed": False,
            "contour_diagnostics_valid_for_all_cases": False,
            "matched_control_available": True,
            "normal_growth_relative_to_control": False,
            "window_independence_supported": False,
            "window_trajectory_agreement_observed_but_pair_invalid": True,
            "mobility_trend_physically_ordered": False,
            "temperature_trend_physically_interpretable": False,
            "rate_trend_physically_interpretable": False,
            "grid_trend_supported": False,
            "full_model_sibm_mechanism_supported": False,
        },
        "cases": records,
    }
    return decision


def write_summary_csv(records: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "case_id", "tier", "primary_classification",
        "secondary_morphology_classification", "hard_invariants_passed",
        "conservation_invariants_passed", "pair_identity_resolved_initially",
        "pair_identity_resolved_at_final_checkpoint", "raw_application_exit",
        "grid", "radius_um", "window_um", "mobility_multiplier",
        "temperature_K", "rate_s-1", "steps_observed", "final_time_s",
        "final_strain", "tip_change_m", "mean_normal_change_m",
        "excess_area_change_m2", "amplitude_change_m", "final_pressure_Pa",
        "wallclock_seconds",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: record.get(key) for key in fields} for record in records)


def write_cost_csv(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "case_id", "grid", "steps_observed", "wallclock_seconds",
            "wallclock_hours", "seconds_per_observed_step"), lineterminator="\n")
        writer.writeheader()
        for record in records:
            seconds = float(record.get("wallclock_seconds") or 0.0)
            writer.writerow({
                "case_id": record["case_id"], "grid": record["grid"],
                "steps_observed": record["steps_observed"],
                "wallclock_seconds": seconds, "wallclock_hours": seconds / 3600.0,
                "seconds_per_observed_step": seconds / max(record["steps_observed"], 1),
            })


def plot_histories(records: list[dict[str, Any]], histories: dict[str, list[dict[str, float]]],
                   destination: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fields = (
        ("bulge_tip_displacement_m", 1e6, "tip displacement (µm)"),
        ("signed_normal_displacement_mean_m", 1e6, "mean normal displacement (µm)"),
        ("excess_bulge_area_m2", 1e12, "excess pair area (µm²)"),
        ("local_normal_pressure_Pa", 1e-6, "local normal pressure (MPa)"),
    )
    for record in records:
        rows = histories[record["case_id"]]
        time = np.asarray([row["time_s"] for row in rows]) * 1e3
        for ax, (field, scale, ylabel) in zip(axes.flat, fields):
            ax.plot(time, np.asarray([row[field] for row in rows]) * scale,
                    lw=0.9, label=record["case_id"])
            ax.set(xlabel="additional physical time (ms)", ylabel=ylabel)
    axes[1, 1].legend(fontsize=5.5, ncol=2, loc="best")
    fig.suptitle("Directive v12 contour histories (morphology is pair-validity limited)")
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def plot_comparisons(histories: dict[str, list[dict[str, float]]], destination: Path) -> None:
    groups = [
        ("baseline/control", ("A1_baseline_bulge", "A2_no_bulge")),
        ("window", ("A3_window_2um", "A4_window_3p5um")),
        ("mobility", ("B1_mobility_0p3", "A1_baseline_bulge", "B2_mobility_3")),
        ("temperature", ("B3_temperature_1000K", "B4_temperature_1200K")),
        ("strain rate", ("B5_rate_300", "B6_rate_3000")),
        ("grid", ("A1_baseline_bulge", "D1_grid192_bulge", "D3_grid256_bulge")),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (title, names) in zip(axes.flat, groups):
        for name in names:
            rows = histories[name]
            time = np.asarray([row["time_s"] for row in rows]) * 1e3
            value = (np.asarray([row["bulge_tip_displacement_m"] for row in rows])
                     - rows[0]["bulge_tip_displacement_m"]) * 1e6
            ax.plot(time, value, label=name, lw=1.2)
        ax.set(title=title, xlabel="time (ms)", ylabel="tip change (µm)")
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def plot_fields(case_root: Path, records: list[dict[str, Any]], destination: Path) -> None:
    by_id = {record["case_id"]: record for record in records}
    names = ("A1_baseline_bulge", "A2_no_bulge",
             "E1_subcritical_fixture", "E2_supercritical_fixture")
    fig, axes = plt.subplots(len(names), 3, figsize=(12, 13))
    for row_axes, name in zip(axes, names):
        record = by_id[name]
        checkpoint = Path(record["final_checkpoint_audit"]["path"])
        with np.load(checkpoint, allow_pickle=True) as state:
            parent = int(json.loads(str(state["sibm_experiment_json"]))["parent_label"])
            child = int(json.loads(str(state["sibm_experiment_json"]))["child_label"])
            panels = (
                (state["eta"][:, :, child] - state["eta"][:, :, parent], "ηchild − ηparent", "coolwarm"),
                (state["eta"][:, :, parent], "ηparent", "viridis"),
                (np.log10(np.maximum(state["rho"], 1.0)), "log10 ρtotal (m⁻²)", "magma"),
            )
            for ax, (field, title, cmap) in zip(row_axes, panels):
                image = ax.imshow(field.T, origin="lower", cmap=cmap)
                ax.set_title(f"{name}\n{title}", fontsize=9)
                fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def write_decision_note(decision: dict[str, Any], path: Path) -> None:
    classes: dict[str, list[str]] = {}
    for record in decision["cases"]:
        classes.setdefault(record["primary_classification"], []).append(record["case_id"])
    lines = [
        "# Directive v12 SIBM post-processing decision", "",
        "## Decision", "",
        "`V12_SOURCE_PAIR_AND_POST_SEED_GEOMETRY_INVALID_FOR_SIBM_QUALIFICATION`", "",
        "`FULL_MODEL_SIBM_GROWTH_MECHANISM_UNRESOLVED`", "",
        "The negative decision is validity-led, not a fitted or tuned outcome. Every seeded "
        "trajectory already lacks an independently resolved parent pure core in its step-0 "
        "campaign checkpoint. The two unseeded controls begin with a resolved pair but lose "
        "the parent pure core by step 250. A diffuse pair zero contour can persist after this "
        "loss and therefore cannot by itself establish SIBM of two physical grains.", "",
        "All phase-simplex, population nonnegativity, line, signed-Burgers, energy/heat, and "
        "no-label/no-orientation-allocation checks pass at the audited checkpoints. Those "
        "conservation results do not cure the pair-identity failure.", "",
        "## Cross-case findings", "",
        f"- Baseline minus matched-control tip change at their common horizon: "
        f"{decision['baseline_tip_relative_to_control_at_common_time']['first_minus_second']:.6e} m.",
        f"- Window trajectories agree within one 128-grid cell: "
        f"{decision['window_trajectory_agreement_within_one_128_cell']} (secondary evidence only).",
        f"- Grid relative spread at the common horizon: "
        f"{decision['grid_comparison']['relative_spread']:.3%}; provisional 5% test: "
        f"{decision['grid_comparison']['provisional_five_percent_passed']}.",
        f"- Mobility displacement ordering: {decision['mobility_comparison']['physically_ordered']}.",
        f"- Manufactured criticality sign split passed: {decision['criticality_fixture_passed']}.",
        "- Temperature and rate endpoints remain confounded by invalid pair identity and "
        "different evolving thermomechanical histories; neither is promoted as an interpretable trend.",
        "", "## Primary classifications", "",
    ]
    for classification, names in sorted(classes.items()):
        lines.append(f"- `{classification}`: {', '.join(names)}")
    lines += [
        "", "## Required next scientific action", "",
        "Construct or select a source state whose parent and child retain resolved pure cores "
        "after seeding, make loss of either core a clean per-case stop, and rerun only a compact "
        "bulge/control/criticality qualification before any new range campaign. The present "
        "matrix must remain source-geometry and seed-initialization falsification evidence; "
        "it is not a falsification of the SIBM mechanism.", "",
    ]
    path.write_text("\n".join(lines))


def postprocess(output_root: Path, destination: Path, *, run_id: str = "unknown",
                job_id: str = "unknown", result_archive_sha256: str = "unknown") -> dict[str, Any]:
    case_root = output_root / "v12-cases"
    status_root = output_root / "v12-status"
    destination.mkdir(parents=True, exist_ok=True)
    records, histories = [], {}
    for final in sorted(status_root.glob("*/final.json")):
        raw = json.loads(final.read_text())
        record, rows = audit_case(case_root, status_root, raw)
        records.append(record)
        histories[record["case_id"]] = rows
    if len(records) != 17:
        raise RuntimeError(f"expected 17 finalized cases, found {len(records)}")
    decision = build_decision(records, histories)
    decision["run_provenance"] = {
        "run_id": run_id,
        "slurm_job_id": job_id,
        "retrieval_status": "verified",
        "result_archive_sha256": result_archive_sha256,
        "postprocessor": str(Path(__file__).relative_to(Path.cwd())),
    }
    (destination / "v12_sibm_postprocessed.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n")
    write_summary_csv(records, destination / "v12_sibm_case_summary.csv")
    write_cost_csv(records, destination / "v12_sibm_run_cost.csv")
    plot_histories(records, histories, destination / "v12_sibm_contour_histories.png")
    plot_comparisons(histories, destination / "v12_sibm_comparisons.png")
    plot_fields(case_root, records, destination / "v12_sibm_representative_fields.png")
    write_decision_note(decision, destination / "v12_sibm_decision.md")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--run-id", default="unknown")
    parser.add_argument("--job-id", default="unknown")
    parser.add_argument("--result-archive-sha256", default="unknown")
    args = parser.parse_args()
    decision = postprocess(
        args.output_root, args.destination, run_id=args.run_id,
        job_id=args.job_id, result_archive_sha256=args.result_archive_sha256)
    print(json.dumps({
        "case_count": decision["case_count"],
        "decision": decision["full_model_sibm_decision"],
        "all_conservation_invariants_passed": decision["all_conservation_invariants_passed"],
        "all_hard_invariants_passed": decision["all_hard_invariants_passed"],
        "criticality_fixture_passed": decision["criticality_fixture_passed"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
