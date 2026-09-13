#!/usr/bin/env python3
"""Decision-grade summary of one full-v34 output directory."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def _number(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out


def _rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _values(rows, name):
    return [_number(row.get(name)) for row in rows]


def _finite_max(rows, name, default=math.nan):
    values = [value for value in _values(rows, name) if math.isfinite(value)]
    return max(values) if values else default


def _finite_min(rows, name, default=math.nan):
    values = [value for value in _values(rows, name) if math.isfinite(value)]
    return min(values) if values else default


def _finite_last(rows, name, default=math.nan):
    values = [value for value in _values(rows, name) if math.isfinite(value)]
    return values[-1] if values else default


def _finite_sum(rows, name, default=0.0):
    values = [value for value in _values(rows, name) if math.isfinite(value)]
    return sum(values) if values else default


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint(case_dir):
    paths = sorted(Path(case_dir).glob("drx_v25_restart_*.npz"))
    if not paths:
        return None, {}
    path = paths[-1]
    with np.load(path, allow_pickle=True) as state:
        result = {"path": path.name, "sha256": _sha256(path)}
        if "H_nuc" in state.files and "E_nuc" in state.files:
            ratio = np.asarray(state["H_nuc"]) / np.maximum(np.asarray(state["E_nuc"]), 1e-300)
            result["max_hazard_threshold_ratio"] = float(np.nanmax(ratio))
            result["minimum_stochastic_threshold"] = float(np.nanmin(state["E_nuc"]))
        for name in ("nuc_cand_active", "nuc_cand_age", "nuc_cand_best_barrier"):
            if name in state.files:
                array = np.asarray(state[name])
                if name.endswith("active"):
                    result["checkpoint_active_candidates"] = int(np.sum(array))
                elif name.endswith("age"):
                    result["checkpoint_max_candidate_age"] = int(np.max(array))
        if "step" in state.files:
            result["global_step"] = int(state["step"])
        if "sim_time" in state.files:
            result["physical_time_s"] = float(state["sim_time"])
        for name in (
                "area_hazard_total_exposure", "area_hazard_residual_exposure",
                "area_hazard_expected_raw_trigger_count", "area_hazard_next_threshold",
                "area_hazard_completed_events", "atomic_promotion_attempt_total",
                "atomic_promotion_commit_total", "atomic_promotion_rollback_total"):
            if name in state.files:
                result[name] = np.asarray(state[name]).item()
        if "atomic_promotion_events_json" in state.files:
            events = json.loads(str(state["atomic_promotion_events_json"].item()))
            result["atomic_promotion_events"] = events
        if "physical_grain_tracker_json" in state.files:
            tracker = json.loads(str(state["physical_grain_tracker_json"].item()))
            result["physical_grain_records"] = [
                record for record in tracker.get("records", [])
                if record.get("embryo_promoted", False)
            ]
    return path, result


def _relative(value, reference):
    if not math.isfinite(value) or not math.isfinite(reference):
        return math.nan
    return abs(value - reference) / max(abs(reference), 1e-300)


def summarize(case_dir, branch, matched_csv=None):
    case_dir = Path(case_dir)
    csv_path = case_dir / "drx_v25_restart_asb_diagnostics.csv"
    rows = _rows(csv_path)
    checkpoint_path, checkpoint = _checkpoint(case_dir)
    stress = _values(rows, "sigma_MPa")
    finite_stress = [value for value in stress if math.isfinite(value)]
    peak_stress = max(finite_stress) if finite_stress else math.nan
    final_stress = finite_stress[-1] if finite_stress else math.nan
    softening = ((peak_stress - final_stress) / peak_stress
                 if math.isfinite(peak_stress) and peak_stress > 0.0 else math.nan)

    chain = {
        "eligible_sites_max": int(_finite_max(rows, "nuc_candidates", 0.0)),
        "hazard_rate_max_s-1": _finite_max(rows, "nuc_hazard_max", 0.0),
        "cumulative_hazard_max": _finite_max(rows, "nuc_Hmax", 0.0),
        "candidate_creations": int(_finite_sum(rows, "nuc_candidate_new")) if "nuc_candidate_new" in rows[0] else 0,
        "active_candidates_max": int(_finite_max(rows, "nuc_candidate_active", 0.0)),
        "promotable_candidates_max": int(_finite_max(rows, "nuc_candidate_promotable", 0.0)),
        "allocated_hazard_births_max": int(_finite_max(rows, "grain_hazard_births", 0.0)),
        "allocated_labels_max": int(_finite_max(rows, "n_grains", 0.0)),
        "physical_grains": (int(_finite_last(rows, "physical_drx_grains", 0.0))
                            if "physical_drx_grains" in rows[0] else None),
    }
    ratio = checkpoint.get("max_hazard_threshold_ratio", math.nan)
    if chain["eligible_sites_max"] == 0:
        first_failure = "site_activation"
    elif chain["hazard_rate_max_s-1"] <= 0.0:
        first_failure = "hazard_rate"
    elif math.isfinite(ratio) and ratio < 1.0 and chain["candidate_creations"] == 0:
        first_failure = "hazard_integration_did_not_reach_stochastic_threshold"
    elif chain["candidate_creations"] == 0:
        first_failure = "candidate_creation_or_unrecorded_raw_trigger"
    elif chain["active_candidates_max"] == 0:
        first_failure = "candidate_persistence"
    elif chain["promotable_candidates_max"] == 0:
        first_failure = "promotion_eligibility"
    elif chain["allocated_hazard_births_max"] == 0:
        first_failure = "phase_support_or_label_allocation"
    elif chain["physical_grains"] and chain["physical_grains"] > 0:
        first_failure = None
    else:
        first_failure = "physical_grain_recognition"
    chain["first_failing_stage"] = first_failure
    if "nuc_raw_trigger_total" in rows[0]:
        chain["raw_stochastic_attempts"] = int(
            _finite_last(rows, "nuc_raw_trigger_total", 0.0))
        chain["raw_stochastic_attempts_provenance"] = "explicit_cumulative_counter"
    else:
        chain["raw_stochastic_attempts"] = None
        chain["raw_stochastic_attempts_provenance"] = (
            "not_recorded_separately_by_this_source; final_H_over_E_below_one_does_not_"
            "exclude_a_prior_trigger_followed_by_a_comoving_GB_or_swept-cell_reset")
    if (chain["raw_stochastic_attempts"] is None
            and chain["first_failing_stage"]
            == "hazard_integration_did_not_reach_stochastic_threshold"):
        chain["first_failing_stage"] = (
            "hazard_exposure_or_unrecorded_raw_trigger_requires_explicit_counter")
    elif (chain["raw_stochastic_attempts"] is not None
          and chain["raw_stochastic_attempts"] > 0
          and chain["candidate_creations"] == 0):
        chain["first_failing_stage"] = "candidate_creation_or_viability"

    lifecycle = None
    hazard_exposure = None
    promotion = None
    if "hazard_exposure_total" in rows[0]:
        raw = int(_finite_last(rows, "nuc_raw_trigger_total", 0.0))
        viable = int(_finite_last(rows, "nuc_raw_viable_trigger_total", 0.0))
        records = int(_finite_last(rows, "embryo_records_total", 0.0))
        active = int(_finite_last(rows, "embryo_active_total", 0.0))
        promotable = int(_finite_last(rows, "embryo_promotable_total", 0.0))
        commits = int(_finite_last(rows, "atomic_promotion_commit_total", 0.0))
        physical = int(_finite_last(rows, "physical_drx_grains", 0.0))
        if physical > 0:
            lifecycle = "PHYSICAL_DRX_GRAIN"
            chain["first_failing_stage"] = None
        elif commits > 0:
            lifecycle = "PROMOTED_LABEL_NOT_PHYSICAL_GRAIN"
        elif promotable > 0 or int(_finite_last(rows, "atomic_promotion_attempt_total", 0.0)) > 0:
            lifecycle = "SUPERCRITICAL_EMBRYO_NO_PROMOTION"
        elif records > 0 and active > 0:
            lifecycle = "PERSISTENT_SUBCRITICAL_EMBRYO"
        elif raw > 0:
            lifecycle = "TRIGGERED_NO_PERSISTENT_EMBRYO"
        else:
            lifecycle = "ZERO_TRIGGER_AFTER_BOUNDED_EXPOSURE"
            chain["first_failing_stage"] = "hazard_exposure_below_global_event_clock"
        hazard_exposure = {
            "site_max": _finite_last(rows, "hazard_exposure_site_max"),
            "site_mean": _finite_last(rows, "hazard_exposure_site_mean"),
            "site_quantiles": rows[-1].get("hazard_exposure_site_quantiles"),
            "total": _finite_last(rows, "hazard_exposure_total"),
            "expected_raw_trigger_count": _finite_last(rows, "expected_raw_trigger_count"),
            "probability_at_least_one_raw_trigger": _finite_last(rows, "probability_at_least_one_raw_trigger"),
            "discarded": _finite_last(rows, "hazard_exposure_discarded", 0.0),
            "transferred": _finite_last(rows, "hazard_exposure_transferred", 0.0),
            "newly_initialized": _finite_last(rows, "hazard_exposure_newly_initialized", 0.0),
            "threshold_redraws_after_event": int(_finite_last(rows, "hazard_threshold_redraws_after_event", 0.0)),
            "threshold_redraws_other": int(_finite_last(rows, "hazard_threshold_redraws_other", 0.0)),
            "deferred_event_present": bool(_finite_last(rows, "hazard_deferred_event_present", 0.0)),
        }
        promotion = {
            "raw_triggers": raw, "viable_triggers": viable,
            "rejected_not_viable": int(_finite_sum(rows, "nuc_raw_rejected_not_viable", 0.0)),
            "embryo_records": records, "active_embryos": active,
            "promotable_embryos": promotable,
            "retired_embryos": int(_finite_last(rows, "embryo_retired_total", 0.0)),
            "attempts": int(_finite_last(rows, "atomic_promotion_attempt_total", 0.0)),
            "commits": commits,
            "rollbacks": int(_finite_last(rows, "atomic_promotion_rollback_total", 0.0)),
            "heat_released_J": _finite_last(rows, "atomic_promotion_heat_released_J", 0.0),
            "energy_closure_J": _finite_last(rows, "atomic_promotion_energy_closure_J", 0.0),
            "line_closure_m": _finite_last(rows, "atomic_promotion_line_closure_m", 0.0),
            "signed_burgers_change_m2": _finite_last(rows, "atomic_promotion_signed_burgers_change_m2", 0.0),
            "physical_drx_grains": physical,
            "recrystallized_area_fraction": _finite_last(rows, "physical_recrystallized_area_fraction", 0.0),
        }
        events = checkpoint.get("atomic_promotion_events", [])
        if events:
            promotion["committed_event_ledgers"] = events
        grain_records = checkpoint.get("physical_grain_records", [])
        if grain_records:
            promotion["promoted_grain_records"] = grain_records

    summary = {
        "schema": "asb-drx-full-v34-case-summary/v1",
        "architecture": "full_2d_phase_field",
        "branch": branch,
        "physical_horizon": {
            "diagnostic_rows": len(rows),
            "final_step": int(_finite_last(rows, "step", -1)),
            "final_strain": _finite_last(rows, "eps_pct") / 100.0,
            "final_time_s": _finite_last(rows, "t_us") * 1e-6,
        },
        "stress_temperature": {
            "peak_stress_MPa": peak_stress,
            "final_stress_MPa": final_stress,
            "post_peak_softening_fraction": softening,
            "maximum_temperature_K": _finite_max(rows, "T_max"),
            "maximum_temperature_range_K": _finite_max(rows, "asb_T_range"),
        },
        "localization_observables": {
            "maximum_top5_plastic_rate_fraction": _finite_max(rows, "asb_gdot_top5_frac"),
            "maximum_top5_plastic_power_fraction": _finite_max(rows, "asb_qdot_top5_frac"),
            "maximum_temperature_anisotropy": _finite_max(rows, "asb_band_anisotropy_T"),
            "minimum_hot_cold_density_ratio": _finite_min(rows, "asb_rho_hot_over_cold"),
        },
        "candidate_to_grain_chain": chain,
        "checkpoint": checkpoint,
        "asb_classification": "NOT_EVALUATED_REQUIRES_MATCHED_ISOTHERMAL_CONTROL_AND_FIELD_WIDTH_REFINEMENT",
        "physical_grain_classification": lifecycle or "NOT_AVAILABLE_UNTIL_PATCH_E",
        "claim_level": "integrated_regression",
    }
    if hazard_exposure is not None:
        summary["hazard_exposure"] = hazard_exposure
        summary["embryo_and_atomic_promotion"] = promotion
    if matched_csv is not None:
        control = _rows(matched_csv)
        control_peak = _finite_max(control, "sigma_MPa")
        control_tmax = _finite_max(control, "T_max")
        summary["matched_control"] = {
            "diagnostics": str(Path(matched_csv).resolve()),
            "peak_stress_relative_difference": _relative(peak_stress, control_peak),
            "maximum_temperature_relative_difference": _relative(
                summary["stress_temperature"]["maximum_temperature_K"], control_tmax),
            "top5_plastic_rate_fraction_absolute_difference": abs(
                summary["localization_observables"]["maximum_top5_plastic_rate_fraction"]
                - _finite_max(control, "asb_gdot_top5_frac")),
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--matched-csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.case_dir, args.branch, args.matched_csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
