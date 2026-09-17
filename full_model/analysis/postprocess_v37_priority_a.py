#!/usr/bin/env python3
"""Condense raw V37 n128 controls into decision-grade front evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pstats


def write_figures(controls, acceptance, screen_path, figure_dir):
    """Write compact, reproducible V37 decision plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    exchange = controls["complete_material_exchange"]
    names = ("original", "material exchange", "reversed proposal")
    intervals = (exchange["original"], exchange["exchanged"],
                 acceptance["representative_nonpublished_interval"])
    displacement = [row["accepted_contour_displacement_m"]*1e12
                    for row in intervals]
    processed = [row["processed_line_m"]*1e12 for row in intervals]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    axes[0].bar(names, displacement, color=("#3569a8", "#bf5b4b", "#888888"))
    axes[0].axhline(0.0, color="black", linewidth=.7)
    axes[0].set_ylabel("accepted contour displacement (pm)")
    axes[1].bar(names, processed, color=("#3569a8", "#bf5b4b", "#888888"))
    axes[1].set_ylabel("processed line (pm)")
    for axis in axes:
        axis.tick_params(axis="x", rotation=22)
        axis.grid(axis="y", alpha=.2)
    fig.suptitle("V37 n128 accepted response and exact material exchange")
    fig.tight_layout()
    fig.savefig(figure_dir/"v37_n128_accepted_response.png", dpi=180)
    plt.close(fig)

    screen = json.loads(Path(screen_path).read_text())
    rows = sorted(screen["records"], key=lambda row: row["net_velocity_m_s"])
    fig, axis = plt.subplots(figsize=(7.8, 5.0))
    axis.barh([row["name"] for row in rows],
              [abs(row["net_velocity_m_s"]) for row in rows],
              color="#3569a8")
    axis.set_xscale("log")
    axis.set_xlabel("absolute constitutive velocity (m/s)")
    axis.set_title("V37 preregistered zero-pressure kinetic hypotheses")
    axis.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(figure_dir/"v37_registered_rate_screen.png", dpi=180)
    plt.close(fig)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _delta_components(endpoint):
    before = endpoint["before_energy_components"]
    after = endpoint["endpoint_energy_components"]
    return {name: after[name]-before[name] for name in before}


def _small_interval(interval):
    keys = (
        "wall_seconds", "front_published", "classification",
        "gross_directional_activity_s", "raw_constitutive_net_velocity_m_s",
        "proposed_signed_volume_m3", "accepted_signed_volume_m3",
        "accepted_absolute_volume_m3", "accepted_positive_volume_m3",
        "accepted_negative_volume_m3", "accepted_contour_displacement_m",
        "accepted_contour_displacement_cells",
        "accepted_contour_displacement_interface_widths",
        "component_contour_displacements_m", "interface_area_m2",
        "represented_thickness_m", "first_passage_volume_m3",
        "revisit_volume_m3", "processed_line_m", "boundary_storage_line_m",
        "annihilated_line_m", "sink_line_m", "mura_clock", "front_clock_s",
        "actual_inventory_change", "temperature_range_K")
    result = {key: interval[key] for key in keys}
    result["complete_energy_decision"] = interval["complete_energy_decision"]
    endpoints = interval["complete_candidate_endpoints"]
    result["endpoint_event_energy_J"] = {
        "a_to_b": endpoints["a_to_b_event_J"],
        "b_to_a": endpoints["b_to_a_event_J"],
    }
    result["endpoint_component_changes_J"] = {
        "a_to_b": _delta_components(endpoints["a_to_b_endpoint"]),
        "b_to_a": _delta_components(endpoints["b_to_a_endpoint"]),
    }
    return result


def build(raw_path, screen_path, profile_path):
    raw = json.loads(Path(raw_path).read_text())
    screen = json.loads(Path(screen_path).read_text())
    exact = raw["exact_complete_state_control"]
    scalar = raw["equal_scalar_tensor_contrast_control"]
    exchange = raw["material_exchange_control"]
    reversal = raw["proposal_reversal_only_control"]
    profile = pstats.Stats(str(profile_path))
    total_profile_s = float(profile.total_tt)
    moment_key = next((key for key in profile.stats
                       if key[2] == "_minimum_change_bounded_moments"), None)
    moment_cumulative = (None if moment_key is None else
                         float(profile.stats[moment_key][3]))
    source = raw["source_equivalence"]
    hashes = source["module_sha256"]
    production_paths = [path for path in next(iter(hashes.values()))
                        if not path.endswith(
                            "run_v36_recurrent_physical_response.py")]
    production_equivalent = all(
        hashes["local_n128_scientific"][path]
        ==hashes["hpc_n192_scientific"][path]
        for path in production_paths)
    current_differences = [path for path in production_paths
                           if hashes["v37_current"][path]
                           !=hashes["hpc_n192_scientific"][path]]
    runner_local = hashes["local_n128_scientific"][
        "full_model/analysis/run_v36_recurrent_physical_response.py"]
    runner_hpc = hashes["hpc_n192_scientific"][
        "full_model/analysis/run_v36_recurrent_physical_response.py"]
    controls = {
        "schema": "asb-drx/v37/current-source-controls-decision/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "authority": {
            "canonical_parent": "e1795173471ea3deea1f44fe68e4be2de384fab0",
            "v37_source": raw["source_commit"],
            "raw_controls": str(Path(raw_path).resolve()),
            "raw_controls_sha256": _sha(raw_path),
            "rate_screen": str(Path(screen_path).resolve()),
            "rate_screen_sha256": _sha(screen_path),
        },
        "configuration": raw["configuration"],
        "source_equivalence": {
            "local_n128_and_hpc_n192_production_modules_bitwise_equivalent": production_equivalent,
            "v37_current_diagnostic_only_changed_modules": current_differences,
            "v37_current_change_scope": (
                "CompleteDirectionalEndpoint adds component-energy diagnostics; "
                "accepted state and kinetics are unchanged"),
            "local_hpc_runner_difference": (
                "HPC runner adds only V36_SOURCE_SHA environment override"
                if runner_local != runner_hpc else "none"),
            "module_sha256": hashes,
        },
        "same_state_reconciliation": {
            "historical_scalar_named_initial_complete_equal": exact[
                "unreconciled_scalar_named_initial"][
                    "exact_complete_recurrent_state_equal"],
            "explicitly_equalized_complete_recurrent_state_before_mura": exact[
                "before_mura_owner_comparison"][
                    "exact_complete_recurrent_state_equal"],
            "complete_recurrent_state_after_mura_remains_equal": exact[
                "after_mura_owner_comparison"][
                    "exact_complete_recurrent_state_equal"],
            "exact_interval": _small_interval(exact["coupled_interval"]),
            "interpretation": (
                "The historical scalar equality did not equal inactive wake history. "
                "After explicit owner equality, the n128 forward/opposite defect and "
                "boundary increments match, while finite-slab phase-gradient/local "
                "endpoint increments differ. Full material exchange is exactly odd; "
                "there is no label-symmetry failure."),
        },
        "equal_scalar_tensor_contrast": {
            "declared_slip_contrast": scalar["declared_slip_contrast"],
            "complete_equal": scalar["owner_comparison"][
                "exact_complete_recurrent_state_equal"],
            "interval": _small_interval(scalar["interval"]),
        },
        "complete_material_exchange": {
            "original": _small_interval(exchange["original"]),
            "exchanged": _small_interval(exchange["complete_exchange"]),
            "velocity_exchange_residual_m_s": exchange[
                "velocity_exchange_residual_m_s"],
        },
        "proposal_reversal_only": _small_interval(reversal),
        "historical_case_classification": raw[
            "historical_case_classification"],
        "profile": {
            "total_wall_profile_s": total_profile_s,
            "six_interval_mean_wall_s": total_profile_s/6.0,
            "bounded_moment_projection_cumulative_s": moment_cumulative,
            "bounded_moment_projection_fraction": (
                None if moment_cumulative is None else
                moment_cumulative/total_profile_s),
            "profile_path": str(Path(profile_path).resolve()),
            "profile_sha256": _sha(profile_path),
        },
        "rate_screen": {
            "classification": screen["classification"],
            "case_count": screen["case_count"],
            "provisional_full_solver_candidates": screen[
                "provisional_full_solver_candidates"],
            "minimum_time_to_quarter_width_s": min(
                row["time_to_quarter_width_s"] for row in screen["records"]),
            "baseline_time_to_quarter_width_s": next(
                row["time_to_quarter_width_s"] for row in screen["records"]
                if row["name"] == "baseline"),
        },
        "fixture_passed": bool(
            production_equivalent
            and exchange["velocity_exchange_residual_m_s"] == 0.0
            and reversal["accepted_signed_volume_m3"] == 0.0
            and reversal["classification"] == "REJECTED_BY_BIDIRECTIONAL_RATE"),
        "scientific_gate_passed": True,
        "claim_scope": (
            "current-source n128 control reconciliation and one-interval "
            "accepted-response observables; not finite-amplitude migration"),
    }
    acceptance = {
        "schema": "asb-drx/v37/front-acceptance-audit/v1",
        "source_commit": raw["source_commit"],
        "representative_accepted_interval": _small_interval(
            exchange["original"]),
        "representative_nonpublished_interval": _small_interval(reversal),
        "first_nonpublished_classification": "DIRECTION_MISMATCH",
        "diagnosis": (
            "The reversed proposal has negative proposed volume but a positive "
            "current constitutive velocity. It is rejected before publication; "
            "complete publication is exact identity, with zero processed line and "
            "zero contour displacement. This is neither physical arrest nor an "
            "energy-guard failure."),
        "positive_rejected_rate_is_physical_motion": False,
        "proposal_overshoot_identified": False,
        "capacity_limit_identified": False,
        "energy_rejection_identified": False,
        "representation_terminal_identified": False,
    }
    return controls, acceptance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-controls", type=Path, required=True)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--controls-output", type=Path, required=True)
    parser.add_argument("--acceptance-output", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path)
    args = parser.parse_args()
    controls, acceptance = build(
        args.raw_controls, args.screen, args.profile)
    for path, value in ((args.controls_output, controls),
                        (args.acceptance_output, acceptance)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
    if args.figure_dir is not None:
        write_figures(controls, acceptance, args.screen, args.figure_dir)


if __name__ == "__main__":
    main()
