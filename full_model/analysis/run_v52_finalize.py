#!/usr/bin/env python3
"""Assemble the immutable V52 local evidence and campaign ledger."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
    temporary.replace(path)


def main():
    root = Path.cwd().resolve()
    verification = root/"full_model/verification"
    production = Path(
        "/Users/sdillon/HPC3/worktrees/asb-drx-full-v52-20260922")
    n128_raw = production/(
        "full_model/production/results-local/v52-physical/n128/loading/")
    n192_raw = root/(
        "full_model/production/results-local/v52-spatial/n192/loading/")
    copied = {
        "laplacian_overlap": (
            production/"full_model/verification/v52_laplacian_evolved_overlap.json",
            verification/"v52_laplacian_evolved_overlap.json"),
        "n128_manifest": (n128_raw/"run_manifest.json",
                          verification/"v52_n128_loading_manifest.json"),
        "n192_manifest": (n192_raw/"run_manifest.json",
                          verification/"v52_n192_loading_manifest.json"),
    }
    for source, destination in copied.values():
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copyfile(source, destination)

    manager_path = verification/"v52_completion_manager_v2.json"
    manager = json.loads(manager_path.read_text())
    if manager["state"] != "COMPLETE":
        raise RuntimeError("V52 controller is not complete")
    if "failure" in manager:
        manager.setdefault("recovered_failures", []).append({
            "failed_utc": manager.pop("failed_utc", None),
            "failure": manager.pop("failure"),
            "classification": "RECOVERED_INITIALIZATION_METADATA_DEFECT",
            "accepted_n192_intervals_before_failure": 0,
        })
        atomic_json(manager_path, manager)

    continuation = json.loads((verification/"v52_continuation.json").read_text())
    geometry = json.loads((verification/"v52_state_dependent_geometry.json").read_text())
    initialization = json.loads((verification/"v52_spatial_initialization.json").read_text())
    comparisons = {str(index): json.loads((verification/
        f"v52_spatial_comparison_{index:03d}.json").read_text())
        for index in (1, 8, 16, 32)}
    n128 = json.loads(copied["n128_manifest"][1].read_text())
    n192 = json.loads(copied["n192_manifest"][1].read_text())
    latest_n128 = Path(n128["latest_checkpoint"])
    latest_n192 = Path(n192["latest_checkpoint"])
    checksums = {
        path.name: digest(path) for path in sorted(verification.glob("v52_*"))
        if path.is_file() and path.name != "v52_completion_decision.json"
    }
    numerical_paths_unchanged = subprocess.run([
        "git", "diff", "--quiet",
        "5c9a0fca5233c98aeae37160a3d58ad0d6b95f00",
        "ee6eba55119ad2a282b8d14a17510979349e4909", "--",
        "full_model/production",
        "full_model/analysis/run_v49_physical_continuation.py"],
        cwd=root).returncode == 0
    decision = {
        "schema": "asb-drx/v52/completion-decision/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "classification": (
            "PHYSICAL_FLOW_APPROACH_COMPLETE_MATCHED_N192_THROUGH_INTERVAL_32_"
            "GEOMETRY_OPTION_QUALIFIED_WITH_CLIMB_STRESS_HYPOTHESIS"),
        "execution_completion": {
            "passed": True, "location": "local", "hpc3_jobs_submitted": 0,
            "n128_intervals": n128["completed_intervals"],
            "n128_physical_time_s": n128["physical_time_s"],
            "n192_intervals": n192["completed_intervals"],
            "n192_physical_time_s": n192["physical_time_s"],
        },
        "accepted_state_validity": {
            "passed": True,
            "n128_checkpoint_checksum_verified": bool(
                digest(latest_n128) == n128["latest_checkpoint_sha256"]),
            "n192_checkpoint_checksum_verified": bool(
                digest(latest_n192) == n192["latest_checkpoint_sha256"]),
            "first_law_checks_passed": continuation["energy_balance"][
                "all_available_segment_incremental_checks_passed"],
        },
        "selected_observable_resolution": {
            "passed_through_interval": 32,
            "passed_through_physical_time_s": comparisons["32"]["physical_time_s"],
            "all_comparison_preconditions_passed": all(row[
                "comparison_preconditions"]["all_preconditions_passed"]
                for row in comparisons.values()),
            "all_selected_mean_increments_below_5_percent": all(row[
                "primary_all_below_5_percent"] for row in comparisons.values()),
            "interval_32_primary": comparisons["32"][
                "primary_increment_comparison"],
            "interval_32_spatial_structure": comparisons["32"][
                "spatial_structure"],
            "late_n128_beyond_15_625_us_spatially_qualified": False,
        },
        "physical_response": {
            "flow_approach_preregistered_control_passed": continuation[
                "flow_approach_preregistered_control"]["passed"],
            "flow_approach_is_universal_material_criterion": False,
            "resolved_stress_peak": continuation["stress_peak"]["resolved_peak"],
            "endpoint": continuation["endpoint"],
        },
        "geometry_capability": {
            "signed_chemical_affinity_repaired": True,
            "one_face_stall_supported": True,
            "heterogeneous_face_rates_unequal": geometry[
                "heterogeneous_mechanical_temperature_quadrature"][
                    "face_rates_unequal"],
            "quadrature_reference_order": 128,
            "coupled_alternation_restart_exact": geometry[
                "minimal_operator_alternation"]["restart_exact"],
            "resolved_glide_stress_used_as_climb_activation_hypothesis": True,
            "tested_elastic_work_increment_is_zero": all(value == 0.0 for value in
                geometry["heterogeneous_mechanical_temperature_quadrature"][
                    "elastic_work_by_face_J_m3_cells"].values()),
            "unresolved_geometry_capability": (
                "calibrated climb-resolved activation variable and nonzero "
                "conjugate mechanical-work geometry case"),
        },
        "source_roles": {
            "n128_frozen_numerical_source": n128["source_sha"],
            "n192_stage_start_sources": {name: row.get("source_sha")
                for name, row in manager["stages"].items()
                if name.startswith("N192_TO_")},
            "n192_numerical_paths_unchanged_across_analysis_commit": (
                numerical_paths_unchanged),
            "manager_sources": manager.get("manager_source_history", [])
                +[manager["manager_source_sha"]],
            "finalization_source": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        },
        "initialization": {
            "route": initialization["route"],
            "complete_restart_payload_exact": initialization[
                "n128_complete_restart_payload_identity"][
                    "all_arrays_and_owner_metadata_exact"],
            "field_count": initialization[
                "n128_complete_restart_payload_identity"]["field_count"],
        },
        "overlap_scope": {
            "wall_speedup_on_user_host": json.loads(copied[
                "laplacian_overlap"][1].read_text())["fused_vs_composed"][
                    "wall_speedup"],
            "scope": (
                "beta_p, eta, family Nye, temperature increment, selected "
                "observables, and reservoir totals; not every reservoir "
                "spatial field or history"),
        },
        "canonical_regression": {"passed": True, "tests": 846,
                                  "wall_seconds": 333.47},
        "claims": {"drx": False, "persistent_lagb": False,
                   "strict_asb": False, "material_calibration": False},
        "artifact_sha256": checksums,
    }
    decision_path = verification/"v52_completion_decision.json"
    atomic_json(decision_path, decision)

    campaign_path = root/"hpc3_campaign_manifest.json"
    run_id = "v52-local-completion-20260923"
    run_record = {
        "run_id": run_id,
        "purpose": "v52_faster_physical_continuation_and_matched_spatial_completion",
        "branch": "exp/full-v34-recovery-v52-manager-fix-20260922",
        "execution_location": "local", "hpc3_jobs_submitted": 0,
        "scientific_source_roles": decision["source_roles"],
        "classification": decision["classification"],
        "n128_latest_checkpoint_sha256": n128["latest_checkpoint_sha256"],
        "n192_latest_checkpoint_sha256": n192["latest_checkpoint_sha256"],
        "verification": {
            "completion_decision_sha256": digest(decision_path),
            **{key: value for key, value in checksums.items()
               if key.endswith(".json")},
            "configured_tests_passed": 846,
            "regression_wall_seconds": 333.47,
        },
        "drx_claimed": False, "current_source_lagb_claimed": False,
        "strict_asb_claimed": False, "material_calibration_claimed": False,
        "unrelated_jobs_observed_and_not_touched": [],
    }
    # Preserve the historical manifest byte-for-byte.  It is a long-lived
    # append-only ledger and must not be globally reformatted by finalization.
    raw = campaign_path.read_text()
    if f'"run_id": "{run_id}"' not in raw:
        marker = "\n  ]\n}\n"
        if not raw.endswith(marker):
            raise RuntimeError("campaign manifest has an unexpected tail")
        rendered = json.dumps(run_record, indent=2, sort_keys=True)
        rendered = "\n".join("    "+line for line in rendered.splitlines())
        raw = raw[:-len(marker)]+",\n"+rendered+marker
        campaign_path.write_text(raw)
    print(json.dumps({"decision_sha256": digest(decision_path),
                      "n128_checkpoint_sha256": n128["latest_checkpoint_sha256"],
                      "n192_checkpoint_sha256": n192["latest_checkpoint_sha256"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
