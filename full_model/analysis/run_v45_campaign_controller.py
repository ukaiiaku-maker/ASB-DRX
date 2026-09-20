#!/usr/bin/env python3
"""Rebuild the durable V45 branch-state controller from retained evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


VERIFY = Path("full_model/verification")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name):
    path = VERIFY/name
    return (json.loads(path.read_text()), path) if path.is_file() else (None, path)


def main():
    ordering, ordering_path = load("v45_ordering_qualification.json")
    resumed, resumed_path = load("v45_resumed_n128_endpoint.json")
    retained_accuracy, retained_accuracy_path = load("v45_matched_accuracy.json")
    refined, refined_path = load("v45_refined_current_accuracy.json")
    geometry, geometry_path = load("v45_dynamic_geometry_measure.json")
    ordered, ordered_path = load("v45_ordered_residual_sensitivity.json")
    spectrum, spectrum_path = load("v45_spatial_spectrum.json")
    accuracy_state = (
        "COMPLETE" if refined is not None else
        "PARTIAL_RESTARTABLE" if resumed is not None else "READY")
    branches = {
        "ordering_solver": {
            "state": "PASSED" if ordering else "READY",
            "classification": (
                "MATRIX_FREE_TRANSIENT_AND_MEASURED_ASYMPTOTIC_OVERLAP"
                if ordering else None),
        },
        "retained_n128_completion": {
            "state": "COMPLETE" if resumed else "READY",
            "classification": (
                "EXACT_POST_FRONT_CONTINUATION_COMPLETE_NO_DOUBLE_PUBLICATION"
                if resumed else None),
        },
        "matched_accuracy": {
            "state": accuracy_state,
            "classification": (None if refined is None else refined[
                "classification"]),
            "spatial_error_localization": (
                None if spectrum is None else
                "LOW_MODES_PASS_ERROR_GROWS_WITH_WAVE_INDEX"),
            "retained_cohort_numerical_passed": (None if retained_accuracy is None
                else bool(retained_accuracy["classification"][
                    "n128_temporal_threshold_passed"] and
                    retained_accuracy["classification"][
                    "hybrid_source_spatial_threshold_passed"])),
        },
        "dynamic_geometry": {
            "state": "FAILED_NUMERICAL_SCOPED" if geometry and not geometry[
                "classification"]["continuum_coupled_energy_grid_converged"]
                else "PASSED" if geometry else "READY",
            "classification": (None if geometry is None else geometry[
                "classification"]),
        },
        "ordered_residual": {
            "state": "PASSED_DIAGNOSTIC_NOT_PROMOTED" if ordered else "READY",
            "classification": (None if ordered is None else ordered[
                "spatial_observation"]["classification"]),
        },
        "original_horizon_continuation": {
            "state": (
                "READY" if refined and refined["temporal_accuracy_passed"] and
                refined["spatial_accuracy_passed"] else
                "FAILED_NUMERICAL_DEPENDENCY" if refined else "WAITING_DEPENDENCY"),
            "classification": (
                "not launched unless selected temporal and spatial accuracy pass"),
        },
    }
    evidence = {}
    for label, path in (
            ("ordering", ordering_path), ("resumed", resumed_path),
            ("retained_accuracy", retained_accuracy_path),
            ("refined_accuracy", refined_path), ("geometry", geometry_path),
            ("ordered_residual", ordered_path),
            ("spatial_spectrum", spectrum_path)):
        if path.is_file():
            evidence[label] = {"path": str(path.resolve()), "sha256": digest(path)}
    payload = {
        "schema": "asb-drx/v45/campaign-controller/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "execution_location": "local", "branches": branches,
        "evidence": evidence,
        "fixture_passed": bool(ordering and resumed and geometry),
        "scientific_gate_passed": bool(
            refined and refined["temporal_accuracy_passed"] and
            refined["spatial_accuracy_passed"] and geometry and geometry[
                "classification"]["continuum_coupled_energy_grid_converged"]),
        "drx_claimed": False, "strict_asb_claimed": False,
        "material_calibration_claimed": False,
        "next_exact_task": (
            "regularize or resolve the high-wave-number Mura wall content and "
            "separate geometry-owned line energy from singular continuum "
            "gradient energy before any longer physical horizon"),
    }
    output = VERIFY/"v45_campaign_controller.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps({
        "sha256": digest(output),
        "matched_accuracy_state": accuracy_state,
        "scientific_gate_passed": payload["scientific_gate_passed"]},
        sort_keys=True))


if __name__ == "__main__":
    main()
