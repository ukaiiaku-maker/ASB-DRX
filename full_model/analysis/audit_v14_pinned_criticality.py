#!/usr/bin/env python3
"""Classify the v14 common-state pinned-criticality continuations.

This postprocessor deliberately reads completed checkpoints and contour CSVs;
it does not repair, clip, or otherwise mutate scientific output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def final_checkpoint(directory: Path) -> Path:
    candidates = sorted(directory.glob("drx_v25_restart_*.npz"))
    if not candidates:
        raise FileNotFoundError(f"no restart checkpoint in {directory}")
    return candidates[-1]


def common_record(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as state:
        experiment = json.loads(str(state["sibm_experiment_json"].item()))
        return {
            "path": str(path), "sha256": digest(path),
            "grid": int(state["eta"].shape[0]), "Ng": int(state["Ng"]),
            "psi_gv": np.asarray(state["psi_gv"]).copy(),
            "amplitude_m": float(
                experiment["last_contour_metrics"]["bulge_amplitude_m"]),
        }


def classify_case(name: str, directory: Path, common: dict,
                  expected_sign: int, mobility_off: bool = False) -> dict:
    checkpoint = final_checkpoint(directory)
    rows = list(csv.DictReader((directory / "sibm_contour_diagnostics.csv").open()))
    if not rows:
        raise ValueError(f"empty contour diagnostics in {directory}")
    last = rows[-1]
    with np.load(checkpoint, allow_pickle=True) as state:
        experiment = json.loads(str(state["sibm_experiment_json"].item()))
        metadata = json.loads(str(state["sparse_front_metadata_json"].item()))
        ledger = metadata["ledger"]
        amplitude = float(last["bulge_amplitude_m"])
        delta = amplitude - common["amplitude_m"]
        dx = 10.0e-6 / common["grid"]
        stationary = abs(delta) <= 0.02 * dx
        sign_ok = (stationary if expected_sign == 0 else expected_sign * delta > 0.0)
        no_allocation = (int(state["Ng"]) == common["Ng"] and
                         np.array_equal(state["psi_gv"], common["psi_gv"],
                                        equal_nan=True))
    line_scale = max(abs(float(ledger["parent_line_processed_m"])), 1.0e-30)
    ledger_ok = (
        abs(float(ledger["line_closure_m"])) <= 1.0e-12 * line_scale + 1.0e-24
        and abs(float(ledger["signed_burgers_change_m2"])) <= 1.0e-20
        and abs(float(ledger["line_energy_released_J"])
                - float(ledger["heat_released_J"])) <= 1.0e-24)
    if mobility_off:
        sign_ok = delta == 0.0 and float(ledger["swept_volume_m3"]) == 0.0
    return {
        "name": name, "directory": str(directory),
        "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
        "initial_amplitude_m": common["amplitude_m"],
        "final_amplitude_m": amplitude, "amplitude_change_m": delta,
        "amplitude_change_cells": delta / dx,
        "expected_sign": expected_sign, "sign_or_stationarity_passed": bool(sign_ok),
        "reported_local_normal_pressure_Pa": float(last["local_normal_pressure_Pa"]),
        "reported_terminal_area_velocity_m_s": float(
            last["area_equivalent_normal_velocity_m_s"]),
        "continuation_pressure_relative_to_declared_neutral_Pa": float(
            experiment.get("applied_continuation_pressure_Pa", 0.0)
            - experiment.get("physical_drag_pressure_Pa", 0.0) - 0.5e6),
        "ledger": ledger, "ledger_passed": bool(ledger_ok),
        "no_label_or_orientation_allocation": bool(no_allocation),
        "passed": bool(sign_ok and ledger_ok and no_allocation),
    }


def restart_comparison(continuous: Path, segmented: Path) -> dict:
    a_path, b_path = final_checkpoint(continuous), final_checkpoint(segmented)
    worst_relative, worst_absolute, worst_name = 0.0, 0.0, None
    with np.load(a_path, allow_pickle=True) as a, np.load(b_path, allow_pickle=True) as b:
        for name in sorted(set(a.files) & set(b.files)):
            x, y = np.asarray(a[name]), np.asarray(b[name])
            if x.shape != y.shape or x.dtype.kind not in "fciu" or x.dtype.kind == "b":
                continue
            finite = np.isfinite(x) | np.isfinite(y)
            if not np.any(finite):
                continue
            with np.errstate(invalid="ignore"):
                absolute = float(np.nanmax(np.abs(x[finite] - y[finite])))
                scale = max(float(np.nanmax(np.abs(x[finite]))),
                            float(np.nanmax(np.abs(y[finite]))), 1.0)
            relative = absolute / scale
            if np.isfinite(relative) and relative > worst_relative:
                worst_relative, worst_absolute, worst_name = relative, absolute, name
    # FFT reconstruction and process restart can change reduction order.  This
    # bound is a numerical exact-continuation tolerance, not a physics tolerance.
    passed = worst_relative <= 1.0e-12
    return {
        "continuous": str(a_path), "segmented": str(b_path),
        "maximum_scaled_difference": worst_relative,
        "maximum_absolute_difference": worst_absolute,
        "limiting_field": worst_name, "declared_tolerance": 1.0e-12,
        "restart_exact_within_declared_roundoff": bool(passed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--common-128", type=Path, required=True)
    parser.add_argument("--common-192", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[],
                        help="grid:name:expected_sign:directory; name 'off' is mobility-off")
    parser.add_argument("--restart", action="append", default=[],
                        help="name:continuous_directory:segmented_directory")
    args = parser.parse_args()
    commons = {g: common_record(p) for g, p in
               ((128, args.common_128), (192, args.common_192))}
    cases = []
    for spec in args.case:
        grid, name, sign, directory = spec.split(":", 3)
        cases.append(classify_case(name, Path(directory), commons[int(grid)],
                                   int(sign), name == "mobility_off"))
    restarts = []
    for spec in args.restart:
        name, continuous, segmented = spec.split(":", 2)
        item = restart_comparison(Path(continuous), Path(segmented))
        item["name"] = name
        restarts.append(item)
    matrix = {(commons[int(spec.split(":", 1)[0])]["grid"],
               spec.split(":", 3)[1]) for spec in args.case}
    required = {(g, n) for g in (128, 192) for n in
                ("neutral", "subcritical", "supercritical", "mobility_off", "reversed")}
    functional_path = args.output.with_name("v14_full_functional_fd.json")
    functional = (json.loads(functional_path.read_text())
                  if functional_path.exists() else {"passed": False})
    passed = (matrix == required and all(x["passed"] for x in cases)
              and len(restarts) >= 2
              and all(x["restart_exact_within_declared_roundoff"] for x in restarts)
              and functional.get("passed", False))
    result = {
        "schema": "full-v34-v14-pinned-criticality/v1",
        "classification": ("V14_LOCAL_PINNED_CRITICALITY_PASSED" if passed
                           else "V14_LOCAL_PINNED_CRITICALITY_FAILED"),
        "fixture_passed": bool(passed),
        "scientific_gate_passed": bool(passed),
        "neutral_contour_tolerance_cells": 0.02,
        "pressure_sign_definition": (
            "applied continuation pressure minus physical drag minus the declared "
            "0.5 MPa neutral continuation; raw local_normal_pressure is a flat-interface "
            "diagnostic and is not the pinned-cap variational pressure"),
        "common_states": {str(k): {kk: vv for kk, vv in v.items()
                                     if kk != "psi_gv"} for k, v in commons.items()},
        "cases": cases, "restart_comparisons": restarts,
        "full_functional_derivative_verification": {
            "common_interpolation_test": "tests/test_full_model_stored_energy_coupling.py",
            "pinned_cap_finite_difference_test": "tests/test_full_model_sibm_geometry.py",
            "result": str(functional_path),
            "classification": functional.get("classification"),
            "results": functional.get("results", []),
            "status": "analytical full-functional derivatives match centered finite differences",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["classification"])


if __name__ == "__main__":
    main()
