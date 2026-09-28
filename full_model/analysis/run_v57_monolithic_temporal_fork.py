#!/usr/bin/env python3
"""Run and reduce one-full versus two-half monolithic V55 intervals."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


SCHEMA = "asb-drx/v57/monolithic-temporal-fork/v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint_step(path):
    with np.load(path, allow_pickle=True) as raw:
        return int(raw["step"])


def run_branch(source, checkpoint, output, strain_increment, steps):
    output.mkdir(parents=True, exist_ok=False)
    with np.load(checkpoint, allow_pickle=True) as raw:
        parameters = json.loads(str(raw["P_json"].item()))
    parameters.update(
        nSteps=int(steps), dt_base_mode="strain_increment",
        dt_strain_step=float(strain_increment), restart_file=str(checkpoint),
        restart_reset_clock=False, restart_interval=1, save_interval=1,
        diag_interval=1, restart_wallclock_interval_s=900.0,
        write_field_npz=False, save_main_panels=False, save_signed_panels=False,
        disable_plots_on_save_error=True)
    env = os.environ.copy()
    env.update(DRX_PARAMS=json.dumps(parameters, separators=(",", ":")),
               DRX_OUTDIR=str(output), MPLBACKEND="Agg", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    driver = source/"full_model/production/drx_full_v34_recovery.py"
    with (output/"run.log").open("w") as log:
        result = subprocess.run(
            [sys.executable, driver.name], cwd=driver.parent, env=env,
            stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"temporal branch failed: {output}")
    paths = sorted(output.glob("drx_v25_restart_*.npz"), key=checkpoint_step)
    if not paths:
        raise RuntimeError(f"temporal branch produced no checkpoint: {output}")
    return paths[-1]


def ledger(raw, name):
    return json.loads(str(raw[name].item()))


def scalar_observables(raw, initial_fraction):
    front = ledger(raw, "coupled_front_metadata_json")["ledger"]
    common = ledger(raw, "common_front_metadata_json")["ledger"]
    energy = ledger(raw, "v30_asb_cumulative_json")
    return {
        "net_transformed_material_fraction": float(
            np.mean(raw["sparse_front__chi"])-initial_fraction),
        "gross_swept_volume_m3": float(
            front["a_to_b_swept_volume_m3"]+front["b_to_a_swept_volume_m3"]),
        "revisit_volume_m3": float(front["revisit_volume_m3"]),
        "processed_line_m": float(common["processed_line_m"]),
        "transmitted_line_m": float(common["transmitted_line_m"]),
        "annihilated_line_m": float(common["annihilated_line_m"]),
        "sink_line_m": float(common["sink_line_m"]),
        "boundary_line_m": float(common["boundary_line_m"]),
        "external_work_J_m3": float(energy["external_work_J_m3"]),
        "deposited_heat_J_m3": float(energy["deposited_heat_J_m3"]),
        "physical_stored_change_J_m3": float(
            energy["physical_stored_change_J_m3"]),
        "stress_Pa": float(raw["sigma_bar"]),
        "temperature_mean_K": float(np.mean(raw["T"])),
        "temperature_peak_K": float(np.max(raw["T"])),
    }


def hard_valid(raw):
    common = ledger(raw, "common_front_metadata_json")["ledger"]
    mura = ledger(raw, "v21_balance_ledger_json")
    energy = ledger(raw, "v30_asb_cumulative_json")
    scale = max(abs(float(energy["external_work_J_m3"])),
                abs(float(energy["deposited_heat_J_m3"])),
                abs(float(energy["physical_stored_change_J_m3"])), 1.0)
    energy_ok = (abs(float(energy["first_law_residual_J_m3"]))/scale < 1e-8
                 or abs(float(energy["first_law_residual_J_m3"])) < 1e-6)
    return bool(common["maximum_abs_first_law_residual_J"] < 1e-20
                and mura["maximum_relative_burgers_rate_residual"] < 1e-10
                and mura["maximum_relative_line_balance_residual"] < 1e-10
                and mura["maximum_relative_energy_balance_residual"] < 1e-10
                and energy_ok)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=source, text=True).strip()
    if dirty or head != args.expected_source_sha:
        raise RuntimeError(f"source identity mismatch: head={head}, dirty={bool(dirty)}")
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True)
    with np.load(args.checkpoint, allow_pickle=True) as initial:
        params = json.loads(str(initial["P_json"].item()))
        base_increment = float(params["dt_strain_step"])
        initial_fraction = float(np.mean(initial["sibm_initial_child_fraction"]))
    full_path = run_branch(
        source, args.checkpoint, args.output_root/"one_full_interval",
        base_increment, 1)
    half_path = run_branch(
        source, args.checkpoint, args.output_root/"two_half_intervals",
        .5*base_increment, 2)
    with np.load(args.checkpoint, allow_pickle=True) as initial, np.load(
            full_path, allow_pickle=True) as full, np.load(
            half_path, allow_pickle=True) as half:
        initial_scalar = scalar_observables(initial, initial_fraction)
        full_scalar = scalar_observables(full, initial_fraction)
        half_scalar = scalar_observables(half, initial_fraction)
        scalar_errors = {}
        for name in initial_scalar:
            left = full_scalar[name]-initial_scalar[name]
            right = half_scalar[name]-initial_scalar[name]
            scale = max(abs(left), abs(right), 1e-30)
            scalar_errors[name] = {
                "one_full_increment": left, "two_half_increment": right,
                "absolute_difference": abs(left-right),
                "relative_increment_error": abs(left-right)/scale,
            }
        array_errors = {}
        names = (
            "sparse_front__chi", "sparse_front__processed_max", "gamma_slip",
            "v20_beta_p", "v20_family_nye_m1", "T", "rho_GB",
            "common_front__parent__forest_plus_m2",
            "common_front__child__forest_plus_m2",
            "common_front__parent__wall_plus_m2",
            "common_front__child__wall_plus_m2",
        )
        for name in names:
            left = np.asarray(full[name], dtype=float)-np.asarray(
                initial[name], dtype=float)
            right = np.asarray(half[name], dtype=float)-np.asarray(
                initial[name], dtype=float)
            difference = left-right
            scale = max(float(np.linalg.norm(left.ravel())),
                        float(np.linalg.norm(right.ravel())), 1e-30)
            array_errors[name] = {
                "increment_l2_relative_error": float(
                    np.linalg.norm(difference.ravel())/scale),
                "increment_linf_absolute_error": float(
                    np.max(np.abs(difference))),
            }
        common_endpoint = bool(
            np.isclose(float(full["sim_time"]), float(half["sim_time"]),
                       rtol=1e-13, atol=1e-30)
            and np.allclose(full["E_tot"], half["E_tot"],
                            rtol=1e-13, atol=1e-15))
        maximum_error = max(
            [item["relative_increment_error"] for item in scalar_errors.values()]
            +[item["increment_l2_relative_error"]
              for item in array_errors.values()])
        validity = bool(hard_valid(full) and hard_valid(half))
        qualified = bool(common_endpoint and validity and maximum_error <= .05)
        result = {
            "schema": SCHEMA, "source_commit": head,
            "source_checkpoint": str(args.checkpoint.resolve()),
            "source_checkpoint_sha256": digest(args.checkpoint),
            "one_full_checkpoint": str(full_path.resolve()),
            "one_full_checkpoint_sha256": digest(full_path),
            "two_half_checkpoint": str(half_path.resolve()),
            "two_half_checkpoint_sha256": digest(half_path),
            "base_strain_increment": base_increment,
            "common_endpoint": common_endpoint, "hard_valid": validity,
            "maximum_selected_increment_relative_error": maximum_error,
            "provisional_relative_tolerance": .05,
            "qualified": qualified, "scalar_increment_errors": scalar_errors,
            "array_increment_errors": array_errors,
            "one_full_physical_time_s": float(full["sim_time"]),
            "two_half_physical_time_s": float(half["sim_time"]),
            "one_full_applied_tensor": np.asarray(full["E_tot"]).tolist(),
            "two_half_applied_tensor": np.asarray(half["E_tot"]).tolist(),
        }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({key: result[key] for key in (
        "common_endpoint", "hard_valid",
        "maximum_selected_increment_relative_error", "qualified")}, indent=2))


if __name__ == "__main__":
    main()
