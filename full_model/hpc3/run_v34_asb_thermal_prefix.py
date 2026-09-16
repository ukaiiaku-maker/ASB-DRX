#!/usr/bin/env python3
"""Create the common early-valid checkpoint for the V34 thermal matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--target-step", type=int, default=100)
    args = parser.parse_args()
    source = args.source_root.resolve()
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=source, text=True).strip()
    if actual != args.expected_source_sha or dirty:
        raise RuntimeError(f"prefix source mismatch/dirty: {actual}, {bool(dirty)}")
    output = (args.run_root/"shared_prefix_T0900_R3e4_seed43").resolve()
    output.mkdir(parents=True, exist_ok=True)
    parameters = {
        "v31_asb_common_mura_ledger": True,
        "Nx": 128, "Ny": 128, "poly_n": 1,
        "nSteps": int(args.target_step)+1,
        "T0": 900.0, "edot_app": 3.0e4,
        "dt_base_mode": "strain_increment", "dt_strain_step": 1.0e-4,
        "rho0_mode": "absolute", "rho0_abs": 3.5e17,
        "poly_seed": 43, "v19_noise_seed": 43,
        "v19_one_grain_mode": True,
        "v19_density_noise_fraction": 0.02,
        "v19_signed_noise_fraction": 0.01,
        "v19_mechanical_heterogeneity": "eigenstrain_particle",
        "k_thermal": 0.0, "T_bath_coupling": 0.0,
        "thermal_control_semantics": "auto",
        "causal_temperature_ablation": "none",
        "use_hazard_nucleation": False, "use_component_relabel": False,
        "disable_nucleation": True,
        "diag_interval": 10, "save_interval": 100, "restart_interval": 100,
        "restart_wallclock_interval_s": 900.0,
        "write_field_npz": False, "save_main_panels": False,
        "save_signed_panels": False, "diag_print_extended": True,
        "restart_file": None, "restart_reset_clock": True,
    }
    env = os.environ.copy()
    env.update(DRX_PARAMS=json.dumps(parameters, separators=(",", ":")),
               DRX_OUTDIR=str(output), MPLBACKEND="Agg", OMP_NUM_THREADS="1")
    driver = source/"full_model/production/drx_full_v34_recovery.py"
    with (output/"run-prefix.log").open("w") as log:
        process = subprocess.run([sys.executable, driver.name], cwd=driver.parent,
                                 env=env, stdout=log, stderr=subprocess.STDOUT)
    checkpoint = output/f"drx_v25_restart_{args.target_step:06d}.npz"
    if process.returncode == 0 and not checkpoint.exists():
        raise RuntimeError(f"prefix did not produce {checkpoint}")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint.exists() else None
    with np.load(checkpoint, allow_pickle=True) as data:
        step = int(data["step"]); temperature = np.asarray(data["T"], dtype=float)
    record = {
        "schema": "asb-drx/v34/thermal-shared-prefix/v1",
        "source_commit": actual, "exit_code": process.returncode,
        "checkpoint": str(checkpoint), "checkpoint_sha256": digest,
        "step": step, "temperature_min_K": float(temperature.min()),
        "temperature_mean_K": float(temperature.mean()),
        "temperature_max_K": float(temperature.max()),
        "thermal_operator": "NO_CONDUCTION_LOCAL_ADIABATIC",
    }
    (output/"v34_shared_prefix_record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True)+"\n")
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
