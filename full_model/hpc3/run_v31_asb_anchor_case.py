#!/usr/bin/env python3
"""Run or resume one fault-isolated V31 resolved-ASB anchor case."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


CASES = {
    0: ("heterogeneous_adiabatic", True, "adiabatic"),
    1: ("heterogeneous_isothermal", True, "isothermal"),
    2: ("homogeneous_adiabatic", False, "adiabatic"),
    3: ("homogeneous_isothermal", False, "isothermal"),
}


def checkpoint_step(path: Path) -> int:
    with np.load(path, allow_pickle=True) as data:
        return int(data["step"])


def inventory(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "output_inventory.sha256.json":
            result[str(path.relative_to(root))] = hashlib.sha256(
                path.read_bytes()).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", type=int, required=True, choices=CASES)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--target-steps", type=int, default=5000)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()

    name, heterogeneous, thermal = CASES[args.case_id]
    source = args.source_root.resolve()
    actual_source_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if actual_source_sha != args.expected_source_sha:
        raise RuntimeError(
            f"source SHA mismatch: expected {args.expected_source_sha}, "
            f"found {actual_source_sha}")
    output = (args.run_root/name).resolve()
    output.mkdir(parents=True, exist_ok=True)
    driver = source/"full_model/production/drx_full_v34_recovery.py"
    checkpoints = sorted(output.glob("drx_v25_restart_*.npz"))
    restart = max(checkpoints, key=checkpoint_step) if checkpoints else None
    completed = checkpoint_step(restart)+1 if restart else 0
    remaining = max(int(args.target_steps)-completed, 0)
    if remaining == 0:
        (output/"output_inventory.sha256.json").write_text(
            json.dumps(inventory(output), indent=2, sort_keys=True)+"\n")
        print(f"{name}: already complete at {completed} steps")
        return

    isothermal = thermal == "isothermal"
    parameters = {
        "v31_asb_common_mura_ledger": True,
        "Nx": 128, "Ny": 128, "poly_n": 1,
        "nSteps": remaining,
        "T0": 1100.0, "edot_app": 3.0e4,
        "dt_base_mode": "strain_increment", "dt_strain_step": 1.0e-4,
        "rho0_mode": "absolute", "rho0_abs": 3.5e17,
        "poly_seed": 43, "v19_noise_seed": 43,
        "v19_one_grain_mode": True,
        "v19_density_noise_fraction": 0.02 if heterogeneous else 0.0,
        "v19_signed_noise_fraction": 0.01 if heterogeneous else 0.0,
        "v19_mechanical_heterogeneity": (
            "eigenstrain_particle" if heterogeneous else "none"),
        "k_thermal": 0.15 if isothermal else 0.0,
        "T_bath_coupling": 2.0e10 if isothermal else 0.0,
        "use_hazard_nucleation": False,
        "use_component_relabel": False,
        "disable_nucleation": True,
        "diag_interval": 10,
        "save_interval": 100,
        "restart_interval": 100,
        "restart_wallclock_interval_s": 900.0,
        "write_field_npz": False,
        "save_main_panels": False,
        "save_signed_panels": False,
        "diag_print_extended": True,
        "restart_file": str(restart) if restart else None,
        "restart_reset_clock": False if restart else True,
    }
    environment = os.environ.copy()
    environment.update(
        DRX_PARAMS=json.dumps(parameters, separators=(",", ":")),
        DRX_OUTDIR=str(output), MPLBACKEND="Agg", OMP_NUM_THREADS="1")
    command = [sys.executable, driver.name]
    log_name = f"run-from-{completed:06d}.log"
    with (output/log_name).open("w") as log:
        process = subprocess.run(
            command, cwd=driver.parent, env=environment,
            stdout=log, stderr=subprocess.STDOUT)
    record = {
        "case_id": args.case_id, "case_name": name,
        "heterogeneous": heterogeneous, "thermal_control": thermal,
        "source_root": str(source), "source_commit": actual_source_sha,
        "restart_file": str(restart) if restart else None,
        "completed_steps_before_run": completed,
        "requested_steps_this_run": remaining,
        "target_steps": int(args.target_steps),
        "preflight": bool(args.preflight), "exit_code": process.returncode,
    }
    (output/"v31_anchor_run_record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True)+"\n")
    (output/"output_inventory.sha256.json").write_text(
        json.dumps(inventory(output), indent=2, sort_keys=True)+"\n")
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
