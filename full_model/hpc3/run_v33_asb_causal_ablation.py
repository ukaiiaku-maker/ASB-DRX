#!/usr/bin/env python3
"""Run one source-locked V33 temperature-feedback causal ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


CASES = (
    ("broad_T1100_baseline", 1100.0, "none"),
    ("broad_T1100_freeze_flow", 1100.0, "freeze_flow"),
    ("broad_T1100_freeze_recovery", 1100.0, "freeze_recovery"),
    ("localized_T0900_freeze_flow", 900.0, "freeze_flow"),
    ("localized_T0900_freeze_recovery", 900.0, "freeze_recovery"),
)


def case_definition(case_id: int) -> dict[str, object]:
    if case_id < 0 or case_id >= len(CASES):
        raise ValueError(f"case_id must be in [0,{len(CASES)-1}]")
    name, temperature, ablation = CASES[case_id]
    return {
        "case_id": case_id, "case_name": name,
        "temperature_K": temperature, "strain_rate_s-1": 3.0e4,
        "seed": 43, "causal_temperature_ablation": ablation,
        "thermal_operator": "NO_CONDUCTION_LOCAL_ADIABATIC",
    }


def checkpoint_step(path: Path) -> int:
    with np.load(path, allow_pickle=True) as data:
        return int(data["step"])


def inventory(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()
            and path.name != "output_inventory.sha256.json"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", type=int, required=True, choices=range(len(CASES)))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--target-steps", type=int, default=5000)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()

    case = case_definition(args.case_id)
    source = args.source_root.resolve()
    actual_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=source, text=True).strip()
    if actual_sha != args.expected_source_sha or dirty:
        raise RuntimeError(
            f"causal source must be clean at {args.expected_source_sha}; "
            f"found sha={actual_sha}, dirty={bool(dirty)}")

    output = (args.run_root/str(case["case_name"])).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = list(output.glob("drx_v25_restart_*.npz"))
    restart = max(checkpoints, key=checkpoint_step) if checkpoints else None
    completed = checkpoint_step(restart)+1 if restart else 0
    remaining = max(int(args.target_steps)-completed, 0)
    if args.preflight:
        remaining = min(remaining, 2)
    if remaining == 0:
        print(f"{case['case_name']}: already complete")
        return

    parameters = {
        "v31_asb_common_mura_ledger": True,
        "Nx": 128, "Ny": 128, "poly_n": 1, "nSteps": remaining,
        "T0": case["temperature_K"], "edot_app": case["strain_rate_s-1"],
        "dt_base_mode": "strain_increment", "dt_strain_step": 1.0e-4,
        "rho0_mode": "absolute", "rho0_abs": 3.5e17,
        "poly_seed": 43, "v19_noise_seed": 43,
        "v19_one_grain_mode": True,
        "v19_density_noise_fraction": 0.02,
        "v19_signed_noise_fraction": 0.01,
        "v19_mechanical_heterogeneity": "eigenstrain_particle",
        "k_thermal": 0.0, "T_bath_coupling": 0.0,
        "causal_temperature_ablation": case["causal_temperature_ablation"],
        "causal_reference_temperature_K": case["temperature_K"],
        "use_hazard_nucleation": False, "use_component_relabel": False,
        "disable_nucleation": True,
        "diag_interval": 10, "save_interval": 100, "restart_interval": 100,
        "restart_wallclock_interval_s": 900.0,
        "write_field_npz": False, "save_main_panels": False,
        "save_signed_panels": False, "diag_print_extended": True,
        "restart_file": str(restart) if restart else None,
        "restart_reset_clock": False if restart else True,
    }
    driver = source/"full_model/production/drx_full_v34_recovery.py"
    environment = os.environ.copy()
    environment.update(
        DRX_PARAMS=json.dumps(parameters, separators=(",", ":")),
        DRX_OUTDIR=str(output), MPLBACKEND="Agg", OMP_NUM_THREADS="1")
    with (output/f"run-from-{completed:06d}.log").open("w") as log:
        process = subprocess.run(
            [sys.executable, driver.name], cwd=driver.parent,
            env=environment, stdout=log, stderr=subprocess.STDOUT)
    record = case | {
        "source_commit": actual_sha, "source_root": str(source),
        "completed_steps_before_run": completed,
        "requested_steps_this_run": remaining,
        "target_steps": int(args.target_steps), "preflight": bool(args.preflight),
        "restart_file": str(restart) if restart else None,
        "exit_code": process.returncode,
    }
    (output/"v33_causal_run_record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True)+"\n")
    (output/"output_inventory.sha256.json").write_text(
        json.dumps(inventory(output), indent=2, sort_keys=True)+"\n")
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
