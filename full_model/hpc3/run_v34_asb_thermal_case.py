#!/usr/bin/env python3
"""Run one V34 thermal causal branch from a verified common checkpoint."""

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
    ("full_law_local_adiabatic", "none", "auto", 0.0, 0.0),
    ("frozen_flow_T_local_adiabatic", "freeze_flow", "auto", 0.0, 0.0),
    ("frozen_recovery_T_local_adiabatic", "freeze_recovery", "auto", 0.0, 0.0),
    ("exact_prescribed_T_thermostat", "none", "exact_prescribed_temperature", 0.0, 0.0),
    ("finite_conduction_periodic_insulated", "none", "auto", 0.15, 0.0),
)


def case_definition(case_id: int) -> dict[str, object]:
    if case_id < 0 or case_id >= len(CASES):
        raise ValueError(f"case_id must be in [0,{len(CASES)-1}]")
    name, ablation, semantics, conductivity, bath = CASES[case_id]
    return {
        "case_id": case_id, "case_name": name,
        "causal_temperature_ablation": ablation,
        "thermal_control_semantics": semantics,
        "conductivity_W_m_K": conductivity,
        "bath_coupling_W_m3_K": bath,
    }


def checkpoint_step(path: Path) -> int:
    with np.load(path, allow_pickle=True) as data:
        return int(data["step"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", type=int, required=True, choices=range(len(CASES)))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--shared-checkpoint", type=Path, required=True)
    parser.add_argument("--shared-checkpoint-sha256", required=True)
    parser.add_argument("--target-step", type=int, default=2500)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    case = case_definition(args.case_id)
    source = args.source_root.resolve()
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=source, text=True).strip()
    if actual != args.expected_source_sha or dirty:
        raise RuntimeError(f"branch source mismatch/dirty: {actual}, {bool(dirty)}")
    shared = args.shared_checkpoint.resolve()
    actual_digest = hashlib.sha256(shared.read_bytes()).hexdigest()
    if actual_digest != args.shared_checkpoint_sha256:
        raise RuntimeError("shared checkpoint checksum mismatch")
    if checkpoint_step(shared) != 100:
        raise RuntimeError("V34 branches require the declared step-100 checkpoint")

    output = (args.run_root/str(case["case_name"])).resolve()
    output.mkdir(parents=True, exist_ok=True)
    own = list(output.glob("drx_v25_restart_*.npz"))
    restart = max(own, key=checkpoint_step) if own else shared
    completed = checkpoint_step(restart)+1
    remaining = max(int(args.target_step)+1-completed, 0)
    if args.preflight:
        remaining = min(remaining, 2)
    if remaining == 0:
        print(f"{case['case_name']}: already complete at step {completed-1}")
        return
    parameters = {
        "v31_asb_common_mura_ledger": True,
        "Nx": 128, "Ny": 128, "poly_n": 1, "nSteps": remaining,
        "T0": 900.0, "edot_app": 3.0e4,
        "dt_base_mode": "strain_increment", "dt_strain_step": 1.0e-4,
        "rho0_mode": "absolute", "rho0_abs": 3.5e17,
        "poly_seed": 43, "v19_noise_seed": 43,
        "v19_one_grain_mode": True,
        "v19_density_noise_fraction": 0.02,
        "v19_signed_noise_fraction": 0.01,
        "v19_mechanical_heterogeneity": "eigenstrain_particle",
        "k_thermal": case["conductivity_W_m_K"],
        "T_bath_coupling": case["bath_coupling_W_m3_K"],
        "thermal_control_semantics": case["thermal_control_semantics"],
        "prescribed_temperature_K": 900.0,
        "causal_temperature_ablation": case["causal_temperature_ablation"],
        "causal_reference_temperature_K": 900.0,
        "use_hazard_nucleation": False, "use_component_relabel": False,
        "disable_nucleation": True,
        "diag_interval": 10, "save_interval": 100, "restart_interval": 100,
        "restart_wallclock_interval_s": 900.0,
        "write_field_npz": False, "save_main_panels": False,
        "save_signed_panels": False, "diag_print_extended": True,
        "restart_file": str(restart), "restart_reset_clock": False,
    }
    env = os.environ.copy()
    env.update(DRX_PARAMS=json.dumps(parameters, separators=(",", ":")),
               DRX_OUTDIR=str(output), MPLBACKEND="Agg", OMP_NUM_THREADS="1")
    driver = source/"full_model/production/drx_full_v34_recovery.py"
    with (output/f"run-from-{completed:06d}.log").open("w") as log:
        process = subprocess.run([sys.executable, driver.name], cwd=driver.parent,
                                 env=env, stdout=log, stderr=subprocess.STDOUT)
    record = case | {
        "schema": "asb-drx/v34/thermal-causal-case/v1",
        "source_commit": actual, "exit_code": process.returncode,
        "shared_checkpoint": str(shared),
        "shared_checkpoint_sha256": actual_digest,
        "restart_file": str(restart), "completed_steps_before_run": completed,
        "requested_steps_this_run": remaining, "target_step": int(args.target_step),
        "preflight": bool(args.preflight),
    }
    (output/"v34_thermal_run_record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True)+"\n")
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
