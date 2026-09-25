#!/usr/bin/env python3
"""Run a restartable common-state ASB plus prepared-boundary DRX case."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_step(path: Path) -> int:
    with np.load(path, allow_pickle=True) as data:
        return int(data["step"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", type=int, required=True)
    parser.add_argument("--case-table", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--target-step", type=int, required=True)
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    cases = json.loads(args.case_table.read_text())
    case = cases[args.case_id]
    source = args.source_root.resolve()
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=source, text=True).strip()
    if dirty or head != args.expected_source_sha:
        raise RuntimeError(f"source identity mismatch: head={head}, dirty={bool(dirty)}")
    output = (args.run_root / case["id"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = list(output.glob("drx_v25_restart_*.npz"))
    restart = max(checkpoints, key=checkpoint_step) if checkpoints else None
    completed = checkpoint_step(restart) + 1 if restart else 0
    target = min(args.target_step, completed + 1) if args.preflight else args.target_step
    remaining = max(target + 1 - completed, 0)
    if remaining == 0:
        print(json.dumps({"case": case["id"], "already_complete": True,
                          "completed_step": completed - 1}, sort_keys=True))
        return
    interval = 1 if args.preflight else 50
    parameters = {
        "Nx": args.grid, "Ny": args.grid, "grain_max": 8, "poly_n": 8,
        "nSteps": remaining,
        "T0": float(case["T0_K"]),
        "edot_app": float(case["strain_rate_s"]),
        "dt_base_mode": "strain_increment", "dt_strain_step": 1.0e-4,
        "finite_loading_init_from_inverse": True,
        "rho0_mode": "absolute", "rho0_abs": 3.5e17,
        "poly_seed": 43, "v19_noise_seed": 43,
        "v19_density_noise_fraction": 0.02,
        "v19_signed_noise_fraction": 0.01,
        "v19_mechanical_heterogeneity": "eigenstrain_particle",
        "v19_particle_radius_um": float(case["particle_radius_um"]),
        "k_thermal": float(case["conductivity_W_m_K"]),
        "T_bath_coupling": 0.0,
        "thermal_control_semantics": "auto",
        "causal_temperature_ablation": str(
            case.get("causal_temperature_ablation", "none")),
        "v34_authoritative_common_temperature_routing": True,
        "v31_asb_common_mura_ledger": True,
        "v32_existing_boundary_common_state": True,
        "v33_common_mura_evolution_enabled": True,
        "v33_common_temperature_evolution_enabled": True,
        "sibm_clean_bicrystal_initialize": True,
        "sibm_clean_parent_density_m2": 4.0e17,
        "sibm_clean_child_density_m2": 1.0e17,
        "use_sparse_common_front_state": True,
        "use_sibm_existing_boundary": True,
        "use_rho_state_partition": True,
        "stored_energy_coupling_mode": "common_variational",
        "sibm_parent_label_override": 0,
        "sibm_child_label_override": 1,
        "sibm_initial_bulge_radius_um": 0.0,
        "sibm_pin_endpoints": False,
        "sibm_active_window_radius_um": 4.5,
        "sibm_applied_pressure_Pa": 0.0,
        "sibm_physical_drag_pressure_Pa": 0.0,
        "sibm_mobility_multiplier": (
            1.0 if bool(case.get("front_enabled", True)) else 0.0),
        "sibm_front_operator": "coupled_bidirectional_v30",
        "sibm_legacy_afterburner_reproduction": False,
        "moving_front_attempt_frequency_s": float(
            case["front_attempt_frequency_s"]),
        "disable_nucleation": True,
        "use_hazard_nucleation": False,
        "use_stateful_embryos": False,
        "use_component_relabel": False,
        "diag_interval": interval, "save_interval": interval,
        "restart_interval": interval, "restart_wallclock_interval_s": 900.0,
        "write_field_npz": False, "save_main_panels": False,
        "save_signed_panels": False, "diag_print_extended": True,
        "restart_file": None if restart is None else str(restart),
        "restart_reset_clock": restart is None,
    }
    env = os.environ.copy()
    env.update(DRX_PARAMS=json.dumps(parameters, separators=(",", ":")),
               DRX_OUTDIR=str(output), MPLBACKEND="Agg",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1")
    driver = source / "full_model/production/drx_full_v34_recovery.py"
    log_path = output / f"run-from-{completed:06d}.log"
    with log_path.open("w") as log:
        process = subprocess.run([sys.executable, driver.name], cwd=driver.parent,
                                 env=env, stdout=log,
                                 stderr=subprocess.STDOUT)
    checkpoints = list(output.glob("drx_v25_restart_*.npz"))
    latest = max(checkpoints, key=checkpoint_step) if checkpoints else None
    record = {
        "schema": "asb-drx/v55/integrated-response-run/v1",
        "case": case,
        "source_commit": head,
        "case_table": str(args.case_table.resolve()),
        "case_table_sha256": digest(args.case_table),
        "grid": args.grid,
        "target_step": args.target_step,
        "preflight": args.preflight,
        "completed_steps_before_run": completed,
        "requested_steps_this_run": remaining,
        "exit_code": process.returncode,
        "latest_step": None if latest is None else checkpoint_step(latest),
        "latest_checkpoint": None if latest is None else str(latest),
        "latest_checkpoint_sha256": None if latest is None else digest(latest),
        "fresh_common_state": restart is None,
        "nucleation_disabled": True,
        "direct_label_allocation_disabled": True,
        "front_enabled": bool(case.get("front_enabled", True)),
        "applied_front_pressure_Pa": 0.0,
        "enabled_mechanisms": [
            "common_mura_dislocation_evolution",
            "plastic_flow_and_independent_heat",
            "finite_thermal_conduction",
            "bidirectional_complete-energy_existing-boundary_front"
        ],
    }
    (output / "v55_integrated_run_record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
