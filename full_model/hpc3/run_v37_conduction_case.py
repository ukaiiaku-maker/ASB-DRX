#!/usr/bin/env python3
"""Restartable V37 finite-conduction response-family case."""

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


def terminal_reason(directory: Path) -> str:
    text = "\n".join(path.read_text(errors="replace")[-20000:]
                     for path in sorted(directory.glob("run-from-*.log")))
    if "THERMAL VALIDITY STOP" in text:
        return "THERMAL_MODEL_VALIDITY_BOUNDARY"
    if "MECHANICAL VALIDITY STOP" in text:
        return "MECHANICAL_MODEL_VALIDITY_BOUNDARY"
    return "REQUESTED_HORIZON"


def load_cases(path: Path) -> list[dict[str, object]]:
    cases = json.loads(path.read_text())
    required = {"id", "T0_K", "strain_rate_s", "particle_radius_um",
                "conductivity_W_m_K"}
    if not isinstance(cases, list) or not cases:
        raise ValueError("case table must be a nonempty JSON list")
    if any(not required <= set(case) for case in cases):
        raise ValueError("case table row is missing required fields")
    if len({str(case["id"]) for case in cases}) != len(cases):
        raise ValueError("case ids must be unique")
    if any(float(case["conductivity_W_m_K"]) <= 0.0 for case in cases):
        raise ValueError("V37 physical screen requires positive conductivity")
    return cases


def validate_source_identity(source: Path, expected: str) -> tuple[str, bool]:
    """Allow a configuration-only descendant of the frozen physics source."""
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=source, text=True).strip()
    if dirty:
        raise RuntimeError(f"source dirty: {bool(dirty)}")
    if actual == expected:
        return actual, True
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", expected, actual], cwd=source)
    if ancestor.returncode != 0:
        raise RuntimeError(f"frozen source {expected} is not ancestor of {actual}")
    protected = [
        "full_model/production",
        "full_model/hpc3/v37_conduction_cases.json",
    ]
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", f"{expected}..{actual}", "--", *protected],
        cwd=source, text=True).strip()
    if changed:
        raise RuntimeError(f"protected executable inputs changed:\n{changed}")
    return actual, False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", type=int, required=True)
    parser.add_argument("--case-table", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--target-step", type=int, default=2500)
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    cases = load_cases(args.case_table)
    if args.case_id < 0 or args.case_id >= len(cases):
        raise ValueError("case-id outside case table")
    case = cases[args.case_id]
    source = args.source_root.resolve()
    actual, exact_source = validate_source_identity(
        source, args.expected_source_sha)
    output = (args.run_root/str(case["id"])).resolve()
    output.mkdir(parents=True, exist_ok=True)
    existing = list(output.glob("drx_v25_restart_*.npz"))
    restart = max(existing, key=checkpoint_step) if existing else None
    completed = checkpoint_step(restart)+1 if restart else 0
    target = min(args.target_step, completed+2) if args.preflight else args.target_step
    remaining = max(target+1-completed, 0)
    if remaining == 0:
        print(f"{case['id']}: already complete at step {completed-1}")
        return
    interval = 1 if args.preflight else 100
    parameters = {
        "v31_asb_common_mura_ledger": True,
        "Nx": args.grid, "Ny": args.grid, "poly_n": 1, "nSteps": remaining,
        "T0": float(case["T0_K"]), "edot_app": float(case["strain_rate_s"]),
        "dt_base_mode": "strain_increment", "dt_strain_step": 1.0e-4,
        "rho0_mode": "absolute", "rho0_abs": 3.5e17,
        "poly_seed": 43, "v19_noise_seed": 43,
        "v19_one_grain_mode": True,
        "v19_density_noise_fraction": 0.02,
        "v19_signed_noise_fraction": 0.01,
        "v19_mechanical_heterogeneity": "eigenstrain_particle",
        "v19_particle_radius_um": float(case["particle_radius_um"]),
        "k_thermal": float(case["conductivity_W_m_K"]),
        "T_bath_coupling": 0.0,
        "thermal_control_semantics": "auto",
        "causal_temperature_ablation": "none",
        "v34_authoritative_common_temperature_routing": True,
        "use_hazard_nucleation": False, "use_component_relabel": False,
        "disable_nucleation": True,
        "diag_interval": interval, "save_interval": interval,
        "restart_interval": interval, "restart_wallclock_interval_s": 900.0,
        "write_field_npz": False, "save_main_panels": False,
        "save_signed_panels": False, "diag_print_extended": True,
        "restart_file": None if restart is None else str(restart),
        "restart_reset_clock": restart is None,
    }
    env = os.environ.copy()
    env.update(DRX_PARAMS=json.dumps(parameters, separators=(",", ":")),
               DRX_OUTDIR=str(output), MPLBACKEND="Agg", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    driver = source/"full_model/production/drx_full_v34_recovery.py"
    log_path = output/f"run-from-{completed:06d}.log"
    with log_path.open("w") as log:
        process = subprocess.run([sys.executable, driver.name], cwd=driver.parent,
                                 env=env, stdout=log, stderr=subprocess.STDOUT)
    checkpoints = list(output.glob("drx_v25_restart_*.npz"))
    latest = max(checkpoints, key=checkpoint_step) if checkpoints else None
    latest_step = checkpoint_step(latest) if latest else None
    record = {
        "schema": "asb-drx/v37/conduction-case/v1",
        "case_index": args.case_id, "case": case,
        "archive_commit": actual, "production_source_commit": args.expected_source_sha,
        "exact_source_head": exact_source, "source_dirty": False,
        "case_table": str(args.case_table.resolve()),
        "case_table_sha256": digest(args.case_table),
        "grid": args.grid, "target_step": args.target_step,
        "preflight": args.preflight, "exit_code": process.returncode,
        "completed_steps_before_run": completed,
        "requested_steps_this_run": remaining,
        "latest_step": latest_step,
        "latest_checkpoint": None if latest is None else str(latest),
        "latest_checkpoint_sha256": None if latest is None else digest(latest),
        "terminal_reason": terminal_reason(output),
        "terminal": bool(latest_step == args.target_step or (
            latest is not None and terminal_reason(output) != "REQUESTED_HORIZON")),
        "strict_asb_claimed": False,
    }
    (output/"v37_conduction_run_record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True)+"\n")
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
