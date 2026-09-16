#!/usr/bin/env python3
"""Run the 32/64 production-entry gates for the V30 coupled front."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = ROOT/"full_model"/"production"


def parameters(n, rho_a, rho_b, mobility):
    return {
        "Nx": n, "Ny": n, "grain_max": 8, "nSteps": 3,
        "edot_app": .001, "dt_strain_step": 1e-10,
        "sibm_clean_bicrystal_initialize": True,
        "sibm_clean_parent_density_m2": rho_a,
        "sibm_clean_child_density_m2": rho_b,
        "use_sparse_common_front_state": True,
        "use_sibm_existing_boundary": True,
        "stored_energy_coupling_mode": "common_variational",
        "sibm_parent_label_override": 0, "sibm_child_label_override": 1,
        "sibm_initial_bulge_radius_um": 0.0,
        "sibm_pin_endpoints": False, "sibm_active_window_radius_um": 4.5,
        "sibm_mobility_multiplier": mobility,
        "sibm_front_operator": "coupled_bidirectional_v30",
        "sibm_legacy_afterburner_reproduction": False,
        # S3 freezes unrelated constitutive/transport channels while retaining
        # conservative transfer, boundary storage, cleanup, and heat.
        "sibm_sequential_stage": "S3",
        "disable_nucleation": True, "use_hazard_nucleation": False,
        "use_component_relabel": False, "diag_interval": 1,
        "save_interval": 100000, "restart_interval": 2,
        "plot_interval": 100000, "write_field_npz": False,
        "save_main_panels": False, "save_signed_panels": False,
    }


def run_case(root, grid, name, rho_a, rho_b, mobility):
    directory = (root/f"n{grid}"/name).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    config = parameters(grid, rho_a, rho_b, mobility)
    environment = dict(
        os.environ, DRX_OUTDIR=str(directory), MPLBACKEND="Agg",
        OMP_NUM_THREADS="1", DRX_PARAMS=json.dumps(config, separators=(",", ":")))
    with (directory/"stdout.log").open("w") as stdout, \
            (directory/"stderr.log").open("w") as stderr:
        completed = subprocess.run(
            [sys.executable, "drx_full_v34_recovery.py"], cwd=PRODUCTION,
            env=environment, stdout=stdout, stderr=stderr, text=True)
    if completed.returncode:
        return {"grid": grid, "case": name, "completed": False,
                "returncode": completed.returncode,
                "stderr": (directory/"stderr.log").read_text()[-4000:]}
    checkpoint = sorted(directory.glob("drx_v25_restart_*.npz"))[-1]
    with np.load(checkpoint, allow_pickle=True) as data:
        coupled = json.loads(str(data["coupled_front_metadata_json"].item()))
        experiment = json.loads(str(data["sibm_experiment_json"].item()))
    ledger = coupled["ledger"]
    return {
        "grid": grid, "case": name, "completed": True,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "schema": coupled["schema"],
        "attempts": ledger["attempts"], "accepted": ledger["accepted"],
        "rejected_direction": ledger["rejected_direction"],
        "stationary_trials": ledger["stationary_trials"],
        "a_to_b_swept_volume_m3": ledger["a_to_b_swept_volume_m3"],
        "b_to_a_swept_volume_m3": ledger["b_to_a_swept_volume_m3"],
        "maximum_abs_line_closure_m": ledger["maximum_abs_line_closure_m"],
        "maximum_abs_signed_closure_m2": ledger[
            "maximum_abs_signed_closure_m2"],
        "heat_J": ledger["heat_J"],
        "last_detailed_balance_log_residual": experiment[
            "front_last_decision"]["detailed_balance_log_residual"],
        "old_afterburner_called": bool(
            experiment.get("legacy_afterburner_calls", 0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=ROOT/"results-local/v30-front-local-gates")
    parser.add_argument("--output", type=Path, default=ROOT/
                        "full_model/verification/v30_front_production_local.json")
    args = parser.parse_args()
    cases = {
        "equal": (2.5e17, 2.5e17, 1.0),
        "favorable": (4e17, 1e17, 1.0),
        "reversed": (1e17, 4e17, 1.0),
        "mobility_off": (4e17, 1e17, 0.0),
    }
    records = [run_case(args.root, grid, name, *values)
               for grid in (32, 64) for name, values in cases.items()]
    by_key = {(row["grid"], row["case"]): row for row in records}
    passed = all(row.get("completed", False) for row in records)
    for grid in (32, 64):
        equal = by_key[grid, "equal"]
        favorable = by_key[grid, "favorable"]
        reversed_case = by_key[grid, "reversed"]
        off = by_key[grid, "mobility_off"]
        passed &= (
            equal.get("accepted") == 0
            and equal.get("a_to_b_swept_volume_m3") == 0.0
            and equal.get("b_to_a_swept_volume_m3") == 0.0
            and favorable.get("a_to_b_swept_volume_m3", 0.0) > 0.0
            and reversed_case.get("b_to_a_swept_volume_m3", 0.0) > 0.0
            and off.get("accepted") == 0
            and max(row.get("maximum_abs_line_closure_m", np.inf)
                    for row in records if row["grid"] == grid) < 1e-15
            and max(abs(row.get("last_detailed_balance_log_residual", np.inf))
                    for row in records if row["grid"] == grid) < 1e-10)
        passed &= not any(row.get("old_afterburner_called", True)
                          for row in records if row["grid"] == grid)
    result = {
        "schema": "asb-drx/v30-front-production-local-gates/v1",
        "records": records,
        "fixture_passed": True,
        "production_adapter_passed": bool(passed),
        "scientific_gate_passed": bool(passed),
        "hpc_front_authorized": bool(passed),
        "classification": ("PRODUCTION_COUPLED_BIDIRECTIONAL_FRONT_QUALIFIED_LOCAL"
                           if passed else "SIBM_FRONT_PRODUCTION_LOCAL_GATE_FAILURE"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"passed": passed, "records": records}, indent=2))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
