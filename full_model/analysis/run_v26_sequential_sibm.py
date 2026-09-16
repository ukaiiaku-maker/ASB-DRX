#!/usr/bin/env python3
"""Run the cumulative S0--S5 production-driver SIBM qualification matrix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = ROOT/"full_model"/"production"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from full_model.production.symmetric_sibm import production_stage_overrides


def parameters(n, steps, stage, parent, child, mobility, *, label_swapped=False):
    result = {
        "Nx": n, "Ny": n, "grain_max": 8,
        "nSteps": steps, "edot_app": .001,
        "dt_strain_step": 1e-10, "sibm_clean_bicrystal_initialize": True,
        "sibm_clean_parent_density_m2": parent,
        "sibm_clean_child_density_m2": child,
        "use_sparse_common_front_state": True,
        "use_sibm_existing_boundary": True,
        "stored_energy_coupling_mode": "common_variational",
        "sibm_parent_label_override": 1 if label_swapped else 0,
        "sibm_child_label_override": 0 if label_swapped else 1,
        "sibm_initial_bulge_radius_um": 0.0, "sibm_pin_endpoints": False,
        "sibm_active_window_radius_um": 4.5,
        "sibm_mobility_multiplier": mobility,
        "disable_nucleation": True, "use_hazard_nucleation": False,
        "use_component_relabel": False, "diag_interval": max(1, steps//10),
        "save_interval": 100000, "restart_interval": max(steps-1, 1),
        "plot_interval": 100000, "write_field_npz": False,
        "save_main_panels": False, "save_signed_panels": False,
    }
    result.update(production_stage_overrides(stage))
    return result


def run_case(root, name, config):
    directory = (root/name).resolve(); directory.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, DRX_OUTDIR=str(directory),
                       DRX_PARAMS=json.dumps(config, separators=(",", ":")),
                       MPLBACKEND="Agg", OMP_NUM_THREADS="1")
    with (directory/"stdout.log").open("w") as stdout, \
            (directory/"stderr.log").open("w") as stderr:
        subprocess.run([sys.executable, "drx_full_v34_recovery.py"],
                       cwd=PRODUCTION, env=environment, check=True,
                       stdout=stdout, stderr=stderr)
    contour = pd.read_csv(directory/"sibm_contour_diagnostics.csv")
    displacement = contour["signed_normal_displacement_mean_m"].to_numpy()
    label_swapped = (int(config["sibm_parent_label_override"]) == 1)
    signed_contour_area = float(np.sum(
        contour["parent_to_child_contour_area_m2"].to_numpy()
        -contour["child_to_parent_contour_area_m2"].to_numpy()))
    terminal_path = directory/"sibm_terminal_event.json"
    terminal = (json.loads(terminal_path.read_text())
                if terminal_path.exists() else None)
    checkpoints = sorted(directory.glob("drx_v25_restart_*.npz"))
    with np.load(checkpoints[-1], allow_pickle=True) as data:
        meta = json.loads(str(data["sparse_front_metadata_json"].item()))
    ledger = meta["ledger"]
    width = max(np.sqrt(5e-7/5e6), 10e-6/config["Nx"])
    increments = np.diff(displacement)
    significant = increments[np.abs(increments) > 1e-14*width]
    return {
        "case": name, "stage": config["sibm_sequential_stage"],
        "displacement_m": float(displacement[-1]),
        "displacement_interface_widths": float(displacement[-1]/width),
        "fixed_reference_displacement_interface_widths": float(
            (-1.0 if label_swapped else 1.0)*displacement[-1]/width),
        "label_swapped": label_swapped,
        "observed_sign": int(np.sign(displacement[-1])),
        "cumulative_signed_contour_area_m2": signed_contour_area,
        "response_sign": int(np.sign(signed_contour_area)),
        "fixed_reference_response_sign": int(
            (-1 if label_swapped else 1)*np.sign(signed_contour_area)),
        "terminal_event": terminal,
        "stable_velocity_sign": bool(significant.size == 0 or np.all(
            np.sign(significant) == np.sign(significant[-1]))),
        "first_pressure_Pa": float(contour["local_normal_pressure_Pa"].iloc[0]),
        "final_pressure_Pa": float(contour["local_normal_pressure_Pa"].iloc[-1]),
        "front_ledger": ledger,
        "line_closure_abs_m": abs(float(ledger["line_closure_m"])),
        "heat_equals_line_energy": bool(
            float(ledger["heat_released_J"]) ==
            float(ledger["line_energy_released_J"])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=ROOT/"results-local/v26-sequential-sibm")
    parser.add_argument("--output", type=Path,
                        default=ROOT/"full_model/verification/v26_sequential_sibm.json")
    parser.add_argument("--grid", type=int, default=32)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--stages", nargs="+", default=[f"S{i}" for i in range(6)])
    args = parser.parse_args(); args.root.mkdir(parents=True, exist_ok=True)
    cases = {
        "equal": (2.5e17, 2.5e17, 1000.0, False),
        "favorable": (4e17, 1e17, 1000.0, False),
        "reversed": (1e17, 4e17, 1000.0, False),
        "mobility_off": (4e17, 1e17, 0.0, False),
        "label_swapped": (1e17, 4e17, 1000.0, True),
    }
    records = []
    for stage in args.stages:
        for case, (parent, child, mobility, swapped) in cases.items():
            name = f"{stage.lower()}-{case}"
            records.append(run_case(
                args.root, name,
                parameters(args.grid, args.steps, stage, parent, child, mobility,
                           label_swapped=swapped)))
    by_stage = {}
    for stage in args.stages:
        rows = {row["case"].split("-", 1)[1]: row for row in records
                if row["stage"] == stage.upper()}
        closure = max((row["line_closure_abs_m"] for row in rows.values()), default=0.0)
        passed = (abs(rows["equal"]["displacement_interface_widths"]) < .05
                  and rows["favorable"]["response_sign"] > 0
                  and rows["reversed"]["response_sign"] < 0
                  and rows["mobility_off"]["response_sign"] == 0
                  and rows["favorable"]["fixed_reference_response_sign"]
                  *rows["label_swapped"]["fixed_reference_response_sign"] < 0
                  and closure <= 1e-18
                  and all(row["heat_equals_line_energy"] for row in rows.values()))
        by_stage[stage.upper()] = {"passed": bool(passed),
                                   "maximum_line_closure_abs_m": closure}
    first_failure = next((stage for stage in args.stages
                          if not by_stage[stage.upper()]["passed"]), None)
    result = {
        "schema": "asb-drx/v26-sequential-production-sibm/v1",
        "grid": args.grid, "steps": args.steps, "records": records,
        "stage_decisions": by_stage,
        "first_directionality_breaking_stage": first_failure,
        "all_requested_stages_passed": first_failure is None,
        "hpc_sibm_authorized": first_failure is None and "S5" in args.stages,
        "classification": (
            "LOCAL_SEQUENTIAL_SIBM_CHANNELS_QUALIFIED_HPC_PENDING"
            if first_failure is None and "S5" in args.stages else
            "SIBM_CHANNEL_DIRECTIONALITY_FAILURE"),
        "local_stage_gate_passed": first_failure is None,
        "fixture_passed": True,
        "scientific_gate_passed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"stage_decisions": by_stage,
                      "first_failure": first_failure}, indent=2))


if __name__ == "__main__":
    main()
