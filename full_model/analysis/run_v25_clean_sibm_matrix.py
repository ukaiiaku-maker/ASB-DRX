#!/usr/bin/env python3
"""Run and classify the fresh zero-load V25 phase-only SIBM matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = ROOT/"full_model"/"production"


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def parameters(n, steps, parent, child, mobility):
    return {
        "Nx": n, "Ny": n, "nSteps": steps, "edot_app": .001,
        "dt_strain_step": 1e-10, "sibm_clean_bicrystal_initialize": True,
        "sibm_clean_parent_density_m2": parent,
        "sibm_clean_child_density_m2": child,
        "use_sparse_common_front_state": True,
        "use_sibm_existing_boundary": True,
        "stored_energy_coupling_mode": "common_variational",
        "sibm_parent_label_override": 0, "sibm_child_label_override": 1,
        "sibm_initial_bulge_radius_um": 0.0, "sibm_pin_endpoints": False,
        "sibm_active_window_radius_um": 4.5,
        "sibm_mobility_multiplier": mobility,
        "sibm_all_defects_frozen": True,
        "sibm_front_processing_enabled": False,
        "disable_nucleation": True, "use_hazard_nucleation": False,
        "use_component_relabel": False, "diag_interval": 10,
        "save_interval": 100000, "restart_interval": steps-1,
        "plot_interval": 100000, "write_field_npz": False,
        "save_main_panels": False, "save_signed_panels": False,
    }


def run_case(root, name, p, reuse=False):
    directory = root/name; directory.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, DRX_OUTDIR=str(directory),
                       DRX_PARAMS=json.dumps(p, separators=(",", ":")),
                       MPLBACKEND="Agg", OMP_NUM_THREADS="1")
    if not reuse:
        with (directory/"stdout.log").open("w") as stdout, \
                (directory/"stderr.log").open("w") as stderr:
            subprocess.run([sys.executable, "drx_full_v34_recovery.py"],
                           cwd=PRODUCTION, env=environment, check=True,
                           stdout=stdout, stderr=stderr)
    contour = pd.read_csv(directory/"sibm_contour_diagnostics.csv")
    checkpoints = sorted(directory.glob("drx_v25_restart_*.npz"))
    width = max(np.sqrt(5e-7/5e6), 10e-6/p["Nx"])
    displacement = contour["signed_normal_displacement_mean_m"].to_numpy()
    increments = np.diff(displacement)
    significant = increments[np.abs(increments) > 1e-14*width]
    frozen_names = ("rho", "rp", "rm", "rho_forest", "rho_wall", "rho_GB",
                    "T", "psi_plastic", "gamma_slip", "eps_p", "E_tot",
                    "sigma_bar", "collective_activity_memory")
    exact = True
    with np.load(checkpoints[0], allow_pickle=True) as first, np.load(
            checkpoints[-1], allow_pickle=True) as last:
        # Reference geometry is captured before the first accepted phase step;
        # checkpoint eta already contains contrast-dependent motion.
        eta_hash = array_hash(first["sibm_reference_eta"])
        for field in frozen_names:
            exact &= np.array_equal(first[field], last[field])
        zero_load = (np.max(np.abs(first["E_tot"])) == 0.0
                     and np.max(np.abs(first["eps_p"])) == 0.0
                     and float(first["sigma_bar"]) == 0.0)
    return {
        "case": name, "phase_geometry_sha256": eta_hash,
        "normal_displacement_m": float(displacement[-1]),
        "normal_displacement_interface_widths": float(displacement[-1]/width),
        "stable_velocity_sign": bool(significant.size == 0 or np.all(
            np.sign(significant) == np.sign(significant[-1]))),
        "zero_load": bool(zero_load), "all_defect_freeze_bitwise_exact": bool(exact),
        "contour_samples": int(len(contour)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=ROOT/"results-local/v25-clean-sibm-matrix")
    parser.add_argument("--output", type=Path,
                        default=ROOT/"full_model/verification/v25_clean_sibm_matrix.json")
    parser.add_argument("--grid", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args(); args.root.mkdir(parents=True, exist_ok=True)
    specifications = {
        "equal": (2.5e17, 2.5e17, 1000.0),
        "favorable": (4e17, 1e17, 1000.0),
        "reversed": (1e17, 4e17, 1000.0),
        "mobility_off": (4e17, 1e17, 0.0),
    }
    records = []
    for name, (parent, child, mobility) in specifications.items():
        records.append(run_case(
            args.root, name,
            parameters(args.grid, args.steps, parent, child, mobility), args.reuse))
    by_name = {item["case"]: item for item in records}
    geometry_identical = len({item["phase_geometry_sha256"]
                              for item in records}) == 1
    passed = (
        geometry_identical
        and all(item["zero_load"] and item["all_defect_freeze_bitwise_exact"]
                for item in records)
        and abs(by_name["equal"]["normal_displacement_interface_widths"]) < .05
        and by_name["favorable"]["normal_displacement_interface_widths"] >= .25
        and by_name["reversed"]["normal_displacement_interface_widths"] <= -.25
        and by_name["mobility_off"]["normal_displacement_interface_widths"] == 0.0
        and by_name["favorable"]["stable_velocity_sign"]
        and by_name["reversed"]["stable_velocity_sign"])
    result = {
        "schema": "asb-drx/v25-clean-production-sibm-phase-only/v1",
        "grid": args.grid, "steps": args.steps, "records": records,
        "common_phase_geometry": geometry_identical,
        "phase_only_directional_controls_passed": bool(passed),
        "sequential_full_driver_activation_completed": False,
        "classification": (
            "CLEAN_PRODUCTION_PHASE_ONLY_SIBM_SIGN_QUALIFIED"
            if passed else "CLEAN_PRODUCTION_PHASE_ONLY_SIBM_NOT_QUALIFIED"),
        "fixture_passed": True, "scientific_gate_passed": False,
        "hpc_sibm_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(result["classification"])


if __name__ == "__main__":
    main()
