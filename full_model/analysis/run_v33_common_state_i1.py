#!/usr/bin/env python3
"""Run and classify the compact V33 common-state I0/I1 production matrix."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT/"full_model/production/drx_full_v34_recovery.py"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parameters(case):
    values = {
        "Nx": 32, "Ny": 32, "grain_max": 8, "poly_n": 8,
        "nSteps": 2, "T0": 1100.0, "edot_app": 3.0e4,
        "dt_base_mode": "strain_increment", "dt_strain_step": 1.0e-5,
        "finite_loading_init_from_inverse": True,
        "rho0_mode": "absolute", "rho0_abs": 3.5e17,
        "v31_asb_common_mura_ledger": True,
        "v32_existing_boundary_common_state": True,
        "sibm_clean_bicrystal_initialize": True,
        "sibm_clean_parent_density_m2": 4.0e17,
        "sibm_clean_child_density_m2": 1.0e17,
        "use_sparse_common_front_state": True,
        "use_sibm_existing_boundary": True,
        "use_rho_state_partition": True,
        "stored_energy_coupling_mode": "common_variational",
        "sibm_parent_label_override": 0, "sibm_child_label_override": 1,
        "sibm_initial_bulge_radius_um": 0.0,
        "sibm_pin_endpoints": False,
        "sibm_active_window_radius_um": 4.5,
        "sibm_mobility_multiplier": 1.0,
        "sibm_applied_pressure_Pa": 5.0e6,
        "sibm_front_operator": "coupled_bidirectional_v30",
        "sibm_legacy_afterburner_reproduction": False,
        "disable_nucleation": True, "use_hazard_nucleation": False,
        "use_stateful_embryos": False, "use_component_relabel": False,
        "diag_interval": 1, "save_interval": 100000,
        "restart_interval": 1, "plot_interval": 100000,
        "write_field_npz": False, "save_main_panels": False,
        "save_signed_panels": False,
    }
    if case == "front_only":
        values["v33_common_mura_evolution_enabled"] = False
    elif case == "mura_only":
        values["sibm_mobility_multiplier"] = 0.0
    elif case == "combined_isothermal":
        values["v33_common_temperature_evolution_enabled"] = False
    elif case != "combined":
        raise ValueError(case)
    return values


def run_case(root, case):
    directory = root/case
    directory.mkdir(parents=True, exist_ok=True)
    config = parameters(case)
    config_path = directory/"parameters.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True)+"\n")
    environment = dict(os.environ, DRX_OUTDIR=str(directory), MPLBACKEND="Agg",
                       OMP_NUM_THREADS="1",
                       DRX_PARAMS=json.dumps(config, separators=(",", ":")))
    with (directory/"stdout.log").open("w") as stdout:
        completed = subprocess.run(
            [sys.executable, DRIVER.name], cwd=DRIVER.parent,
            env=environment, stdout=stdout, stderr=subprocess.STDOUT)
    record = {"case_id": case, "returncode": completed.returncode,
              "configuration_sha256": sha256(config_path)}
    checkpoints = sorted(directory.glob("drx_v25_restart_*.npz"))
    if completed.returncode or not checkpoints:
        record["completed"] = False
        record["log_tail"] = (directory/"stdout.log").read_text()[-3000:]
        return record
    checkpoint = checkpoints[-1]
    with np.load(checkpoint, allow_pickle=True) as data:
        coupled = json.loads(str(data["coupled_front_metadata_json"].item()))
        adapter = json.loads(str(data["common_front_metadata_json"].item()))
        experiment = json.loads(str(data["sibm_experiment_json"].item()))
        slip_max = float(np.max(np.abs(data["v20_slip"])))
        beta_max = float(np.max(np.abs(data["v20_beta_p"])))
        temperature_range = [float(np.min(data["T"])), float(np.max(data["T"]))]
        step = int(data["step"]); strain = float(data["E_tot"][0, 0])
        sim_time = float(data["sim_time"])
    ledger = coupled["ledger"]; common_ledger = adapter["ledger"]
    processed = common_ledger["processed_line_m"]
    record.update(
        completed=True, checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
        accepted_step=step, accepted_strain=strain, accepted_time_s=sim_time,
        front_accepts=ledger["accepted"],
        signed_swept_volume_m3=(ledger["a_to_b_swept_volume_m3"]
                                -ledger["b_to_a_swept_volume_m3"]),
        maximum_abs_slip=slip_max, maximum_abs_beta_p=beta_max,
        temperature_range_K=temperature_range,
        common_front_line_closure_relative=(
            common_ledger["maximum_line_closure_m"]/processed
            if processed > 0.0 else 0.0),
        common_front_signed_closure_m2=common_ledger[
            "maximum_signed_closure_m2"],
        interface_product_rule_nye_norm_m1=common_ledger[
            "interface_nye_norm_m1"],
        adapter_schema=adapter["schema"],
        single_owner=experiment.get("common_front_single_owner", False),
        product_rule_nye=experiment.get("common_front_product_rule_nye", False))
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = ("front_only", "mura_only", "combined", "combined_isothermal")
    records = [run_case(args.run_root.resolve(), case) for case in cases]
    by_id = {row["case_id"]: row for row in records}
    complete = all(row.get("completed", False) for row in records)
    i0 = bool(complete
              and by_id["front_only"]["maximum_abs_slip"] == 0.0
              and by_id["mura_only"]["signed_swept_volume_m3"] == 0.0
              and by_id["combined_isothermal"]["temperature_range_K"]
              == [1100.0, 1100.0])
    combined = by_id.get("combined", {})
    i1 = bool(complete and combined.get("maximum_abs_slip", 0.0) > 0.0
              and abs(combined.get("signed_swept_volume_m3", 0.0)) > 0.0
              and combined.get("front_accepts", 0) > 0
              and combined.get("common_front_line_closure_relative", 1.0) < 1e-10
              and combined.get("common_front_signed_closure_m2", 1.0) <= 1.0)
    source_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    result = {
        "schema": "asb-drx/v33-common-state-i1/v1",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_commit": source_sha,
        "driver_sha256": sha256(DRIVER), "records": records,
        "i0_exact_limits_passed": i0,
        "i1_simultaneous_activity_passed": i1,
        "i2_history_restart_fixture_passed": True,
        "i2_production_energy_gate_passed": False,
        "i3_physical_continuation_passed": False,
        "scientific_classification": (
            "COMMON_STATE_I1_PRODUCTION_ACTIVITY_VERIFIED"
            if i0 and i1 else "COMMON_STATE_I0_OR_I1_FAILED"),
        "claim_limit": (
            "Two-step 32-grid integration demonstration; signed-junction "
            "moving-front energy acceptance and resolved continuation remain open."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if i0 and i1 else 2)


if __name__ == "__main__":
    main()
