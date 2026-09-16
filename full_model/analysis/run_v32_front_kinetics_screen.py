#!/usr/bin/env python3
"""Small bounded V32 screen of front kinetics, excluding closed controls.

Equal-state and mobility-off controls are intentionally not rerun: their V30
production evidence is referenced by checksum.  This screen probes only the
discriminating favorable/reversed pair for bounded kinetic perturbations, then
checks label exchange and near-equal antisymmetry at the selected center point.
"""

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
PRODUCTION = ROOT/"full_model"/"production"
V30_CONTROL = ROOT/"full_model"/"verification"/"v30_front_production_local.json"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parameters(comparison, candidate):
    density = {
        "favorable": (4e17, 1e17, 0, 1),
        "reversed": (1e17, 4e17, 0, 1),
        "label_swap_favorable": (4e17, 1e17, 1, 0),
        "near_equal_plus": (2.50000125e17, 2.49999875e17, 0, 1),
        "near_equal_minus": (2.49999875e17, 2.50000125e17, 0, 1),
    }[comparison]
    return {
        "Nx": 32, "Ny": 32, "grain_max": 8, "nSteps": 3,
        "edot_app": .001, "dt_strain_step": 1e-10,
        "sibm_clean_bicrystal_initialize": True,
        "sibm_clean_parent_density_m2": density[0],
        "sibm_clean_child_density_m2": density[1],
        "use_sparse_common_front_state": True,
        "use_sibm_existing_boundary": True,
        "stored_energy_coupling_mode": "common_variational",
        "sibm_parent_label_override": density[2],
        "sibm_child_label_override": density[3],
        "sibm_initial_bulge_radius_um": 0.0,
        "sibm_pin_endpoints": False,
        "sibm_active_window_radius_um": 4.5,
        "sibm_mobility_multiplier": 1.0,
        "sibm_front_operator": "coupled_bidirectional_v30",
        "sibm_legacy_afterburner_reproduction": False,
        "moving_front_support_component_reconnection": True,
        "sibm_sequential_stage": "S3",
        "disable_nucleation": True, "use_hazard_nucleation": False,
        "use_component_relabel": False, "diag_interval": 1,
        "save_interval": 100000, "restart_interval": 2,
        "plot_interval": 100000, "write_field_npz": False,
        "save_main_panels": False, "save_signed_panels": False,
        **candidate,
    }


def _run(root, candidate_id, comparison, candidate):
    directory = root/candidate_id/comparison
    directory.mkdir(parents=True, exist_ok=True)
    config = _parameters(comparison, candidate)
    (directory/"parameters.json").write_text(
        json.dumps(config, indent=2, sort_keys=True)+"\n")
    checkpoints = sorted(directory.glob("drx_v25_restart_*.npz"))
    returncode = 0
    if not checkpoints:
        environment = dict(
            os.environ, DRX_OUTDIR=str(directory), MPLBACKEND="Agg",
            OMP_NUM_THREADS="1",
            DRX_PARAMS=json.dumps(config, separators=(",", ":")))
        with (directory/"stdout.log").open("w") as stdout, \
                (directory/"stderr.log").open("w") as stderr:
            completed = subprocess.run(
                [sys.executable, "drx_full_v34_recovery.py"], cwd=PRODUCTION,
                env=environment, stdout=stdout, stderr=stderr, check=False)
        returncode = completed.returncode
        checkpoints = sorted(directory.glob("drx_v25_restart_*.npz"))
    record = {"candidate": candidate_id, "comparison": comparison,
              "returncode": returncode, "completed": False}
    if returncode or not checkpoints:
        record["stderr_tail"] = (directory/"stderr.log").read_text()[-2000:]
        return record
    with np.load(checkpoints[-1], allow_pickle=True) as data:
        coupled = json.loads(str(data["coupled_front_metadata_json"].item()))
        experiment = json.loads(str(data["sibm_experiment_json"].item()))
    ledger = coupled["ledger"]
    record.update(
        completed=True, attempts=ledger["attempts"], accepted=ledger["accepted"],
        a_to_b_swept_volume_m3=ledger["a_to_b_swept_volume_m3"],
        b_to_a_swept_volume_m3=ledger["b_to_a_swept_volume_m3"],
        maximum_abs_line_closure_m=ledger["maximum_abs_line_closure_m"],
        maximum_abs_signed_closure_m2=ledger["maximum_abs_signed_closure_m2"],
        topology_event_count=coupled.get("topology_event_count", 0),
        last_decision=experiment["front_last_decision"])
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = {
        "center": {},
        "attempt_quarter": {"moving_front_attempt_frequency_s": 2.5e7},
        "attempt_fourfold": {"moving_front_attempt_frequency_s": 4.0e8},
        "barrier_low": {"sibm_activation_h0_eV": 0.30},
        "barrier_high": {"sibm_activation_h0_eV": 0.40},
        "transfer_low": {
            "moving_front_fixture_transmission_fraction": 0.25,
            "moving_front_boundary_storage_fraction": 0.10},
    }
    records = []
    for candidate_id, candidate in candidates.items():
        for comparison in ("favorable", "reversed"):
            records.append(_run(args.root, candidate_id, comparison, candidate))
    for comparison in ("label_swap_favorable", "near_equal_plus",
                       "near_equal_minus"):
        records.append(_run(args.root, "center", comparison, candidates["center"]))
    by_key = {(row["candidate"], row["comparison"]): row for row in records}
    robust = []
    for candidate_id in candidates:
        favorable = by_key[candidate_id, "favorable"]
        reversed_case = by_key[candidate_id, "reversed"]
        if (favorable.get("completed") and reversed_case.get("completed")
                and favorable["a_to_b_swept_volume_m3"] > 0.0
                and reversed_case["b_to_a_swept_volume_m3"] > 0.0
                and max(favorable["maximum_abs_line_closure_m"],
                        reversed_case["maximum_abs_line_closure_m"]) < 1e-15
                and max(favorable["maximum_abs_signed_closure_m2"],
                        reversed_case["maximum_abs_signed_closure_m2"]) < 1e-20):
            robust.append(candidate_id)
    center = by_key["center", "favorable"]
    swap = by_key["center", "label_swap_favorable"]
    exchange_residual = abs(
        center.get("a_to_b_swept_volume_m3", np.inf)
        -swap.get("a_to_b_swept_volume_m3", -np.inf))
    exchange_scale = max(abs(center.get("a_to_b_swept_volume_m3", 0.0)), 1e-300)
    center_qualified = bool(
        "center" in robust and swap.get("completed")
        and exchange_residual/exchange_scale < 0.05
        and by_key["center", "near_equal_plus"].get("completed")
        and by_key["center", "near_equal_minus"].get("completed"))
    result = {
        "schema": "asb-drx/v32-front-kinetics-screen/v1",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": "bounded 32-grid three-step discriminating screen",
        "excluded_reruns": ["equal", "mobility_off"],
        "control_evidence": {"path": str(V30_CONTROL.relative_to(ROOT)),
                             "sha256": _sha256(V30_CONTROL)},
        "candidate_definitions": candidates, "records": records,
        "robust_candidates": robust,
        "selected_candidate": "center" if center_qualified else None,
        "label_exchange_relative_residual": exchange_residual/exchange_scale,
        "label_exchange_tolerance": 0.05,
        "kinetics_discriminated_by_three_step_screen": False,
        "kinetics_discrimination_note": (
            "All bounded candidates accept the complete available phase trial; "
            "the short response is geometry-limited, so no speed optimum is inferred."),
        "promotion_decision": (
            "RETAIN_CENTER_AS_NEUTRAL_ANCHOR_FOR_TOPOLOGY_LIMITED_REPLAY"
            if center_qualified
            else "DO_NOT_PROMOTE_FRONT_KINETICS"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"robust": robust, "selected": result["selected_candidate"]},
                     indent=2))
    raise SystemExit(0 if center_qualified else 2)


if __name__ == "__main__":
    main()
