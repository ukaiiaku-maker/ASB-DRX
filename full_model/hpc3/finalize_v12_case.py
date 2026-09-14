#!/usr/bin/env python3
"""Write an independent, checksum-backed exit record for one v12 case."""
import csv
import glob
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


case_dir, destination = map(Path, sys.argv[1:3])
checkpoints = sorted(glob.glob(str(case_dir/"drx_v25_restart_*.npz")))
contour_path = case_dir/"sibm_contour_diagnostics.csv"
rows = list(csv.DictReader(open(contour_path))) if contour_path.exists() else []
hard = int(os.environ["APPLICATION_EXIT"]) == 0 and bool(checkpoints) and bool(rows)
ledger = {}; boundary = {}; simplex = float("nan"); nonnegative = False
if checkpoints:
    with np.load(checkpoints[-1], allow_pickle=True) as z:
        simplex = float(np.max(np.abs(np.sum(z["eta"], axis=2)-1.0)))
        nonnegative = all(float(np.min(z[k])) >= 0.0 for k in
                          ("rp", "rm", "rho_forest", "rho_wall"))
        if "sparse_front_metadata_json" in z:
            ledger = json.loads(str(z["sparse_front_metadata_json"]))["ledger"]
        if "sibm_experiment_json" in z:
            boundary = json.loads(str(z["sibm_experiment_json"]))
    hard &= simplex <= 1e-12 and nonnegative
    hard &= abs(float(ledger.get("line_closure_m", 0.0))) <= 1e-16
    hard &= float(ledger.get("signed_burgers_change_m2", 0.0)) == 0.0
    hard &= abs(float(ledger.get("line_energy_released_J", 0.0))-
                float(ledger.get("heat_released_J", 0.0))) <= 1e-20

classification = "SIBM_VALIDITY_FAILURE"
if hard:
    first, last = rows[0], rows[-1]
    tip = float(last["bulge_tip_displacement_m"])-float(first["bulge_tip_displacement_m"])
    area = float(last["excess_bulge_area_m2"])-float(first["excess_bulge_area_m2"])
    amplitude = float(last["bulge_amplitude_m"])-float(first["bulge_amplitude_m"])
    distance = float(last["tip_distance_to_window_m"])
    radius = float(os.environ["RADIUS"])
    if os.environ["TIER"] == "E":
        expected = tip < 0.0 if radius < 0.6 else tip > 0.0
        classification = ("SIBM_CRITICALITY_CONTROL_PASSED" if expected
                          else "SIBM_CRITICALITY_CONTROL_FAILED")
    elif distance <= 1.5*(10e-6/int(os.environ["GRID"])):
        classification = "SIBM_PAIR_WINDOW_LIMITED"
    elif tip > 2*(10e-6/int(os.environ["GRID"])):
        classification = "SIBM_RESOLVED_NORMAL_GROWTH"
    elif area > 0.0 and amplitude <= 10e-6/int(os.environ["GRID"]):
        classification = "SIBM_LATERAL_SPREADING_WITHOUT_NORMAL_ADVANCE"
    elif tip < -(10e-6/int(os.environ["GRID"])):
        classification = "SIBM_RETRACTED"
    elif abs(tip) <= .25*(10e-6/int(os.environ["GRID"])):
        classification = "SIBM_STALLED"
    else:
        classification = "INCONCLUSIVE_INSUFFICIENT_NORMAL_DISPLACEMENT"

record = {
    "schema": "full-v34-v12-sibm-case/v1",
    "case_id": os.environ["CASE_ID"], "tier": os.environ["TIER"],
    "classification": classification, "hard_invariants_passed": bool(hard),
    "contour_diagnostics_valid": bool(rows),
    "application_exit": int(os.environ["APPLICATION_EXIT"]),
    "source_commit": os.environ["SCIENTIFIC_SOURCE_SHA"],
    "source_archive_sha256": os.environ.get("SOURCE_ARCHIVE_SHA256", "recorded_by_hpc3_runner"),
    "source_checkpoint": os.environ["CHECKPOINT"],
    "source_checkpoint_sha256": digest(os.environ["CHECKPOINT"]),
    "grid": int(os.environ["GRID"]), "steps_requested": int(os.environ["STEPS"]),
    "rate_s-1": float(os.environ["RATE_VALUE"]), "temperature_K": float(os.environ["TEMP_VALUE"]),
    "radius_um": float(os.environ["RADIUS"]), "window_um": float(os.environ["WINDOW"]),
    "mobility_multiplier": float(os.environ["MOBILITY"]), "drag_pressure_Pa": float(os.environ["DRAG"]),
    "start_utc": os.environ["START_UTC"], "end_utc": os.environ["END_UTC"],
    "final_checkpoint": checkpoints[-1] if checkpoints else None,
    "final_checkpoint_sha256": digest(checkpoints[-1]) if checkpoints else None,
    "phase_simplex_max_error": simplex, "physical_populations_nonnegative": nonnegative,
    "front_ledger": ledger, "boundary": boundary,
    "first_contour": rows[0] if rows else None, "last_contour": rows[-1] if rows else None,
    "scientific_flags": {
        "hard_invariants_passed": bool(hard), "contour_diagnostics_valid": bool(rows),
        "matched_control_available": None, "normal_growth_relative_to_control": None,
        "window_independence_supported": None, "mobility_trend_physically_ordered": None,
        "temperature_trend_physically_interpretable": None,
        "rate_trend_physically_interpretable": None, "grid_trend_supported": None,
        "full_model_sibm_mechanism_supported": None,
    },
}
try:
    record["wallclock_seconds"] = (datetime.fromisoformat(os.environ["END_UTC"].replace("Z", "+00:00"))
                                   -datetime.fromisoformat(os.environ["START_UTC"].replace("Z", "+00:00"))).total_seconds()
except ValueError:
    record["wallclock_seconds"] = None
destination.write_text(json.dumps(record, indent=2, sort_keys=True)+"\n")
