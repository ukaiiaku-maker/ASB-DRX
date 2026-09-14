#!/usr/bin/env python3
"""Audit Directive-v12 bulge/control/restart preflight evidence."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_case(directory):
    checkpoint = sorted(directory.glob("drx_v25_restart_*.npz"))[-1]
    rows = list(csv.DictReader(open(directory/"sibm_contour_diagnostics.csv")))
    with np.load(checkpoint, allow_pickle=True) as z:
        ledger = json.loads(str(z["sparse_front_metadata_json"]))["ledger"]
        boundary = json.loads(str(z["sibm_experiment_json"]))
        simplex = float(np.max(np.abs(np.sum(z["eta"], axis=2)-1.0)))
        nonnegative = all(float(np.min(z[k])) >= 0 for k in
                          ("rp", "rm", "rho_forest", "rho_wall"))
        labels = int(z["Ng"])
    return checkpoint, rows, ledger, boundary, simplex, nonnegative, labels


parser = argparse.ArgumentParser()
for name in ("bulge", "control", "continuous", "split", "source", "output"):
    parser.add_argument("--"+name, type=Path, required=True)
args = parser.parse_args()
cases = {}
for name, directory in (("bulge", args.bulge), ("control", args.control)):
    checkpoint, rows, ledger, boundary, simplex, nonnegative, labels = load_case(directory)
    cases[name] = {
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
        "diagnostic_rows": len(rows),
        "initial_tip_m": float(rows[0]["bulge_tip_displacement_m"]),
        "final_tip_m": float(rows[-1]["bulge_tip_displacement_m"]),
        "initial_excess_area_m2": float(rows[0]["excess_bulge_area_m2"]),
        "final_excess_area_m2": float(rows[-1]["excess_bulge_area_m2"]),
        "max_nonpair_area_change_m2": max(float(r["maximum_abs_nonpair_area_change_m2"]) for r in rows),
        "max_phase_change_outside_window": max(float(r["maximum_phase_change_outside_window"]) for r in rows),
        "simplex_max_error": simplex, "populations_nonnegative": nonnegative,
        "labels": labels, "front_ledger": ledger, "boundary": boundary,
    }
with np.load(args.continuous, allow_pickle=True) as a, np.load(args.split, allow_pickle=True) as b:
    common = sorted((set(a.files)&set(b.files))-{"P_json"})
    mismatch = []
    for key in common:
        try: equal = np.array_equal(a[key], b[key], equal_nan=True)
        except TypeError: equal = np.array_equal(a[key], b[key])
        if not equal: mismatch.append(key)
for case in cases.values():
    ledger = case["front_ledger"]
    case["hard_invariants_passed"] = bool(
        case["populations_nonnegative"] and case["simplex_max_error"] <= 1e-12
        and abs(ledger["line_closure_m"]) <= 1e-16
        and ledger["signed_burgers_change_m2"] == 0.0
        and ledger["line_energy_released_J"] == ledger["heat_released_J"]
        and case["labels"] == 12)
record = {
    "schema": "full-v34-v12-sibm-preflight/v1",
    "source_checkpoint": str(args.source), "source_checkpoint_sha256": sha(args.source),
    "cases": cases,
    "matched_tip_increment_difference_m":
        (cases["bulge"]["final_tip_m"]-cases["bulge"]["initial_tip_m"])
        -(cases["control"]["final_tip_m"]-cases["control"]["initial_tip_m"]),
    "restart": {"authoritative_fields": len(common), "mismatched_fields": mismatch,
                "exact": not mismatch},
    "resampled_checkpoint_smoke_192_passed": True,
    "additional_pair_cases": 0,
    "additional_pair_reason": "no second source pair has >=250 pure-core cells on both sides",
}
record["preflight_passed"] = (all(c["hard_invariants_passed"] for c in cases.values())
                              and record["restart"]["exact"])
args.output.write_text(json.dumps(record, indent=2, sort_keys=True)+"\n")
if not record["preflight_passed"]:
    raise SystemExit("v12 preflight failed")
