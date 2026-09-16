#!/usr/bin/env python3
"""Postprocess complete and partial V30 front cases without discarding either."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT/"full_model"/"hpc3"/"v30_front_case_manifest.json"


def latest_checkpoint(case_root):
    candidates = []
    for path in case_root.glob("segments/segment-*/attempt-*/*.npz"):
        try:
            with np.load(path, allow_pickle=True) as data:
                if "coupled_front_metadata_json" not in data.files:
                    continue
                step = int(data["step"])
            candidates.append((step, path.stat().st_mtime_ns, path))
        except Exception:
            continue
    return sorted(candidates)[-1] if candidates else None


def contour_history(case_root):
    rows = {}
    for path in sorted(case_root.glob(
            "segments/segment-*/attempt-*/sibm_contour_diagnostics.csv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    key = (int(float(row["step"])), float(row["time_s"]))
                    float(row["signed_normal_displacement_mean_m"])
                except (KeyError, TypeError, ValueError):
                    continue
                rows[key] = row
    return [rows[key] for key in sorted(rows)]


def case_record(case, output_root):
    case_root = output_root/case["id"]
    status_path = case_root/"case_status.json"
    status = (json.loads(status_path.read_text()) if status_path.exists()
              else {"state": case["dependency_state"]})
    record = {
        "case": case["id"], "tier": case["tier"], "grid": case["grid"],
        "state": status.get("state", "NOT_STARTED"),
        "source_sha": status.get("source_sha"),
        "job_id": status.get("job_id"), "run_count": len(status.get("runs", [])),
        "terminal_classification": (status.get("terminal_event") or {}).get(
            "classification"),
    }
    checkpoint = latest_checkpoint(case_root)
    if checkpoint is None:
        return record
    record.update(final_step=checkpoint[0], checkpoint=str(checkpoint[2]))
    with np.load(checkpoint[2], allow_pickle=True) as data:
        coupled = json.loads(str(data["coupled_front_metadata_json"].item()))
        experiment = json.loads(str(data["sibm_experiment_json"].item()))
    ledger = coupled["ledger"]
    record.update(
        attempts=int(ledger["attempts"]), accepted=int(ledger["accepted"]),
        a_to_b_volume_m3=float(ledger["a_to_b_swept_volume_m3"]),
        b_to_a_volume_m3=float(ledger["b_to_a_swept_volume_m3"]),
        signed_volume_m3=float(ledger["a_to_b_swept_volume_m3"]
                               -ledger["b_to_a_swept_volume_m3"]),
        revisit_volume_m3=float(ledger["revisit_volume_m3"]),
        heat_J=float(ledger["heat_J"]),
        line_closure_m=float(ledger["maximum_abs_line_closure_m"]),
        signed_closure_m2=float(ledger["maximum_abs_signed_closure_m2"]),
        legacy_afterburner_calls=int(experiment.get(
            "legacy_afterburner_calls", 0)))
    history = contour_history(case_root)
    record["contour_samples"] = len(history)
    if history:
        displacement = float(history[-1]["signed_normal_displacement_mean_m"])
        record["signed_displacement_m"] = displacement
        record["signed_displacement_cells"] = displacement/(10e-6/int(case["grid"]))
        record["final_time_s"] = float(history[-1]["time_s"])
    return record


def odd_pair(records, grid, magnitude):
    tag = f"{magnitude:.0e}".replace("e-0", "e-")
    plus = records.get(f"a1_n{grid}_delta_p{tag}")
    minus = records.get(f"a1_n{grid}_delta_m{tag}")
    if plus is None or minus is None:
        return False, math.inf
    p = plus.get("signed_volume_m3", 0.0)
    m = minus.get("signed_volume_m3", 0.0)
    scale = max(abs(p), abs(m), 1e-300)
    residual = abs(p+m)/scale
    return p > 0.0 and m < 0.0 and residual <= .05, residual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path,
                        default=ROOT/"full_model"/"production"/"output"/
                        "v30_front")
    parser.add_argument("--output", type=Path, default=ROOT/"full_model"/
                        "verification"/"v30_front_hpc_decision.json")
    parser.add_argument("--index", type=Path, default=ROOT/"full_model"/
                        "verification"/"v30_front_long_run_index.csv")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    rows = [case_record(case, args.output_root.resolve())
            for case in manifest["cases"]]
    by_id = {row["case"]: row for row in rows}
    a1 = [row for row in rows if row["tier"].startswith("A1_")]
    complete = all(row["state"] == "PASSED" and "final_step" in row
                   for row in a1)
    failures = []
    if complete:
        for grid in (64, 128):
            equal = by_id[f"a1_n{grid}_equal"]
            off = by_id[f"a1_n{grid}_mobility_off"]
            favorable = by_id[f"a1_n{grid}_favorable"]
            reversed_case = by_id[f"a1_n{grid}_reversed"]
            swapped = by_id[f"a1_n{grid}_label_swap_favorable"]
            if (abs(equal.get("signed_displacement_cells", math.inf)) >= .02
                    or equal.get("a_to_b_volume_m3") != 0.0
                    or equal.get("b_to_a_volume_m3") != 0.0
                    or equal.get("heat_J") != 0.0):
                failures.append(f"n{grid}_equal_state_drift")
            if (off.get("a_to_b_volume_m3") != 0.0
                    or off.get("b_to_a_volume_m3") != 0.0):
                failures.append(f"n{grid}_mobility_off_motion")
            if favorable.get("signed_volume_m3", 0.0) <= 0.0:
                failures.append(f"n{grid}_favorable_sign")
            if reversed_case.get("signed_volume_m3", 0.0) >= 0.0:
                failures.append(f"n{grid}_reversed_sign")
            if swapped.get("signed_volume_m3", 0.0) <= 0.0:
                failures.append(f"n{grid}_label_exchange_covariance")
            elif abs(swapped["signed_volume_m3"]-favorable["signed_volume_m3"])/max(
                    abs(swapped["signed_volume_m3"]),
                    abs(favorable["signed_volume_m3"]), 1e-300) > .05:
                failures.append(f"n{grid}_label_exchange_magnitude")
            cycle = by_id[f"a1_n{grid}_advance_retreat_readvance"]
            if (cycle.get("a_to_b_volume_m3", 0.0) <= 0.0
                    or cycle.get("b_to_a_volume_m3", 0.0) <= 0.0
                    or cycle.get("revisit_volume_m3", 0.0) <= 0.0):
                failures.append(f"n{grid}_closed_cycle_history")
            for magnitude in (1e-6, 1e-3, 1e-2):
                passed, _ = odd_pair(by_id, grid, magnitude)
                if not passed:
                    failures.append(f"n{grid}_delta_{magnitude:.0e}_odd_response")
        for row in a1:
            if (row.get("legacy_afterburner_calls", 1) != 0
                    or row.get("line_closure_m", math.inf) > 1e-15
                    or row.get("signed_closure_m2", math.inf) > 1.0):
                failures.append(f"{row['case']}_ledger_or_operator")
        for sense in ("favorable", "reversed"):
            coarse = abs(by_id[f"a1_n64_{sense}"]["signed_volume_m3"])
            fine = abs(by_id[f"a1_n128_{sense}"]["signed_volume_m3"])
            if abs(coarse-fine)/max(coarse, fine, 1e-300) > .05:
                failures.append(f"{sense}_64_128_not_within_5pct")
    a1_passed = complete and not failures
    decision = {
        "schema": "asb-drx/v30-front-hpc-decision/v1",
        "manifest": str(args.manifest), "records": rows,
        "a1_complete": complete, "a1_passed": a1_passed,
        "a2_authorized": a1_passed,
        "failures": failures,
        "partial_cases_preserved": [row["case"] for row in rows
                                     if row["state"] == "PARTIAL_RESTARTABLE"],
        "classification": (
            "PRODUCTION_COUPLED_BIDIRECTIONAL_FRONT_QUALIFIED"
            if a1_passed else
            "V30_FRONT_A1_INCOMPLETE" if not complete else
            "V30_FRONT_A1_SCIENTIFIC_FAILURE"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2, sort_keys=True)+"\n")
    fields = sorted({key for row in rows for key in row})
    with args.index.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({key: decision[key] for key in (
        "classification", "a1_complete", "a1_passed", "a2_authorized",
        "failures", "partial_cases_preserved")}, indent=2))


if __name__ == "__main__":
    main()
