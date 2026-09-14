#!/usr/bin/env python3
"""Decision-grade postprocessing for the single-job v14 HPC3 bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def source_identity(path):
    with np.load(path, allow_pickle=True) as z:
        return int(z["Ng"]), np.asarray(z["psi_gv"]).copy()


def case_record(directory, source):
    checkpoint = sorted(directory.glob("drx_v25_restart_*.npz"))[-1]
    rows = list(csv.DictReader((directory/"sibm_contour_diagnostics.csv").open()))
    with np.load(checkpoint, allow_pickle=True) as z:
        experiment = json.loads(str(z["sibm_experiment_json"].item()))
        ledger = json.loads(str(z["sparse_front_metadata_json"].item()))["ledger"]
        ng0, psi0 = source_identity(source)
        populations = (z["rp"], z["rm"], z["rho_forest"], z["rho_wall"])
        simplex = float(np.max(np.abs(np.sum(z["eta"], axis=2)-1.0)))
        no_allocation = (int(z["Ng"]) == ng0
                         and np.array_equal(z["psi_gv"], psi0, equal_nan=True))
    line_scale = max(abs(float(ledger["parent_line_processed_m"])), 1e-30)
    ledger_passed = (
        abs(float(ledger["line_closure_m"])) <= 1e-12*line_scale+1e-24
        and abs(float(ledger["signed_burgers_change_m2"])) <= 1e-20
        and abs(float(ledger["line_energy_released_J"])
                - float(ledger["heat_released_J"])) <= 1e-24)
    pair = experiment.get("post_seed_pair_geometry", {})
    return {
        "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
        "first_amplitude_m": float(rows[0]["bulge_amplitude_m"]),
        "final_amplitude_m": float(rows[-1]["bulge_amplitude_m"]),
        "final_tip_displacement_m": float(rows[-1]["bulge_tip_displacement_m"]),
        "final_local_normal_pressure_Pa": float(rows[-1]["local_normal_pressure_Pa"]),
        "post_seed_pair_valid": bool(pair.get("valid", False)),
        "post_seed_pair_reasons": pair.get("reasons", []),
        "minimum_population_m2": float(min(np.min(x) for x in populations)),
        "maximum_phase_simplex_error": simplex,
        "no_label_or_orientation_allocation": bool(no_allocation),
        "ledger": ledger, "ledger_passed": bool(ledger_passed),
        "hard_invariants_passed": bool(
            pair.get("valid", False) and min(np.min(x) for x in populations) >= 0.0
            and simplex <= 1e-12 and no_allocation and ledger_passed),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="fetched HPC3 run root")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run_record = Path(".hpc3/runs")/f"{args.root.name}.json"
    run = json.loads(run_record.read_text())
    output = args.root/"work/full_model/production/output"
    status = {p.stem: json.loads(p.read_text()) for p in
              (output/"status").glob("*.json")}
    sources = {
        "pinned_subcritical_128": Path(".hpc3/inputs/v14_equilibrated_common_128.npz"),
        "pinned_supercritical_128": Path(".hpc3/inputs/v14_equilibrated_common_128.npz"),
        "production_unseeded_128": Path(".hpc3/inputs/v13_canonical_bicrystal_128.npz"),
        "production_bulge_128": Path(".hpc3/inputs/v13_canonical_bicrystal_128.npz"),
        "production_bulge_192": Path(".hpc3/inputs/v13_canonical_bicrystal_192.npz"),
    }
    cases = {name: case_record(output/"cases"/name, source)
             for name, source in sources.items()}
    with np.load(sources["pinned_subcritical_128"], allow_pickle=True) as z:
        common = json.loads(str(z["sibm_experiment_json"].item()))
        base_amplitude = float(common["last_contour_metrics"]["bulge_amplitude_m"])
    sub_change = cases["pinned_subcritical_128"]["final_amplitude_m"]-base_amplitude
    super_change = cases["pinned_supercritical_128"]["final_amplitude_m"]-base_amplitude
    amp128 = cases["production_bulge_128"]["final_amplitude_m"]
    amp192 = cases["production_bulge_192"]["final_amplitude_m"]
    grid_spread = abs(amp128-amp192)/max(abs(amp128), abs(amp192), 1e-30)
    statuses_pass = set(status) == set(sources) and all(x["exit"] == 0 for x in status.values())
    hard_pass = all(x["hard_invariants_passed"] for x in cases.values())
    critical_pass = sub_change < 0.0 < super_change
    production_pass = (grid_spread <= 0.05
                       and abs(cases["production_unseeded_128"]["final_tip_displacement_m"])
                       <= 10e-6/128
                       and cases["production_bulge_128"]["final_tip_displacement_m"] > 0.0
                       and cases["production_bulge_192"]["final_tip_displacement_m"] > 0.0)
    passed = statuses_pass and hard_pass and critical_pass and production_pass
    archive = args.root/"results/results-single.tar.gz"
    result = {
        "schema": "full-v34-v14-hpc3-compact/v1",
        "classification": ("V14_HPC3_COMPACT_BUNDLE_PASSED" if passed
                           else "V14_HPC3_COMPACT_BUNDLE_FAILED"),
        "fixture_passed": bool(passed), "scientific_gate_passed": bool(passed),
        "run_id": run["run_id"], "job_id": run.get("job_id"),
        "slurm_state": run.get("state"), "retrieval": run.get("fetch_status"),
        "source_archive_sha256": digest(args.root/"input/source.tar.gz"),
        "result_archive_sha256": digest(archive),
        "status_records": status, "cases": cases,
        "pinned_criticality": {
            "common_amplitude_m": base_amplitude,
            "subcritical_change_m": sub_change,
            "supercritical_change_m": super_change,
            "sign_separation_passed": bool(critical_pass),
        },
        "production_like_comparison": {
            "unseeded_final_tip_displacement_m": cases["production_unseeded_128"]["final_tip_displacement_m"],
            "bulge_128_final_amplitude_m": amp128,
            "bulge_192_final_amplitude_m": amp192,
            "grid_relative_spread": grid_spread,
            "provisional_five_percent_passed": bool(grid_spread <= 0.05),
            "passed": bool(production_pass),
        },
        "claim_limit": (
            "generic controlled numerical qualification; production-like short cases "
            "do not constitute a material calibration or a temperature/rate trend"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(result["classification"])


if __name__ == "__main__":
    main()
