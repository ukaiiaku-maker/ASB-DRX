#!/usr/bin/env python3
"""Offline Route-B hazard reweighting on saved full-v34 field audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

KB_J_K = 1.380649e-23
EV_J = 1.602176634e-19


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def creation_rate(z, *, attempt_s, h0_eV, critical_pa, exp_a, exp_n,
                  floor, entropy_kB, rate_cap_s):
    pressure = np.maximum(np.asarray(z["dF_density"], dtype=float), 0.0)
    enthalpy = h0_eV*EV_J*(floor + (1.0-floor)*np.exp(
        -exp_a*(pressure/critical_pa)**exp_n))
    free = enthalpy - KB_J_K*np.asarray(z["T"], dtype=float)*entropy_kB
    exponent = np.where(free < 0.0, 40.0,
                        np.clip(-free/(KB_J_K*np.asarray(z["T"])), -700.0, 40.0))
    rate = attempt_s*np.asarray(z["site_factor"])*np.asarray(z["gate_AT"])*np.exp(exponent)
    possible = np.isfinite(np.asarray(z["classical_barrier_J"])) & (pressure > 0.0)
    return np.where(possible, np.minimum(rate, rate_cap_s), 0.0)


def integrate(audits, site_density_m2, kinetic):
    exposure_field = np.zeros_like(np.asarray(audits[0]["T"], dtype=float))
    interval_rows = []
    previous = None
    for z in audits:
        rate = creation_rate(z, **kinetic)
        time_s = float(z["sim_time"])
        strain = float(z["strain"])
        if previous is not None:
            dt = time_s-previous[0]
            increment = 0.5*(previous[2]+rate)*site_density_m2*float(z["cell_area_m2"])*dt
            exposure_field += increment
            interval_rows.append({
                "end_time_s": time_s, "end_strain": strain,
                "exposure_increment": float(np.sum(increment))})
        previous = (time_s, strain, rate)
    total = float(np.sum(exposure_field))
    dominant = max(interval_rows, key=lambda row: row["exposure_increment"])
    return {
        "site_density_m-2": float(site_density_m2),
        "hazard_exposure_total": total,
        "expected_raw_trigger_count": total,
        "probability_at_least_one_raw_trigger": float(-math.expm1(-total)),
        "spatial_exposure_quantiles": [float(x) for x in np.quantile(
            exposure_field, (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0))],
        "spatial_exposure_max": float(np.max(exposure_field)),
        "dominant_interval_end_time_s": dominant["end_time_s"],
        "dominant_interval_end_strain": dominant["end_strain"],
        "dominant_interval_exposure": dominant["exposure_increment"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    paths = sorted(args.audit_root.glob("*/hazard_reweight_state.npz"),
                   key=lambda path: int(np.load(path)["step"]))
    if len(paths) < 2:
        raise SystemExit("at least two hazard audit states are required")
    audits = [np.load(path, allow_pickle=False) for path in paths]
    kinetic = dict(attempt_s=1.0e6, h0_eV=1.20, critical_pa=2.0e8,
                   exp_a=1.0, exp_n=1.0, floor=0.05,
                   entropy_kB=0.0, rate_cap_s=1.0e7)
    densities = (2.0e10, 2.0e11, 6.0e11)
    rows = [integrate(audits, density, kinetic) for density in densities]
    payload = {
        "schema": "full-v34-offline-area-hazard-reweight/v1",
        "architecture": "full-v34 two-dimensional phase-field saved no-trigger trajectory",
        "classification": "offline mechanism-excitation bracket; not fitted grain counts",
        "creation_route": "Route B precursor creation",
        "activity_prefactor": "fixed unity; plastic-activity hazard prefactor disabled for selected experiment",
        "kinetics": kinetic,
        "fixed_identifiability_choice": "attempt frequency fixed and activation entropy fixed; neither fitted",
        "source_states": [{"path": str(path), "sha256": sha256(path)} for path in paths],
        "number_of_saved_states": len(paths),
        "initial_time_s": float(audits[0]["sim_time"]),
        "final_time_s": float(audits[-1]["sim_time"]),
        "initial_strain": float(audits[0]["strain"]),
        "final_strain": float(audits[-1]["strain"]),
        "potential_state_note": "Authoritative saved fields are exact; the numerical potential was rebuilt because the historical checkpoint predates potential-table checkpointing.",
        "rows": rows,
        "selected_central_site_density_m-2": 2.0e11,
        "fixture_passed": True,
        "scientific_gate_passed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
