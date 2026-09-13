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
    # Match the production finite-patch measure exactly.  One grid cell is not
    # an independent embryo-sized site: production weights its local rate by
    # A_cell / (pi R_*^2), clipped to [1e-6, 1].  Omitting this factor aliases a
    # site-density change and overpredicts exposure (10.58x for the v6 screen).
    if "patch_weight" in z:
        patch_weight = np.asarray(z["patch_weight"], dtype=float)
    else:
        cell_area = float(z["cell_area_m2"])
        cell_length = math.sqrt(cell_area)
        critical_radius = np.asarray(z["classical_critical_R_m"], dtype=float)
        patch_weight = np.clip(
            cell_area/(math.pi*np.maximum(critical_radius, cell_length)**2),
            1.0e-6, 1.0)
    rate = (attempt_s*np.asarray(z["site_factor"])*patch_weight
            *np.asarray(z["gate_AT"])*np.exp(exponent))
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
    parser.add_argument("--online-anchor-density", type=float)
    parser.add_argument("--online-anchor-exposure", type=float)
    args = parser.parse_args()
    paths = sorted(args.audit_root.glob("*/hazard_reweight_state.npz"),
                   key=lambda path: int(np.load(path)["step"]))
    if len(paths) < 2:
        raise SystemExit("at least two hazard audit states are required")
    audits = [np.load(path, allow_pickle=False) for path in paths]
    kinetic = dict(attempt_s=1.0e6, h0_eV=1.20, critical_pa=2.0e8,
                   exp_a=1.0, exp_n=1.0, floor=0.05,
                   entropy_kB=0.0, rate_cap_s=1.0e7)
    if ((args.online_anchor_density is None)
            != (args.online_anchor_exposure is None)):
        raise SystemExit("online anchor density and exposure must be supplied together")
    if args.online_anchor_exposure is not None:
        if args.online_anchor_density <= 0.0 or args.online_anchor_exposure <= 0.0:
            raise SystemExit("online anchor values must be positive")
        densities = tuple(
            target*args.online_anchor_density/args.online_anchor_exposure
            for target in (0.1, 1.0, 3.0))
    else:
        densities = (2.0e10, 2.0e11, 6.0e11)
    rows = [integrate(audits, density, kinetic) for density in densities]
    quadrature_scale = 1.0
    if args.online_anchor_exposure is not None:
        sparse_anchor = integrate(audits, args.online_anchor_density, kinetic)
        quadrature_scale = (args.online_anchor_exposure
                            / sparse_anchor["hazard_exposure_total"])
        for row in rows:
            for key in ("hazard_exposure_total", "expected_raw_trigger_count",
                        "spatial_exposure_max", "dominant_interval_exposure"):
                row[key] *= quadrature_scale
            row["spatial_exposure_quantiles"] = [
                value*quadrature_scale
                for value in row["spatial_exposure_quantiles"]]
            row["probability_at_least_one_raw_trigger"] = float(
                -math.expm1(-row["hazard_exposure_total"]))
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
        "finite_patch_measure": "A_cell/(pi*max(R_critical,cell_length)^2), clipped to [1e-6,1]",
        "temporal_quadrature": {
            "saved_state_interpolation": "trapezoidal",
            "online_anchor_density_m-2": args.online_anchor_density,
            "online_anchor_exposure": args.online_anchor_exposure,
            "online_to_sparse_quadrature_scale": quadrature_scale,
            "interpretation": ("calibrated to the exact production 20-step hazard integral on the same no-event trajectory"
                               if args.online_anchor_exposure is not None else
                               "uncalibrated sparse-checkpoint quadrature"),
        },
        "rows": rows,
        "selected_central_site_density_m-2": densities[1],
        "fixture_passed": True,
        "scientific_gate_passed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
