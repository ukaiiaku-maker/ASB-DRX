#!/usr/bin/env python3
"""Build the V37 transport/localization decision from frozen V36 endpoints."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from full_model.analysis.postprocess_v32_asb_anchor import effective_support


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weighted_width(field: np.ndarray, dx: float) -> dict[str, float]:
    """Return coordinate-invariant second-moment widths of a nonnegative field."""
    value = np.maximum(np.asarray(field, dtype=float), 0.0)
    weight = value / max(float(value.sum()), 1e-300)
    ny, nx = value.shape
    yy, xx = np.indices((ny, nx), dtype=float)
    xx = (xx-(nx-1)/2.0)*dx
    yy = (yy-(ny-1)/2.0)*dx
    mx = float(np.sum(weight*xx)); my = float(np.sum(weight*yy))
    x = xx-mx; y = yy-my
    covariance = np.array([
        [np.sum(weight*x*x), np.sum(weight*x*y)],
        [np.sum(weight*x*y), np.sum(weight*y*y)]], dtype=float)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    return {
        "minor_rms_width_m": float(np.sqrt(eigenvalues[0])),
        "major_rms_width_m": float(np.sqrt(eigenvalues[1])),
        "minor_gaussian_fwhm_m": float(2.354820045*np.sqrt(eigenvalues[0])),
        "major_gaussian_fwhm_m": float(2.354820045*np.sqrt(eigenvalues[1])),
    }


def field_metrics(field: np.ndarray, dx: float) -> dict[str, object]:
    value = np.asarray(field, dtype=float)
    shifted = np.maximum(value-float(value.min()), 0.0)
    support = effective_support(np.maximum(value, 0.0))
    active = float(support["inverse_participation_fraction"])
    threshold = float(value.mean()+value.std())
    mask = value > threshold
    # Four-neighbour connected sizes without adding a SciPy runtime dependency.
    seen = np.zeros(mask.shape, dtype=bool); sizes: list[int] = []
    for start in zip(*np.nonzero(mask)):
        if seen[start]:
            continue
        stack = [start]; seen[start] = True; size = 0
        while stack:
            i, j = stack.pop(); size += 1
            for ni, nj in ((i-1, j), (i+1, j), (i, j-1), (i, j+1)):
                if (0 <= ni < mask.shape[0] and 0 <= nj < mask.shape[1]
                        and mask[ni, nj] and not seen[ni, nj]):
                    seen[ni, nj] = True; stack.append((ni, nj))
        sizes.append(size)
    return {
        "minimum": float(value.min()), "mean": float(value.mean()),
        "maximum": float(value.max()), "standard_deviation": float(value.std()),
        "peak_minus_mean": float(value.max()-value.mean()),
        "inverse_participation_fraction": float(active),
        "entropy_effective_fraction": float(
            support["entropy_effective_fraction"]),
        "effective_support_width_m": float(active*dx*value.shape[1]),
        "mean_plus_std_threshold": threshold,
        "threshold_area_fraction": float(mask.mean()),
        "connected_component_count": len(sizes),
        "largest_connected_area_fraction": (
            float(max(sizes)/value.size) if sizes else 0.0),
        "second_moment_widths": weighted_width(shifted, dx),
    }


def endpoint(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=True) as data:
        parameters = json.loads(str(data["P_json"].item()))
        temperature = np.asarray(data["T"], dtype=float)
        activity = np.asarray(data["asb_last_gdot_abs"], dtype=float)
        nx = int(parameters["Nx"]); length = float(parameters["L_phys"])
        dx = length/nx
        return {
            "path": str(path.resolve()), "sha256": sha256(path),
            "step": int(data["step"]), "physical_time_s": float(data["sim_time"]),
            "nominal_strain": ((int(data["step"])+1)
                               *float(parameters["dt_strain_step"])),
            "parameters": {
                "grid": nx, "domain_length_m": length,
                "conductivity_W_m_K": float(parameters["k_thermal"]),
                "volumetric_heat_capacity_J_m3_K": float(parameters["cp_rho_vol"]),
                "strain_rate_s": float(parameters["edot_app"]),
                "initial_temperature_K": float(parameters["T0"]),
                "particle_radius_m": float(parameters["v19_particle_radius_um"])*1e-6,
                "heat_process_zone_sigma_m": (
                    float(parameters["heat_process_zone_sigma_um"])*1e-6),
                "thermal_boundary": "periodic_insulated_no_bath",
            },
            "temperature": field_metrics(temperature, dx),
            "absolute_shear_rate": field_metrics(activity, dx),
            "activity_field_semantics": (
                "absolute shear-rate diagnostic; not independently ledgered plastic power"),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-checkpoint", type=Path, required=True)
    parser.add_argument("--conduction-checkpoint", type=Path, required=True)
    parser.add_argument("--thermal-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    full = endpoint(args.full_checkpoint)
    conduction = endpoint(args.conduction_checkpoint)
    p = conduction["parameters"]
    heat_capacity = float(p["volumetric_heat_capacity_J_m3_K"])
    conductivity = float(p["conductivity_W_m_K"])
    lengths = {
        "heat_process_zone_sigma": float(p["heat_process_zone_sigma_m"]),
        "heterogeneity_radius": float(p["particle_radius_m"]),
        "domain": float(p["domain_length_m"]),
    }
    diffusion = {
        name: heat_capacity*length*length/conductivity
        for name, length in lengths.items()}
    # Registered before V37 trajectory outcomes.  Zero conductivity is retained
    # only as the frozen V36 limiting ablation and is not in this physical screen.
    cases = [
        {"id": "rate_low", "T0_K": 900.0, "strain_rate_s": 1.0e4,
         "particle_radius_um": 0.75, "conductivity_W_m_K": 0.15},
        {"id": "rate_high", "T0_K": 900.0, "strain_rate_s": 1.0e5,
         "particle_radius_um": 0.75, "conductivity_W_m_K": 0.15},
        {"id": "temperature_low", "T0_K": 800.0, "strain_rate_s": 3.0e4,
         "particle_radius_um": 0.75, "conductivity_W_m_K": 0.15},
        {"id": "temperature_high", "T0_K": 1000.0, "strain_rate_s": 3.0e4,
         "particle_radius_um": 0.75, "conductivity_W_m_K": 0.15},
        {"id": "heterogeneity_short", "T0_K": 900.0, "strain_rate_s": 3.0e4,
         "particle_radius_um": 0.375, "conductivity_W_m_K": 0.15},
        {"id": "heterogeneity_long", "T0_K": 900.0, "strain_rate_s": 3.0e4,
         "particle_radius_um": 1.5, "conductivity_W_m_K": 0.15},
    ]
    manufactured_n = 128; dx = float(p["domain_length_m"])/manufactured_n
    yy, xx = np.indices((manufactured_n, manufactured_n), dtype=float)
    broad = np.ones((manufactured_n, manufactured_n), dtype=float)
    narrow = np.exp(-0.5*((yy-(manufactured_n-1)/2.0)*dx/(0.25e-6))**2)
    thermal = json.loads(args.thermal_evidence.read_text())
    result = {
        "schema": "asb-drx/v37/conduction-localization/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_v36_thermal_evidence": {
            "path": str(args.thermal_evidence.resolve()),
            "sha256": sha256(args.thermal_evidence),
            "classification": thermal.get("classification"),
        },
        "full_local_adiabatic": full, "finite_conduction": conduction,
        "transport_scales": {
            "formula": "tau_diff=C_vol*ell^2/k",
            "lengths_m": lengths, "diffusion_times_s": diffusion,
            "loading_time_s": conduction["physical_time_s"],
            "ratios_tau_diff_to_loading": {
                name: value/max(float(conduction["physical_time_s"]), 1e-300)
                for name, value in diffusion.items()},
        },
        "manufactured_metric_qualification": {
            "broad_uniform": field_metrics(broad, dx),
            "narrow_gaussian_sigma_0p25um": field_metrics(narrow, dx),
            "purpose": "distinguish broad support from a narrow band without changing frozen thresholds",
        },
        "registered_screen": {
            "fixed": {"grid": 128, "nominal_strain_target": 0.2501,
                      "thermal_boundary": "periodic_insulated_no_bath",
                      "seed": 43, "conductivity_W_m_K": 0.15},
            "uncertain_physical_inputs": cases,
            "zero_conductivity_role": "frozen V36 limiting ablation only",
            "selection_rule": (
                "promote strongest valid concentration, broad-heating negative, and intermediate control"),
        },
        "classification": "TRANSPORT_INFORMED_V37_SCREEN_REGISTERED",
        "strict_asb_claimed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
