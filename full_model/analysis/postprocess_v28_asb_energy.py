#!/usr/bin/env python3
"""Condition the V27 ASB energy and exact-common-time localization evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy import ndimage


ENERGY_COLUMNS = (
    "F_bulk", "F_r_grad", "F_eta_grad", "F_eta_barrier",
    "F_comp_alpha", "F_comp_GB",
)


def _step(path):
    return int(re.search(r"(\d+)$", path.stem).group(1))


def checkpoints(directory):
    return sorted(Path(directory).glob("drx_v25_restart_*.npz"), key=_step)


def checkpoint_time(path):
    with np.load(path, allow_pickle=True) as data:
        return float(data["sim_time"])


def interpolate_field(directory, name, target_time):
    paths = checkpoints(directory)
    times = np.asarray([checkpoint_time(path) for path in paths])
    upper = int(np.searchsorted(times, target_time, side="left"))
    upper = min(max(upper, 0), len(paths)-1)
    lower = max(upper-1, 0)
    if times[upper] < target_time and upper == len(paths)-1:
        lower = upper
    if lower == upper or times[upper] == times[lower]:
        weight = 0.0
    else:
        weight = float((target_time-times[lower])/(times[upper]-times[lower]))
    with np.load(paths[lower], allow_pickle=True) as first:
        a = np.asarray(first[name], float)
        parameters = json.loads(str(first["P_json"].item()))
    with np.load(paths[upper], allow_pickle=True) as second:
        b = np.asarray(second[name], float)
    return (1.0-weight)*a+weight*b, {
        "lower_checkpoint": str(paths[lower]), "upper_checkpoint": str(paths[upper]),
        "lower_time_s": float(times[lower]), "upper_time_s": float(times[upper]),
        "interpolation_weight": weight, "parameters": parameters,
    }


def localization_metrics(power, spacing_m):
    value = np.maximum(np.asarray(power, float), 0.0)
    total = float(np.sum(value))
    ncell = value.size
    if total <= 0.0:
        return {key: 0.0 for key in (
            "inverse_participation_effective_fraction", "entropy_effective_fraction",
            "minor_second_moment_m2", "connected_band_fwhm_m")}|{"connected_band_count": 0}
    probability = value/total
    ipr_fraction = 1.0/(ncell*float(np.sum(probability**2)))
    entropy_fraction = float(np.exp(-np.sum(
        probability*np.log(np.maximum(probability, 1e-300))))/ncell)
    x, y = np.indices(value.shape, dtype=float)
    x = (x-np.sum(probability*x))*spacing_m
    y = (y-np.sum(probability*y))*spacing_m
    covariance = np.array([[np.sum(probability*x*x), np.sum(probability*x*y)],
                           [np.sum(probability*x*y), np.sum(probability*y*y)]])
    eigval, eigvec = np.linalg.eigh(covariance)
    minor = eigvec[:, 0]
    coordinate = minor[0]*x+minor[1]*y
    bins = np.linspace(float(np.min(coordinate)), float(np.max(coordinate)),
                       max(value.shape)+1)
    profile, edges = np.histogram(coordinate, bins=bins, weights=value)
    centers = .5*(edges[:-1]+edges[1:])
    above = profile >= .5*np.max(profile)
    fwhm = float(centers[above][-1]-centers[above][0]+np.mean(np.diff(centers)))
    labels, count = ndimage.label(value >= .5*np.max(value))
    sizes = np.bincount(labels.ravel())[1:]
    count = int(np.count_nonzero(sizes >= 2))
    return {
        "inverse_participation_effective_fraction": float(ipr_fraction),
        "entropy_effective_fraction": entropy_fraction,
        "minor_second_moment_m2": float(eigval[0]),
        "connected_band_fwhm_m": fwhm,
        "connected_band_count": count,
    }


def interpolate_scalar(frame, column, time_s):
    return float(np.interp(time_s, frame["t_us"].to_numpy(float)*1e-6,
                           frame[column].to_numpy(float)))


def integrate_to(frame, values, end_time):
    time = frame["t_us"].to_numpy(float)*1e-6
    mask = time < end_time
    t = np.append(time[mask], end_time)
    v = np.append(np.asarray(values, float)[mask], np.interp(end_time, time, values))
    return float(getattr(np, "trapezoid", np.trapz)(v, t))


def case_record(directory, common_time):
    directory = Path(directory)
    frame = pd.read_csv(directory/"drx_v25_restart_asb_diagnostics.csv")
    rate, field_meta = interpolate_field(directory, "asb_last_gdot_abs", common_time)
    temperature, _ = interpolate_field(directory, "T", common_time)
    p = field_meta.pop("parameters")
    area = float(p["L_phys"])**2
    energy = {}
    for column in ENERGY_COLUMNS:
        initial = float(frame[column].iloc[0])/area
        final = interpolate_scalar(frame, column, common_time)/area
        energy[column] = {"initial_J_m3": initial, "final_J_m3": final,
                          "change_J_m3": final-initial}
    stress = frame["sigma_MPa"].to_numpy(float)*1e6
    plastic_power = frame["Pplastic_mean_Jm3s"].to_numpy(float)
    heat_power = frame["heat_qdot_MWm3"].to_numpy(float)*1e6
    external = integrate_to(frame, stress*float(p["edot_app"]), common_time)
    plastic = integrate_to(frame, plastic_power, common_time)
    heat = integrate_to(frame, heat_power, common_time)
    defect = float(sum(item["change_J_m3"] for item in energy.values()))
    ill_conditioning = max(abs(item["change_J_m3"]) for item in energy.values())/max(
        abs(external), 1.0)
    alpha = float(p["k_thermal"])/float(p["cp_rho_vol"])
    return {
        "directory": str(directory), "grid": int(p["Nx"]),
        "field_interpolation": field_meta,
        "energy_components": energy,
        "defect_energy_change_J_m3": defect,
        "external_work_J_m3": external, "plastic_work_J_m3": plastic,
        "deposited_heat_J_m3": heat,
        "maximum_internal_to_external_scale_ratio": ill_conditioning,
        "independent_other_dissipation_available": False,
        "localization": localization_metrics(rate, float(p["L_phys"])/int(p["Nx"])),
        "maximum_temperature_K": float(np.max(temperature)),
        "maximum_temperature_excess_from_T0_K": float(np.max(temperature)-p["T0"]),
        "physical_lengths_m": {
            "thermal_diffusion": float(np.sqrt(alpha*common_time)),
            "process_zone_sigma": float(p["heat_process_zone_sigma_um"])*1e-6,
            "phase_interface": float(np.sqrt(p["kappa_eta"]/p["W_eta"])),
            "cell_size": float(p["L_phys"])/int(p["Nx"]),
        },
    }


def relative(a, b):
    return abs(float(a)-float(b))/max(abs(float(a)), abs(float(b)), 1e-300)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v27", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old = json.loads(args.v27.read_text())
    common = float(old["common_matched_end_time_s"])
    records = {name: case_record(row["directory"], common)
               for name, row in old["records"].items()}
    hot = records["128"]; fine = records["192"]; control = records["homogeneous"]
    keys = ("inverse_participation_effective_fraction", "entropy_effective_fraction",
            "minor_second_moment_m2", "connected_band_fwhm_m")
    convergence = {key: relative(hot["localization"][key], fine["localization"][key])
                   for key in keys}
    convergence["temperature_excess_from_T0"] = relative(
        hot["maximum_temperature_excess_from_T0_K"],
        fine["maximum_temperature_excess_from_T0_K"])
    ad_iso_128 = hot["maximum_temperature_K"]-control["maximum_temperature_K"]
    conditioned = all(row["independent_other_dissipation_available"]
                      and row["maximum_internal_to_external_scale_ratio"] < 100.0
                      for row in records.values())
    result = {
        "schema": "asb-drx/v28-conditioned-energy-localization/v1",
        "common_time_s": common, "records": records,
        "matched_128_adiabatic_minus_homogeneous_maximum_K": ad_iso_128,
        "fine_128_192_relative_differences": convergence,
        "energy_ledger_physically_conditioned": bool(conditioned),
        "classification": ("ASB_ENERGY_LEDGER_PHYSICALLY_CONDITIONED"
                           if conditioned else "ASB_ENERGY_GAUGE_OR_DISSIPATION_FAILURE"),
        "fixture_passed": True, "scientific_gate_passed": bool(conditioned),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(result["classification"])


if __name__ == "__main__":
    main()
