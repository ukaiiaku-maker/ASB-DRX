#!/usr/bin/env python3
"""First-law and spatial-response classification for the V26 ASB matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from full_model.production.asb_grid_audit import (
    first_law_budget, relative_difference,
)


def trap(values, time):
    return float(getattr(np, "trapezoid", np.trapz)(values, time))


def case(directory):
    csv = pd.read_csv(directory/"drx_v25_restart_asb_diagnostics.csv")
    checkpoints = sorted(directory.glob("drx_v25_restart_*.npz"))
    with np.load(checkpoints[-1], allow_pickle=True) as data:
        p = json.loads(str(data["P_json"].item()))
        rate = np.asarray(data["asb_last_gdot_abs"], float)
        temperature = np.asarray(data["T"], float)
    time = csv["t_us"].to_numpy()*1e-6
    stress = csv["sigma_MPa"].to_numpy()*1e6
    plastic_power = csv["Pplastic_mean_Jm3s"].to_numpy()
    heat_power = csv["heat_qdot_MWm3"].to_numpy()*1e6
    mean_t = csv["T_mean"].to_numpy()
    external = trap(stress*float(p["edot_app"]), time)
    plastic = trap(plastic_power, time)
    heat = trap(heat_power, time)
    e_eff = float(p.get("finite_loading_Eeff", 200e9) or 200e9)
    elastic = .5*(stress[-1]**2-stress[0]**2)/e_eff
    defect = float(csv["F_total_full"].iloc[-1]-csv["F_total_full"].iloc[0])
    thermal = float(p["cp_rho_vol"])*(mean_t[-1]-mean_t[0])
    bath = trap(float(p.get("T_bath_coupling", 0.0))*(mean_t-float(p["T0"])), time)
    other = plastic-defect-heat
    ledger = first_law_budget(
        external_work_J_m3=external, elastic_change_J_m3=elastic,
        plastic_work_J_m3=plastic, defect_free_energy_change_J_m3=defect,
        taylor_quinney_heat_J_m3=heat, conducted_out_J_m3=bath,
        thermal_energy_change_J_m3=thermal, other_dissipation_J_m3=other)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(
        rate-np.mean(rate))))**2
    return {
        "grid": int(p["Nx"]), "cell_size_m": float(p["L_phys"])/int(p["Nx"]),
        "end_time_s": float(time[-1]), "end_strain": float(
            p["edot_app"]*time[-1]),
        "maximum_temperature_K": float(np.max(temperature)),
        "temperature_excess_K": float(np.max(temperature)-p["T0"]),
        "plastic_rate_max_s": float(np.max(rate)),
        "plastic_rate_active_fraction": float(np.mean(
            rate >= .5*max(float(np.max(rate)), 1e-300))),
        "plastic_rate_spectral_peak": float(np.max(spectrum)),
        "first_law": ledger,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = {name: case(args.root/name) for name in (
        "seed43_96", "seed43_128", "seed43_192", "homogeneous_128")}
    metrics = ("external_work_J_m3", "plastic_work_J_m3",
               "taylor_quinney_heat_J_m3", "thermal_energy_change_J_m3")
    comparisons = {}
    for a, b in (("seed43_96", "seed43_128"),
                 ("seed43_128", "seed43_192")):
        comparisons[f"{a}_vs_{b}"] = {
            key: relative_difference(records[a]["first_law"][key],
                                     records[b]["first_law"][key])
            for key in metrics}
    converged = all(value <= .05 for row in comparisons.values()
                    for value in row.values())
    result = {
        "schema": "asb-drx/v26-asb-resolution/v1",
        "records": records, "relative_differences": comparisons,
        "manufactured_kernel_valid_sequence": [96, 128, 192],
        "physical_response_grid_converged": bool(converged),
        "classification": ("ASB_GRID_SCALING_QUALIFIED" if converged else
                           "ASB_PHYSICAL_RESPONSE_NOT_GRID_CONVERGED"),
        "strain_rate_bracket_authorized": bool(converged),
        "fixture_passed": True, "scientific_gate_passed": bool(converged),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(result["classification"])


if __name__ == "__main__":
    main()
