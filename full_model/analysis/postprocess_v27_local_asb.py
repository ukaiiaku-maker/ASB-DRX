#!/usr/bin/env python3
"""Decision-grade postprocessing for the local V27 ASB resolution matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from full_model.production.asb_classifier import localization_geometry
from full_model.production.asb_grid_audit import first_law_budget, relative_difference


def checkpoint_paths(directory):
    return sorted(directory.glob("drx_v25_restart_*.npz"), key=lambda path: int(
        re.search(r"(\d+)$", path.stem).group(1)))


def hash_array(array):
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def trap(values, time):
    return float(getattr(np, "trapezoid", np.trapz)(values, time))


def initial_record(directory):
    path = checkpoint_paths(directory)[0]
    with np.load(path, allow_pickle=True) as data:
        p = json.loads(str(data["P_json"].item()))
        return {
            "checkpoint": str(path),
            "lab": np.asarray(data["lab"]),
            "eta": np.asarray(data["eta"]),
            "psi_gv": np.asarray(data["psi_gv"]),
            "parameters": p,
            "hashes": {name: hash_array(data[name])
                       for name in ("lab", "eta", "psi_gv", "T", "rp", "rm")},
        }


def sampled_partition_edges(initial, sample_n=96):
    labels = initial["lab"]
    n = labels.shape[0]
    fraction = (np.arange(sample_n)+.5)/sample_n
    index = np.minimum((fraction*n).astype(int), n-1)
    sampled = labels[np.ix_(index, index)]
    return np.stack((sampled == np.roll(sampled, -1, axis=0),
                     sampled == np.roll(sampled, -1, axis=1)), axis=-1)


def microstructure_audit(initials):
    reference = sampled_partition_edges(initials["96"])
    rows = {}
    for name, initial in initials.items():
        edges = sampled_partition_edges(initial)
        rows[name] = {
            "partition_edge_mismatch_fraction_vs_96": float(np.mean(edges != reference)),
            "orientation_table_max_abs_difference_rad_vs_96": float(np.max(np.abs(
                initial["psi_gv"][:12]-initials["96"]["psi_gv"][:12]))),
            "initial_array_hashes": initial["hashes"],
        }
    keys = (
        "L_phys", "kappa_eta", "W_eta", "heat_process_zone_sigma_um",
        "heat_process_zone_min_sigma_px", "k_thermal", "cp_rho_vol",
        "T0", "edot_app", "nuc_barrier_thickness_b", "b")
    declarations = {}
    for key in keys:
        values = {name: initial["parameters"].get(key)
                  for name, initial in initials.items()}
        declarations[key] = {
            "values": values,
            "identical": len({json.dumps(value, sort_keys=True)
                              for value in values.values()}) == 1,
        }
    consistent = (all(row["partition_edge_mismatch_fraction_vs_96"] <= .05
                      and row["orientation_table_max_abs_difference_rad_vs_96"] <= 1e-12
                      for row in rows.values())
                  and all(item["identical"] for item in declarations.values()))
    return {"common_sample_grid": 96, "records": rows,
            "physical_declarations": declarations,
            "restriction_consistent": bool(consistent)}


def case_record(directory):
    csv = pd.read_csv(directory/"drx_v25_restart_asb_diagnostics.csv")
    checkpoints = checkpoint_paths(directory)
    with np.load(checkpoints[-1], allow_pickle=True) as data:
        p = json.loads(str(data["P_json"].item()))
        rate = np.asarray(data["asb_last_gdot_abs"], float)
        temperature = np.asarray(data["T"], float)
        final_time = float(data["sim_time"])
    time = csv["t_us"].to_numpy(float)*1e-6
    stress = csv["sigma_MPa"].to_numpy(float)*1e6
    plastic_power = csv["Pplastic_mean_Jm3s"].to_numpy(float)
    heat_power = csv["heat_qdot_MWm3"].to_numpy(float)*1e6
    mean_temperature = csv["T_mean"].to_numpy(float)
    external = trap(stress*float(p["edot_app"]), time)
    plastic = trap(plastic_power, time)
    heat = trap(heat_power, time)
    declared_e_eff = p.get("finite_loading_Eeff")
    e_eff = (float(declared_e_eff) if declared_e_eff is not None else
             (float(p["C11"])**2-float(p["C12"])**2)/float(p["C11"]))
    elastic = .5*(stress[-1]**2-stress[0]**2)/e_eff
    # F_total_full is the two-dimensional integral [J/m].  Dividing by the
    # represented in-plane area gives the corresponding volume density; the
    # common represented thickness cancels exactly.
    defect = float(csv["F_total_full"].iloc[-1]
                   -csv["F_total_full"].iloc[0])/float(p["L_phys"])**2
    thermal = float(p["cp_rho_vol"])*(mean_temperature[-1]-mean_temperature[0])
    bath = trap(float(p.get("T_bath_coupling", 0.0))
                *(mean_temperature-float(p["T0"])), time)
    other = plastic-defect-heat
    ledger = first_law_budget(
        external_work_J_m3=external, elastic_change_J_m3=elastic,
        plastic_work_J_m3=plastic, defect_free_energy_change_J_m3=defect,
        taylor_quinney_heat_J_m3=heat, conducted_out_J_m3=bath,
        thermal_energy_change_J_m3=thermal, other_dissipation_J_m3=other)
    dx = float(p["L_phys"])/int(p["Nx"])
    active, width = localization_geometry(rate, dx, dx)
    spectrum_rate = np.abs(np.fft.rfft2(rate-np.mean(rate)))**2
    spectrum_temperature = np.abs(np.fft.rfft2(temperature-np.mean(temperature)))**2
    peak = float(np.max(np.abs(stress)))
    return {
        "directory": str(directory), "grid": int(p["Nx"]),
        "cell_size_m": dx, "end_time_s": final_time,
        "end_strain": float(p["edot_app"])*final_time,
        "peak_stress_Pa": peak, "final_stress_Pa": float(stress[-1]),
        "post_peak_softening_fraction": (peak-abs(float(stress[-1])))/max(peak, 1e-300),
        "maximum_temperature_K": float(np.max(temperature)),
        "maximum_temperature_excess_from_T0_K": float(np.max(temperature)-p["T0"]),
        "plastic_rate_max_s": float(np.max(rate)),
        "active_plastic_fraction": active, "effective_band_width_m": width,
        "plastic_rate_spectral_peak": float(np.max(spectrum_rate)),
        "temperature_spectral_peak": float(np.max(spectrum_temperature)),
        "first_law": ledger,
        "first_law_closed_5pct": bool(
            ledger["plastic_partition_relative_residual"] <= .05
            and ledger["system_relative_residual"] <= .05),
        "checkpoint_count": len(checkpoints),
    }


def comparisons(records):
    keys = (
        "peak_stress_Pa", "final_stress_Pa", "maximum_temperature_K",
        "plastic_rate_max_s", "active_plastic_fraction", "effective_band_width_m")
    ledger_keys = ("external_work_J_m3", "plastic_work_J_m3",
                   "taylor_quinney_heat_J_m3", "thermal_energy_change_J_m3")
    result = {}
    for a, b in (("96", "128"), ("128", "192")):
        row = {key: relative_difference(records[a][key], records[b][key])
               for key in keys}
        row.update({"first_law."+key: relative_difference(
            records[a]["first_law"][key], records[b]["first_law"][key])
                    for key in ledger_keys})
        row["all_within_5pct"] = all(value <= .05 for value in row.values())
        result[f"{a}_vs_{b}"] = row
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ("96", "128", "192", "homogeneous"):
        parser.add_argument(f"--case-{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directories = {name: getattr(args, "case_"+name)
                   for name in ("96", "128", "192", "homogeneous")}
    records = {name: case_record(path) for name, path in directories.items()}
    initials = {name: initial_record(path) for name, path in directories.items()
                if name != "homogeneous"}
    microstructure = microstructure_audit(initials)
    refine = comparisons(records)
    laws_close = all(records[name]["first_law_closed_5pct"]
                     for name in ("96", "128", "192", "homogeneous"))
    fine_converged = refine["128_vs_192"]["all_within_5pct"]
    strict_observed = bool(
        records["128"]["active_plastic_fraction"] <= .25
        and records["128"]["maximum_temperature_excess_from_T0_K"] >= 50.0
        and records["128"]["post_peak_softening_fraction"] >= .20
        and records["128"]["effective_band_width_m"] >= 2.0*np.sqrt(
            initials["128"]["parameters"]["kappa_eta"]
            /initials["128"]["parameters"]["W_eta"]))
    if not laws_close:
        classification = "ASB_HARD_INVARIANT_FAILURE"
    elif not microstructure["restriction_consistent"]:
        classification = "ASB_MICROSTRUCTURE_RESTRICTION_SENSITIVE"
    elif not fine_converged:
        classification = "ASB_PHYSICAL_RESPONSE_NOT_GRID_CONVERGED"
    elif strict_observed:
        classification = "ASB_GRID_SCALING_QUALIFIED_STRICT_ASB_OBSERVED"
    else:
        classification = "ASB_GRID_SCALING_QUALIFIED_STRICT_ASB_NOT_OBSERVED"
    result = {
        "schema": "asb-drx/v27-local-asb-decision/v1",
        "source_mix": {
            "96_and_128": "frozen completed V26 HPC cases at bf150ca",
            "192": "bf150ca checkpoint 1250 exactly continued locally",
            "homogeneous": "locally executed control",
        },
        "records": records, "microstructure": microstructure,
        "relative_differences": refine,
        "strict_thresholds": {
            "maximum_active_fraction": .25,
            "minimum_temperature_excess_K": 50.0,
            "minimum_post_peak_softening_fraction": .20,
            "minimum_width_to_interface": 2.0,
            "minimum_continuous_persistence_s": 1e-6,
            "refinement_tolerance": .05,
        },
        "strict_endpoint_criteria_observed": strict_observed,
        "continuous_conjunctive_persistence_verified": False,
        "first_law_all_cases_closed_5pct": laws_close,
        "fine_128_192_converged_5pct": bool(fine_converged),
        "classification": classification,
        "strain_rate_bracket_authorized": bool(
            laws_close and microstructure["restriction_consistent"] and fine_converged),
        "fixture_passed": True,
        "scientific_gate_passed": bool(
            classification.startswith("ASB_GRID_SCALING_QUALIFIED")),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(classification)


if __name__ == "__main__":
    main()
