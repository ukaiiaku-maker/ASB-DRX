#!/usr/bin/env python3
"""On-trajectory Fourier/JVP and reaction-exposure audit for V21."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PRODUCTION = Path(__file__).resolve().parents[1]/"production"
sys.path.insert(0, str(PRODUCTION))
from common_tensorial_wall import (  # noqa: E402
    CommonWallDriving, CommonWallParameters, CommonWallState, fourier_symbol,
)
from tensorial_nye import (  # noqa: E402
    JunctionTopology, bcc_four_family_systems, consistency_metrics,
    TensorialKinematicState,
)


def topology_from_json(record):
    return JunctionTopology(
        int(record["parent_a"]), int(record["parent_b"]),
        int(record["sign_a"]), int(record["sign_b"]),
        np.asarray(record["product_burgers_m"]),
        np.asarray(record["parent_line_directions"]),
        np.asarray(record["product_line_direction"]),
        float(record["product_line_multiplicity"]), record["character"],
        float(record["delta_free_energy_J_m"]),
    )


def fixed_eigenstrain(p, shape):
    nx, ny = shape
    result = np.zeros((nx, ny, 2, 2))
    if str(p.get("v19_mechanical_heterogeneity", "none")).lower() != "eigenstrain_particle":
        return result
    length = float(p["L_phys"]); spacing = length/nx
    centre = p.get("v19_particle_center_fraction", [0.5, 0.5])
    cx, cy = float(centre[0])*length, float(centre[1])*length
    x = np.arange(nx)[:, None]*spacing; y = np.arange(ny)[None, :]*spacing
    rx = np.minimum(np.abs(x-cx), length-np.abs(x-cx))
    ry = np.minimum(np.abs(y-cy), length-np.abs(y-cy))
    radius = float(p.get("v19_particle_radius_um", .75))*1e-6
    width = max(float(p.get("v19_particle_interface_um", .15))*1e-6, spacing)
    support = .5*(1.0-np.tanh((np.sqrt(rx*rx+ry*ry)-radius)/width))
    result[..., 0, 0] = float(p.get("v19_particle_eigenstrain_11", .004))*support
    result[..., 1, 1] = float(p.get("v19_particle_eigenstrain_22", -.004))*support
    shear = float(p.get("v19_particle_eigenstrain_12", 0.0))*support
    result[..., 0, 1] = result[..., 1, 0] = shear
    return result


def participation(vector, layout):
    weight = np.abs(vector)**2
    weight /= max(float(np.sum(weight)), 1e-300)
    result = {
        "density": 0.0, "wall_order": 0.0, "beta_p": 0.0,
        "orientation": 0.0, "temperature": 0.0, "junction": 0.0,
        "signed_polarization": 0.0, "unsigned_density": 0.0,
    }
    by_name = {}
    for index, item in enumerate(layout):
        name, family = item[0], item[1]
        if name in {
                "mobile_plus_m2", "mobile_minus_m2",
                "forest_plus_m2", "forest_minus_m2",
                "wall_plus_m2", "wall_minus_m2"}:
            result["density"] += float(weight[index])
            by_name[(name, family)] = vector[index]
        elif name == "junction_m2": result["junction"] += float(weight[index])
        elif name == "wall_order": result["wall_order"] += float(weight[index])
        elif name == "beta_p": result["beta_p"] += float(weight[index])
        elif name == "orientation_rad": result["orientation"] += float(weight[index])
        elif name == "temperature_K": result["temperature"] += float(weight[index])
    for reservoir in ("mobile", "forest", "wall"):
        for family in range(4):
            plus = by_name.get((reservoir+"_plus_m2", (family,)), 0.0)
            minus = by_name.get((reservoir+"_minus_m2", (family,)), 0.0)
            result["signed_polarization"] += float(abs(plus-minus)**2/2.0)
            result["unsigned_density"] += float(abs(plus+minus)**2/2.0)
    return result


def checkpoint(path, modes):
    with np.load(path, allow_pickle=True) as z:
        p = json.loads(str(z["P_json"].item()))
        parameters = CommonWallParameters(**json.loads(str(
            z["v21_common_parameters_json"].item())))
        topologies = tuple(topology_from_json(item) for item in json.loads(str(
            z["v21_topology_json"].item())))
        systems = bcc_four_family_systems(parameters.burgers_m)
        state = CommonWallState(
            np.asarray(z["rp"]), np.asarray(z["rm"]),
            np.asarray(z["rho_forest_plus"]), np.asarray(z["rho_forest_minus"]),
            np.asarray(z["rho_wall_plus"]), np.asarray(z["rho_wall_minus"]),
            np.asarray(z["v21_junction_m2"]), np.asarray(z["q_wall_v19"]),
            (np.asarray(z["v22_multi_hit_coordination"])
             if "v22_multi_hit_coordination" in z else
             np.zeros_like(np.asarray(z["q_wall_v19"]))),
            np.asarray(z["v20_slip"]), np.asarray(z["v20_beta_p"]),
            np.asarray(z["v20_alignment_m2"]), np.asarray(z["v20_family_nye_m1"]),
            np.asarray(z["psi_lat"]), np.asarray(z["T"]),
        )
        c11, c12 = float(p["C11"]), float(p["C12"])
        effective = (c11*c11-c12*c12)/c11; poisson = c12/c11
        beta2 = state.beta_p[..., :2, :2]
        epsp = .5*(beta2+np.swapaxes(beta2, -1, -2))
        stress = float(z["sigma_bar"])
        epmean = np.mean(epsp, axis=(0, 1))
        mean_strain = np.array([
            [stress/effective+epmean[0, 0], epmean[0, 1]],
            [epmean[0, 1], -poisson*stress/effective+epmean[1, 1]],
        ])
        driving = CommonWallDriving(
            mean_strain=mean_strain,
            fixed_eigenstrain=fixed_eigenstrain(p, state.wall_order.shape))
        spectra = []
        for mode in modes:
            symbol = fourier_symbol(
                state, driving, systems, topologies, parameters, mode,
                relative_step=2e-6)
            values = symbol["eigenvalues_s_inv"]
            vectors = symbol["right_eigenvectors"]
            order = np.argsort(values.real)[::-1]
            branches = []
            for rank in order[:8]:
                part = participation(vectors[:, rank], symbol["layout"])
                branches.append({
                    "real_growth_s_inv": float(values[rank].real),
                    "imaginary_s_inv": float(values[rank].imag),
                    "participation": part,
                })
            # A wall branch must carry signed first-order content.  Pure q or
            # unsigned-density ordering is explicitly not a LAGB precursor.
            wall_candidates = [item for item in branches
                               if item["participation"]["signed_polarization"]
                               >= 0.02]
            spectra.append({
                "mode": list(mode), "wavelength_m": symbol["wavelength_m"],
                "leading": branches[0],
                "leading_wall_candidate": (wall_candidates[0]
                                            if wall_candidates else None),
            })
        kin = TensorialKinematicState(
            state.slip, state.beta_p, state.alignment_m2, state.family_nye_m1)
        nye = consistency_metrics(
            kin, systems, state.orientation_rad, parameters.spacing_m)
        return {
            "path": str(path), "step": int(z["step"]),
            "time_s": float(z["sim_time"]), "strain": float(z["E_tot"][0, 0]),
            "stress_MPa": stress/1e6, "temperature_mean_K": float(np.mean(state.temperature_K)),
            "wall_order_mean": float(np.mean(state.wall_order)),
            "wall_order_std": float(np.std(state.wall_order)),
            "orientation_span_deg": float(np.ptp(state.orientation_rad)*180/np.pi),
            "nye_relative_residual": nye["relative_rms_residual"],
            "nye_divergence_relative_rms": nye["relative_divergence_rms"],
            "reaction_exposure": json.loads(str(z["v21_channel_exposure_json"].item())),
            "balance_ledger": json.loads(str(z["v21_balance_ledger_json"].item())),
            "spectra": spectra,
        }


def integrate(records, selector):
    times = np.asarray([r["time_s"] for r in records])
    values = np.asarray([max(float(selector(r)), 0.0) for r in records])
    return float(np.trapezoid(values, times))


def branch(directory):
    paths = sorted(Path(directory).glob("drx_v25_restart_*.npz"))
    if len(paths) < 3:
        raise ValueError("regular V21 checkpoints required")
    with np.load(paths[0], allow_pickle=True) as first:
        n = int(first["q_wall_v19"].shape[0])
    modes = [(0, 0)]+[(m, 0) for m in range(1, n//2)]+[(0, m) for m in range(1, n//2)]
    records = [checkpoint(path, modes) for path in paths]
    def leading_for(record, finite, wall):
        choices = [s for s in record["spectra"]
                   if ((s["mode"] != [0, 0]) if finite else (s["mode"] == [0, 0]))]
        key = "leading_wall_candidate" if wall else "leading"
        values = [s[key]["real_growth_s_inv"] for s in choices if s[key] is not None]
        return max(values) if values else 0.0
    exposures = {
        "uniform_leading_G": integrate(records, lambda r: leading_for(r, False, False)),
        "finite_leading_G": integrate(records, lambda r: leading_for(r, True, False)),
        "finite_wall_candidate_G": integrate(records, lambda r: leading_for(r, True, True)),
    }
    exposures["finite_over_uniform_advantage_G"] = integrate(
        records, lambda r: max(
            leading_for(r, True, True)-leading_for(r, False, False), 0.0))
    final = records[-1]
    if (exposures["finite_wall_candidate_G"] >= 8.0
            and exposures["finite_over_uniform_advantage_G"] >= 8.0):
        classification = "FINITE_MODE_EXPOSURE_ADEQUATE"
    elif final["wall_order_mean"] > .5 and exposures["uniform_leading_G"] >= 8.0:
        classification = "UNIFORM_ORDERING_ONLY"
    elif exposures["finite_leading_G"] > 0.0:
        classification = "UNSTABLE_BUT_INSUFFICIENT_HORIZON"
    else:
        classification = "MECHANISM_NOT_ENTERED"
    return {"modes": [list(x) for x in modes], "checkpoints": records,
            "cumulative_exposure": exposures, "classification": classification}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--homogeneous", type=Path, required=True)
    parser.add_argument("--particle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "schema": "asb-drx/v21-common-on-trajectory-audit/v1",
        "operator": "Fourier blocks assembled only from JVPs of the production residual",
        "branches": {
            "homogeneous_noise": branch(args.homogeneous),
            "mechanical_particle": branch(args.particle),
        },
        "fixture_passed": True, "scientific_gate_passed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({key: {"classification": value["classification"],
                                 **value["cumulative_exposure"]}
                      for key, value in result["branches"].items()}, indent=2))


if __name__ == "__main__":
    main()
