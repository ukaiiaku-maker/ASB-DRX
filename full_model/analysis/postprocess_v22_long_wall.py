"""Post-process fetched V22 long-wall cases without allocating phase labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from full_model.production.common_tensorial_wall import (
    CommonWallParameters, CommonWallState, wall_polarization_invariants,
)
from full_model.production.tensorial_nye import (
    JunctionTopology, TensorialKinematicState, bcc_four_family_systems,
    consistency_metrics,
)


def topology_from_json(item):
    return JunctionTopology(
        int(item["parent_a"]), int(item["parent_b"]), int(item["sign_a"]),
        int(item["sign_b"]), np.asarray(item["product_burgers_m"]),
        np.asarray(item["parent_line_directions"]),
        np.asarray(item["product_line_direction"]),
        float(item["product_line_multiplicity"]), str(item["character"]),
        float(item["delta_free_energy_J_m"]))


def state_from(z):
    q = np.asarray(z["q_wall_v19"])
    return CommonWallState(
        np.asarray(z["rp"]), np.asarray(z["rm"]),
        np.asarray(z["rho_forest_plus"]), np.asarray(z["rho_forest_minus"]),
        np.asarray(z["rho_wall_plus"]), np.asarray(z["rho_wall_minus"]),
        np.asarray(z["v21_junction_m2"]), q,
        np.asarray(z["v22_multi_hit_coordination"]),
        np.asarray(z["v20_slip"]), np.asarray(z["v20_beta_p"]),
        np.asarray(z["v20_alignment_m2"]), np.asarray(z["v20_family_nye_m1"]),
        np.asarray(z["psi_lat"]), np.asarray(z["T"]))


def dominant_wavelength(field, spacing):
    centered = np.asarray(field)-np.mean(field)
    power = np.abs(np.fft.fftn(centered))**2
    power[0, 0] = 0.0
    index = np.unravel_index(np.argmax(power), power.shape)
    peak = float(power[index]); total = float(np.sum(power))
    fx = np.fft.fftfreq(field.shape[0], d=spacing)[index[0]]
    fy = np.fft.fftfreq(field.shape[1], d=spacing)[index[1]]
    frequency = float(np.hypot(fx, fy))
    return {"mode_index": [int(index[0]), int(index[1])],
            "wavelength_m": None if frequency == 0 else 1/frequency,
            "peak_power_fraction": peak/max(total, 1e-300)}


def reservoir_budget(initial, final, plus_name, minus_name):
    ip = np.asarray(getattr(initial, plus_name)); im = np.asarray(getattr(initial, minus_name))
    fp = np.asarray(getattr(final, plus_name)); fm = np.asarray(getattr(final, minus_name))
    return {
        "initial_unsigned_m2_by_family": np.mean(ip+im, axis=(0, 1)).tolist(),
        "final_unsigned_m2_by_family": np.mean(fp+fm, axis=(0, 1)).tolist(),
        "delta_unsigned_m2_by_family": np.mean(fp+fm-ip-im, axis=(0, 1)).tolist(),
        "initial_signed_m2_by_family": np.mean(ip-im, axis=(0, 1)).tolist(),
        "final_signed_m2_by_family": np.mean(fp-fm, axis=(0, 1)).tolist(),
        "delta_signed_m2_by_family": np.mean(fp-fm-ip+im, axis=(0, 1)).tolist(),
    }


def summarize_case(case, plot_dir):
    checkpoints = sorted(case.glob("drx_v25_restart_*.npz"))
    if len(checkpoints) < 2:
        return {"case": case.name, "complete_enough_to_classify": False,
                "checkpoint_count": len(checkpoints)}
    with np.load(checkpoints[0], allow_pickle=True) as z0, np.load(
            checkpoints[-1], allow_pickle=True) as z:
        initial = state_from(z0); final = state_from(z)
        pjson = json.loads(str(z["P_json"].item()))
        parameters = CommonWallParameters(**json.loads(str(
            z["v21_common_parameters_json"].item())))
        topology = tuple(topology_from_json(item) for item in json.loads(str(
            z["v21_topology_json"].item())))
        systems = bcc_four_family_systems(parameters.burgers_m)
        invariants = wall_polarization_invariants(final, systems, parameters)
        tensorial = TensorialKinematicState(
            final.slip, final.beta_p, final.alignment_m2, final.family_nye_m1)
        nye = consistency_metrics(
            tensorial, systems, final.orientation_rad, parameters.spacing_m)
        alpha_norm = np.linalg.norm(nye["alpha_rho_m1"], axis=(-2, -1))
        orientation_deg = np.rad2deg(
            final.orientation_rad-np.mean(final.orientation_rad))
        wall_signed = np.sum(final.wall_plus_m2-final.wall_minus_m2, axis=2)
        structure = dominant_wavelength(alpha_norm, parameters.spacing_m)
        strain = float(np.asarray(z["E_tot"])[0, 0])
        grain_count = int(np.asarray(z["Ng"]))
        ledger = json.loads(str(z["v21_balance_ledger_json"].item()))
        exposure = json.loads(str(z["v21_channel_exposure_json"].item()))
        summary = {
            "case": case.name, "complete_enough_to_classify": True,
            "initial_checkpoint": checkpoints[0].name,
            "final_checkpoint": checkpoints[-1].name,
            "checkpoint_count": len(checkpoints), "final_step": int(z["step"]),
            "physical_time_s": float(z["sim_time"]), "axial_strain": strain,
            "grain_count": grain_count, "no_grain_allocation": grain_count == 1,
            "wall_budget": reservoir_budget(
                initial, final, "wall_plus_m2", "wall_minus_m2"),
            "wall_order": {"mean": float(np.mean(final.wall_order)),
                           "maximum": float(np.max(final.wall_order)),
                           "standard_deviation": float(np.std(final.wall_order))},
            "wall_polarization": {
                "mean": float(np.mean(invariants["wall_polarization"])),
                "maximum": float(np.max(invariants["wall_polarization"])),
                "gate_mean": float(np.mean(invariants["wall_gate"])),
                "gate_maximum": float(np.max(invariants["wall_gate"]))},
            "multi_hit": {"mean": float(np.mean(final.multi_hit_coordination)),
                          "maximum": float(np.max(final.multi_hit_coordination))},
            "nye": {"beta_vs_family_relative_rms": nye["relative_rms_residual"],
                    "line_divergence_relative_rms": nye["relative_divergence_rms"],
                    "rms_m1": float(np.sqrt(np.mean(alpha_norm**2))),
                    "maximum_m1": float(np.max(alpha_norm))},
            "orientation": {
                "span_deg": float(np.ptp(orientation_deg)),
                "p95_minus_p05_deg": float(np.percentile(orientation_deg, 95)
                                             -np.percentile(orientation_deg, 5)),
                "gradient_rms_deg_m": float(np.sqrt(np.mean(sum(
                    g*g for g in np.gradient(orientation_deg,
                                              parameters.spacing_m)))))},
            "structure_factor": structure,
            "cumulative_channel_exposure": exposure,
            "balance_ledger": ledger,
            "fixture_passed": bool(
                grain_count == 1 and nye["relative_rms_residual"] < 1e-10
                and nye["relative_divergence_rms"] < 1e-10),
            "scientific_gate_passed": False,
            "classification": "PENDING_RELEASE_AND_GRID_DOMAIN_COMPARISON",
        }
        plot_dir.mkdir(parents=True, exist_ok=True)
        panels = ((np.sum(final.wall_plus_m2+final.wall_minus_m2, axis=2)/1e14,
                   "unsigned wall", "viridis"),
                  (wall_signed/1e14, "signed wall", "coolwarm"),
                  (alpha_norm/1e5, "Nye norm", "magma"),
                  (final.wall_order, "wall order", "viridis"),
                  (invariants["wall_polarization"], "polarization", "viridis"),
                  (orientation_deg, "orientation (deg)", "coolwarm"),
                  (final.multi_hit_coordination, "multi-hit", "viridis"),
                  (np.log10(np.abs(np.fft.fftshift(np.fft.fft2(
                      alpha_norm-alpha_norm.mean())))**2+1),
                   "log Nye structure factor", "magma"))
        fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
        for axis, (field, title, cmap) in zip(axes.flat, panels):
            image = axis.imshow(field.T, origin="lower", cmap=cmap)
            axis.set_title(title); axis.set_xticks([]); axis.set_yticks([])
            fig.colorbar(image, ax=axis, shrink=.72)
        fig.suptitle(f"{case.name}: strain={strain:.4f}, step={int(z['step'])}")
        fig.savefig(plot_dir/f"{case.name}.png", dpi=170); plt.close(fig)
        return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot-dir", type=Path, required=True)
    args = parser.parse_args()
    cases = [summarize_case(path, args.plot_dir)
             for path in sorted(args.cases.iterdir()) if path.is_dir()]
    result = {"schema": "asb-drx/v22-long-wall-postprocess/v1",
              "cases": cases,
              "fixture_passed": bool(cases and all(
                  case.get("fixture_passed", False) for case in cases)),
              "scientific_gate_passed": False,
              "classification": "PENDING_CASEWISE_SCIENTIFIC_DECISION"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"case_count": len(cases),
                      "fixture_passed": result["fixture_passed"]}, indent=2))


if __name__ == "__main__":
    main()
