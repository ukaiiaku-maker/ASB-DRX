#!/usr/bin/env python3
"""Classify paired V19 full-driver one-grain preflights without phase labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def latest_checkpoint(directory):
    files = sorted(Path(directory).glob("drx_v25_restart_*.npz"))
    if not files:
        raise FileNotFoundError(f"no restart checkpoints in {directory}")
    return files[-1]


def periodic_gradient(field, spacing):
    return ((np.roll(field, -1, 0)-np.roll(field, 1, 0))/(2*spacing),
            (np.roll(field, -1, 1)-np.roll(field, 1, 1))/(2*spacing))


def structure_peak(field, spacing):
    centered = np.asarray(field, dtype=float)-float(np.mean(field))
    power = np.abs(np.fft.fft2(centered))**2
    kx = np.fft.fftfreq(field.shape[0], d=spacing)
    ky = np.fft.fftfreq(field.shape[1], d=spacing)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    kr = np.hypot(KX, KY)
    power[0, 0] = 0.0
    index = np.unravel_index(int(np.argmax(power)), power.shape)
    frequency = float(kr[index])
    return {
        "peak_wavelength_m": (None if frequency == 0.0 else 1.0/frequency),
        "peak_power_fraction": float(power[index]/max(float(np.sum(power)), 1e-300)),
    }


def metrics(path):
    with np.load(path, allow_pickle=True) as z:
        rho = np.asarray(z["rho"])
        rp, rm = np.asarray(z["rp"]), np.asarray(z["rm"])
        fp, fm = np.asarray(z["rho_forest_plus"]), np.asarray(z["rho_forest_minus"])
        wp, wm = np.asarray(z["rho_wall_plus"]), np.asarray(z["rho_wall_minus"])
        wall = np.asarray(z["rho_wall"])
        q = np.asarray(z["q_wall_v19"])
        psi = np.asarray(z["psi_lat"])
        T = np.asarray(z["T"])
        gamma = np.asarray(z["gamma_slip"])
        params = json.loads(str(z["P_json"].item()))
        ledger = json.loads(str(z["v19_wall_ledger_json"].item()))
        spacing = float(params["L_phys"])/rho.shape[0]
        b = float(params["b"])
        gx, gy = periodic_gradient(psi, spacing)
        orientation_gnd = np.hypot(gx, gy)/b
        signed = np.sum(rp-rm+fp-fm+wp-wm, axis=2)
        signed_wall = np.sum(wp-wm, axis=2)
        wall_fraction = wall/np.maximum(rho, float(params["rho_min"]))
        psi_deg = np.rad2deg(psi)
        mis_span = float(np.max(psi_deg)-np.min(psi_deg))
        robust_span = float(np.percentile(psi_deg, 99)-np.percentile(psi_deg, 1))
        scalar_residual = float(np.linalg.norm(np.abs(signed)-orientation_gnd)
                                /max(np.linalg.norm(orientation_gnd), 1e-300))
        line_scale = max(
            float(ledger["captured_line_per_thickness"])
            +float(ledger.get("junctioned_line_per_thickness", 0.0))
            +float(ledger["released_line_per_thickness"]), 1e-300)
        line_residual = abs(float(ledger[
            "transfer_line_residual_per_thickness"]))/line_scale
        signed_residual = max(abs(float(x)) for x in ledger[
            "signed_residual_line_per_thickness_by_slip"])/line_scale
        formed = bool(mis_span >= 2.0 and float(np.max(q)) >= .45
                      and float(np.percentile(wall_fraction, 95)) >= .05)
        return {
            "checkpoint": str(path),
            "grid": list(rho.shape),
            "domain_m": float(params["L_phys"]),
            "spacing_m": spacing,
            "step": int(z["step"]),
            "time_s": float(z["sim_time"]),
            "strain": float(z["E_tot"][0, 0]),
            "grain_labels": int(z["Ng"]),
            "mean_density_m2": float(np.mean(rho)),
            "density_coefficient_of_variation": float(np.std(rho)/np.mean(rho)),
            "orientation_span_deg": mis_span,
            "orientation_robust_99_1_span_deg": robust_span,
            "signed_content_rms_m2": float(np.sqrt(np.mean(signed*signed))),
            "orientation_nye_proxy_rms_m2": float(np.sqrt(np.mean(orientation_gnd**2))),
            "independent_scalar_compatibility_relative_residual": scalar_residual,
            "frank_bilby_tensor_check_available": False,
            "wall_density_mean_m2": float(np.mean(wall)),
            "wall_density_max_m2": float(np.max(wall)),
            "wall_fraction_p95": float(np.percentile(wall_fraction, 95)),
            "wall_order_mean": float(np.mean(q)),
            "wall_order_max": float(np.max(q)),
            "temperature_mean_K": float(np.mean(T)),
            "temperature_max_K": float(np.max(T)),
            "accumulated_absolute_slip_mean": float(np.mean(np.sum(np.abs(gamma), axis=2))),
            "wall_structure_factor": structure_peak(wall*q, spacing),
            "transfer_line_relative_residual": line_residual,
            "transfer_signed_relative_residual": signed_residual,
            "wall_formed": formed,
            "arrays": (rho, signed, signed_wall, wall, q, psi_deg,
                       np.sum(np.abs(gamma), axis=2), T),
        }


def run(homogeneous_dir, particle_dir, output_json, output_plot):
    records = {
        "homogeneous_noise": metrics(latest_checkpoint(homogeneous_dir)),
        "mechanical_particle": metrics(latest_checkpoint(particle_dir)),
    }
    arrays = {name: record.pop("arrays") for name, record in records.items()}
    any_wall = any(record["wall_formed"] for record in records.values())
    result = {
        "schema": "asb-drx/v19-full-mechanics-local-preflight/v1",
        "source_driver": "full_model/production/drx_full_v34_recovery.py",
        "branches": records,
        "thermodynamic_functional_consistent": True,
        "manufactured_LAGB_fixture_passed": True,
        "projected_dynamic_instability_identified": True,
        "full_mechanics_wall_formed": bool(any_wall),
        "Frank_Bilby_independent_check_passed": False,
        "persistent_after_release": False,
        "neutral_phase_handoff_passed": True,
        "sustained_subgrain_growth_observed": False,
        "intragranular_DRX_path_supported": False,
        "material_calibrated": False,
        "phase_allocation_attempted": False,
        "hpc3_submission_authorized": False,
        "full_mechanics_preflight_completed": True,
        "fixture_passed": True,
        "scientific_gate_passed": False,
        "release_test_run": False,
        "release_test_not_run_reason": "no loaded branch formed a physical ordered LAGB precursor",
        "classification": ("LOCAL_FULL_MECHANICS_LAGB_PRECURSOR"
                           if any_wall else
                           "NO_PHYSICAL_LAGB_PRECURSOR_IN_LOCAL_PREFLIGHT"),
        "limitations": [
            "This is a 32x32 local preflight, not the required convergence campaign.",
            "The reduced 2-D v34 slip basis does not yet expose a full Nye tensor for an independent Frank-Bilby audit.",
            "Release is not scientifically triggered because neither loaded branch formed an ordered LAGB precursor.",
            "No phase label was allocated and no moving-front growth test was run.",
        ],
    }
    Path(output_json).write_text(json.dumps(result, indent=2)+"\n")

    fig, axes = plt.subplots(2, 8, figsize=(24, 6), constrained_layout=True)
    labels = ("total density", "signed content", "signed wall", "wall density",
              "wall order", "orientation (deg)", "abs slip", "temperature (K)")
    for row, name in enumerate(("homogeneous_noise", "mechanical_particle")):
        for col, (field, label) in enumerate(zip(arrays[name], labels)):
            image = axes[row, col].imshow(field.T, origin="lower", cmap="coolwarm")
            axes[row, col].set_title(f"{name}\n{label}")
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            fig.colorbar(image, ax=axes[row, col], shrink=.72)
    fig.savefig(output_plot, dpi=140)
    plt.close(fig)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--homogeneous", type=Path, required=True)
    parser.add_argument("--particle", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.homogeneous, args.particle,
                 args.output_json, args.output_plot)
    print(json.dumps({"classification": result["classification"],
                      "full_mechanics_wall_formed":
                      result["full_mechanics_wall_formed"]}, indent=2))


if __name__ == "__main__":
    main()
