#!/usr/bin/env python3
"""Execute the V66 interaction-derived finite-wavelength discriminator."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from full_model.production.groma_patterning_oracle import (
    GromaPatterningParameters, fastest_linear_mode, rk4_step,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT/"full_model"/"verification"/"v66_groma_patterning_oracle"


def initial_state(n, length, seed=66):
    x = np.arange(n)*length/n
    rng = np.random.default_rng(seed)
    amplitudes = rng.normal(size=40)
    phases = rng.uniform(0.0, 2.0*np.pi, size=40)
    total = sum(amplitudes[m-1]*np.cos(
        2*np.pi*m*x/length+phases[m-1]) for m in range(1, 41))
    signed = sum(amplitudes[m-1]*np.sin(
        2*np.pi*m*x/length+phases[m-1]) for m in range(1, 41))
    total /= np.std(total)
    signed /= np.std(signed)
    return x, 1.0+1e-4*total, 1e-4*signed


def evolve(n, parameters, *, length=128.0, final_time=60.0, dt=.02):
    x, rho, kappa = initial_state(n, length)
    initial_rho = rho.copy()
    initial_kappa = kappa.copy()
    initial_sums = np.array([np.sum(rho), np.sum(kappa)])
    for _ in range(int(round(final_time/dt))):
        rho, kappa = rk4_step(
            rho, kappa, dt, length/n, parameters)
    power = np.abs(np.fft.rfft(rho-np.mean(rho)))**2
    power[0] = 0.0
    mode = int(np.argmax(power))
    return {
        "n": n, "x": x, "rho": rho, "kappa": kappa,
        "initial_rho": initial_rho, "initial_kappa": initial_kappa,
        "selected_mode": mode,
        "selected_wavelength": length/mode,
        "density_contrast": float(np.ptp(rho)),
        "maximum_polarization": float(np.max(np.abs(kappa)/rho)),
        "total_content_residual": float(np.sum(rho)-initial_sums[0]),
        "signed_content_residual": float(np.sum(kappa)-initial_sums[1]),
        "minimum_density": float(np.min(rho)),
        "power": power,
    }


def serial(run):
    return {key: value for key, value in run.items()
            if not isinstance(value, np.ndarray)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    unstable = GromaPatterningParameters(applied_stress=1.25)
    stable = GromaPatterningParameters(applied_stress=1.75)
    coarse = evolve(128, unstable)
    medium = evolve(256, unstable)
    fine = evolve(512, unstable, dt=.01)
    control = evolve(256, stable)
    prediction = fastest_linear_mode(unstable)
    kB_eV_K = 8.617333262145e-5
    temperature_predictions = {}
    for temperature in (800.0, 1000.0, 1200.0):
        candidate = GromaPatterningParameters(
            applied_stress=2.0, mobility_law="exp_floor",
            arrhenius_inverse_temperature=1.9/(kB_eV_K*temperature))
        temperature_predictions[f"{temperature:.0f}_K"] = (
            fastest_linear_mode(candidate))
    arrhenius = GromaPatterningParameters(
        applied_stress=2.0, mobility_law="exp_floor",
        arrhenius_inverse_temperature=1.9/(kB_eV_K*1000.0))
    arrhenius_control = GromaPatterningParameters(
        applied_stress=2.5, mobility_law="exp_floor",
        arrhenius_inverse_temperature=1.9/(kB_eV_K*1000.0))
    arrhenius_coarse = evolve(128, arrhenius, final_time=120.0, dt=.02)
    arrhenius_medium = evolve(256, arrhenius, final_time=120.0, dt=.02)
    arrhenius_fine = evolve(512, arrhenius, final_time=120.0, dt=.01)
    arrhenius_high = evolve(256, arrhenius_control, final_time=120.0, dt=.02)
    arrhenius_refinement = abs(
        arrhenius_fine["density_contrast"]
        -arrhenius_medium["density_contrast"]
    )/arrhenius_fine["density_contrast"]
    wavelength_error = abs(
        fine["selected_wavelength"]-prediction["wavelength"]
    )/prediction["wavelength"]
    amplitude_refinement = abs(
        fine["density_contrast"]-medium["density_contrast"]
    )/fine["density_contrast"]
    payload = {
        "schema": "v66-groma-patterning-oracle-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "citation": ("Groma, Zaiser, and Ispanovity, Physical Review B "
                         "93, 214110 (2016), arXiv:1601.07831"),
            "equations": "dimensionless one-dimensional reduction of Eq. 38",
            "production_status": "mechanism_oracle_not_promoted",
        },
        "parameters": {
            "unstable": unstable.__dict__, "stable_control": stable.__dict__,
            "domain_length_in_reference_dislocation_spacings": 128.0,
            "final_dimensionless_time": 60.0,
            "dt_by_grid": {"128": .02, "256": .02, "512": .01},
            "initial_noise_rms": 1e-4,
        },
        "linear_prediction": prediction,
        "arrhenius_prediction": {
            "immutable_shape": {
                "barrier_eV": 1.9, "floor_fraction": .05,
                "exp_a": 2.2, "exp_n": 2.5,
                "friction_over_critical_stress": 1.0,
                "applied_over_critical_stress": 2.0,
            },
            "by_temperature": temperature_predictions,
            "nonlinear_1000K_n128": serial(arrhenius_coarse),
            "nonlinear_1000K_n256": serial(arrhenius_medium),
            "nonlinear_1000K_n512": serial(arrhenius_fine),
            "nonlinear_1000K_high_stress_control_n256": serial(arrhenius_high),
            "density_contrast_refinement_difference": arrhenius_refinement,
        },
        "nonlinear": {
            "n128": serial(coarse), "n256": serial(medium),
            "n512": serial(fine),
            "stable_control_n256": serial(control),
            "selected_wavelength_relative_error": wavelength_error,
            "density_contrast_refinement_difference": amplitude_refinement,
        },
        "physical_scaling_examples": {
            "rho0_1e13_m2_wavelength_m": (
                fine["selected_wavelength"]/np.sqrt(1e13)),
            "rho0_1e14_m2_wavelength_m": (
                fine["selected_wavelength"]/np.sqrt(1e14)),
        },
        "decision": {
            "finite_wavelength_selected": bool(
                fine["selected_mode"] > 1 and wavelength_error < .10),
            "grid_refinement_below_5_percent": bool(
                amplitude_refinement < .05
                and medium["selected_mode"] == fine["selected_mode"]),
            "content_conserved": bool(
                abs(fine["total_content_residual"]) < 1e-10
                and abs(fine["signed_content_residual"]) < 1e-10),
            "stable_control_damped_relative_to_patterning": bool(
                control["density_contrast"] < fine["density_contrast"]),
            "exp_floor_arrhenius_instability_survives": bool(
                temperature_predictions["1000_K"]["unstable"]
                and arrhenius_fine["density_contrast"]
                >arrhenius_high["density_contrast"]
                and arrhenius_refinement < .05),
            "rx_or_wall_claimed": False,
            "reason": ("The correlation-derived conservative transport selects "
                       "a grid-convergent wavelength without a density double "
                       "well, but this 1-D single-slip density modulation is not "
                       "yet a closed GND-bearing wall or RX boundary."),
        },
    }
    (args.out/"result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)+"\n")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.7), constrained_layout=True)
    axes[0].plot(fine["x"], fine["rho"], label="patterning window")
    axes[0].plot(control["x"], control["rho"], label="high-stress control")
    axes[0].set_xlabel("position / reference dislocation spacing")
    axes[0].set_ylabel("total density / $\\rho_0$")
    axes[0].legend(frameon=False)
    modes = np.arange(len(fine["power"]))
    control_modes = np.arange(len(control["power"]))
    axes[1].semilogy(modes[1:], fine["power"][1:], label="patterning window")
    axes[1].semilogy(control_modes[1:], control["power"][1:],
                     label="high-stress control")
    axes[1].axvline(fine["selected_mode"], color="black", linewidth=.8,
                    linestyle="--", label="selected mode")
    axes[1].set_xlabel("periodic mode index")
    axes[1].set_ylabel("total-density spectral power")
    axes[1].legend(frameon=False)
    normalized_stress = np.linspace(1.01, 3.0, 160)
    for temperature in (800.0, 1000.0, 1200.0):
        growth = []
        for stress in normalized_stress:
            candidate = GromaPatterningParameters(
                applied_stress=float(stress), mobility_law="exp_floor",
                arrhenius_inverse_temperature=1.9/(kB_eV_K*temperature))
            growth.append(fastest_linear_mode(candidate)["growth_rate"])
        axes[2].plot(normalized_stress, growth, label=f"{temperature:.0f} K")
    axes[2].set_xlabel("applied / critical stress")
    axes[2].set_ylabel("fastest EXP-floor growth rate (scaled)")
    axes[2].legend(frameon=False)
    fig.savefig(args.out/"patterning_oracle.png", dpi=180)
    plt.close(fig)
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
