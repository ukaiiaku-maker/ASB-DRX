#!/usr/bin/env python3
"""Test conservative polarized-content absorption into a nascent wall.

This is a post-precursor mechanism discriminator.  It starts from the actual
reaction-disabled signed-transport checkpoint, enables every production
reaction plus the default-off polarized absorption channel, and holds total
strain fixed.  It does not allocate a grain or impose an orientation.
"""

from __future__ import annotations

from dataclasses import fields, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from full_model.analysis.run_v66_groma_reaction_survival import (
    _homogenize, _load_state, _run,
)
from full_model.analysis.run_v66_production_groma_tangent import (
    candidate_parameters,
)
from full_model.production.common_tensorial_wall import CommonWallDriving
from full_model.production.tensorial_nye import (
    bcc_four_family_systems, make_junction_topology,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT/"full_model"/"verification"/"v66_production_groma_loading"
          /"final_state.npz")
OUT = ROOT/"full_model"/"verification"/"v66_polarized_absorption_handoff"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _spatial(state, spacing_m):
    wall_total = np.sum(state.wall_plus_m2+state.wall_minus_m2, axis=2)
    wall_signed = state.wall_plus_m2-state.wall_minus_m2
    wall_polarization = np.sqrt(np.sum(wall_signed*wall_signed, axis=2))/np.maximum(
        wall_total, 1.0)
    theta_hat = np.fft.fftn(state.orientation_rad)
    kx = 2.0*np.pi*np.fft.fftfreq(state.orientation_rad.shape[0], d=spacing_m)
    ky = 2.0*np.pi*np.fft.fftfreq(state.orientation_rad.shape[1], d=spacing_m)
    gx = np.fft.ifftn(1j*kx[:, None]*theta_hat).real
    gy = np.fft.ifftn(1j*ky[None, :]*theta_hat).real
    gradient = np.hypot(gx, gy)
    correlation = (0.0 if np.std(wall_total) == 0.0 or np.std(gradient) == 0.0
                   else float(np.corrcoef(wall_total.ravel(), gradient.ravel())[0, 1]))
    spectrum = np.abs(np.fft.fftn(wall_total-np.mean(wall_total)))**2
    spectrum[0, 0] = 0.0
    peak = np.unravel_index(int(np.argmax(spectrum)), spectrum.shape)
    signed_peak = tuple(min(index, size-index) for index, size in zip(
        peak, spectrum.shape))
    magnitude = float(np.hypot(*signed_peak))
    wavelength = (None if magnitude == 0.0 else
                  spacing_m*state.orientation_rad.shape[0]/magnitude)
    return {
        "wall_density_mean_m2": float(np.mean(wall_total)),
        "wall_density_contrast": float(np.std(wall_total)/max(np.mean(wall_total), 1.0)),
        "wall_vector_polarization_mean": float(np.mean(wall_polarization)),
        "orientation_gradient_rms_m1": float(np.sqrt(np.mean(gradient*gradient))),
        "wall_density_orientation_gradient_correlation": correlation,
        "wall_structure_factor_peak_mode": list(map(int, signed_peak)),
        "wall_structure_factor_peak_wavelength_m": wavelength,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    patterned, shear = _load_state(SOURCE)
    homogeneous = _homogenize(patterned)
    systems = bcc_four_family_systems()
    topologies = (make_junction_topology(
        systems, 0, 1, 1, -1, line_tension_J_m=0.0),)
    parameters, gb2 = candidate_parameters(2.0e-7)
    parameters = replace(
        parameters, polarized_wall_absorption_enabled=True,
        wall_absorption_polarization_power=1.0)
    driving = CommonWallDriving(mean_strain=np.array(
        [[0.0, .5*shear], [.5*shear, 0.0]]))
    # The explicit accepted production step is positivity safe but only first
    # order in the reaction heat.  The microsecond levels below were fixed by
    # the preceding complete-reaction energy-convergence audit; larger steps
    # give spatially converged fields while over-integrating dissipation.
    duration = 1.0e-2
    patterned_coarse_state, patterned_coarse = _run(
        patterned, driving, systems, topologies, parameters, 2.0e-6,
        hold_time=duration, record_interval=2.0e-4)
    patterned_fine_state, patterned_fine = _run(
        patterned, driving, systems, topologies, parameters, 1.0e-6,
        hold_time=duration, record_interval=2.0e-4)
    homogeneous_state, homogeneous_case = _run(
        homogeneous, driving, systems, topologies, parameters, 2.0e-6,
        hold_time=duration, record_interval=2.0e-4)

    spatial = {
        "initial_patterned": _spatial(patterned, parameters.spacing_m),
        "patterned_coarse": _spatial(patterned_coarse_state, parameters.spacing_m),
        "patterned_fine": _spatial(patterned_fine_state, parameters.spacing_m),
        "homogeneous_control": _spatial(homogeneous_state, parameters.spacing_m),
    }
    convergence_keys = (
        "wall_density_mean_m2", "wall_density_contrast",
        "wall_vector_polarization_mean", "orientation_gradient_rms_m1")
    convergence = {key: abs(spatial["patterned_coarse"][key]
                            -spatial["patterned_fine"][key])
                   /max(abs(spatial["patterned_fine"][key]), 1.0)
                   for key in convergence_keys}

    checkpoint = OUT/"final_state.npz"
    np.savez_compressed(checkpoint, **{
        item.name: getattr(patterned_fine_state, item.name)
        for item in fields(patterned_fine_state)},
        physical_hold_time_s=np.asarray(duration),
        engineering_shear=np.asarray(shear))
    figure = OUT/"absorption_handoff.png"
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), constrained_layout=True)
    for label, case in (("patterned", patterned_fine),
                        ("homogeneous", homogeneous_case)):
        time = [row["time_s"] for row in case["records"]]
        axes[0].plot(time, [row["reservoirs"]["wall"]["mean_density_m2"]
                            for row in case["records"]], label=label)
        axes[1].plot(time, [row["reservoirs"]["wall"]["polarization_rms"]
                            for row in case["records"]], label=label)
        axes[2].plot(time, [row["wall_order_max"] for row in case["records"]],
                     label=label)
    axes[0].set(xlabel="hold time (s)", ylabel=r"mean wall density (m$^{-2}$)")
    axes[1].set(xlabel="hold time (s)", ylabel="wall polarization RMS")
    axes[2].set(xlabel="hold time (s)", ylabel="maximum wall order")
    axes[0].legend(frameon=False)
    fig.savefig(figure, dpi=180)
    plt.close(fig)

    energy_ok = patterned_fine["energy"]["relative_residual"] < 5.0e-4
    result = {
        "schema": "v66-polarized-absorption-handoff-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(SOURCE.relative_to(ROOT)),
        "source_sha256": _sha(SOURCE),
        "hold_time_s": duration,
        "fixed_engineering_shear": shear,
        "parameters": {
            "A": 1.25, "D": 0.5, "reduced_modulus_b2_J_m": gb2,
            "wall_barrier_eV": parameters.wall_barrier_eV,
            "wall_absorption_polarization_power": 1.0,
            "arrhenius_mechanics_unchanged": True,
        },
        "cases": {
            "patterned_coarse": patterned_coarse,
            "patterned_fine": patterned_fine,
            "homogeneous_control": homogeneous_case,
        },
        "spatial": spatial,
        "coarse_fine_relative_differences": convergence,
        "decision": {
            "five_percent_timestep_convergence": bool(max(convergence.values()) < .05),
            "energy_gate_passed": energy_ok,
            "polarization_selective_wall_absorption": bool(
                spatial["patterned_fine"]["wall_density_contrast"]
                > spatial["homogeneous_control"]["wall_density_contrast"]),
            "physical_wall_qualified": False,
            "recrystallization_qualified": False,
        },
        "claim_boundary": (
            "post-precursor fixed-strain absorption discriminator; no imposed "
            "orientation, phase label, boundary geometry, or material calibration"),
        "artifacts": {
            str(figure.relative_to(ROOT)): _sha(figure),
            str(checkpoint.relative_to(ROOT)): _sha(checkpoint),
        },
    }
    (OUT/"result.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"spatial": spatial, "convergence": convergence,
                      "energy": patterned_fine["energy"],
                      "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
