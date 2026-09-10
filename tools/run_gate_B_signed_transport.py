#!/usr/bin/env python3
"""Run, plot, and serialize substantive Gate B verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from asb_drx.signed_transport import (
    SignedTransportParameters,
    SignedTransportState,
    advance_pattern,
    dispersion_rate_s_inv,
    initialize_signed_state,
    kinematics_from_populations,
    nye_tensor_1d,
    reaction_step,
    signed_internal_stress_Pa,
    structure_factor_metrics,
    wall_diagnostics,
)


def relative_changes(coarse: dict, fine: dict) -> dict[str, float]:
    fields = (
        "wall_spacing_m", "gnd_rms_m2", "orientation_gradient_rms_m_inv",
        "spectral_peak_fraction",
    )
    return {name: abs(coarse[name] - fine[name]) / abs(fine[name]) for name in fields}


def make_total_band(parameters: SignedTransportParameters, n: int) -> SignedTransportState:
    x = np.arange(n) * parameters.domain_m / n
    total = parameters.reference_density_m2 * (
        1.0 + 0.4 * np.cos(2.0 * np.pi * 8.0 * x / parameters.domain_m)
    )
    plus = np.repeat((0.5 * total)[None, :], 4, axis=0)
    minus = plus.copy()
    slip, beta, rotation = kinematics_from_populations(
        plus, minus, parameters.domain_m, parameters.burgers_m
    )
    zeros = np.zeros_like(plus)
    return SignedTransportState(plus, minus, zeros, zeros, slip, beta, rotation)


def plot_fields(path: Path, state: SignedTransportState, parameters: SignedTransportParameters) -> None:
    x_um = np.arange(state.grid_points) * parameters.domain_m / state.grid_points * 1e6
    total = np.sum(state.mobile_plus_m2 + state.mobile_minus_m2, axis=0)
    kappa = state.mobile_plus_m2 - state.mobile_minus_m2
    alpha = nye_tensor_1d(state.plastic_distortion, parameters.domain_m)
    gnd = np.linalg.norm(alpha, axis=(1, 2)) / parameters.burgers_m
    stress = signed_internal_stress_Pa(state, parameters)
    angle = np.degrees(np.arccos(np.clip((np.trace(state.orientation, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)))
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
    axes[0, 0].plot(x_um, total / 1e15)
    axes[0, 0].set_ylabel(r"total $\rho$ ($10^{15}$ m$^{-2}$)")
    for family in range(4):
        axes[0, 1].plot(x_um, kappa[family] / 1e14, label=f"family {family}")
    axes[0, 1].set_ylabel(r"signed $\kappa^a$ ($10^{14}$ m$^{-2}$)")
    axes[0, 1].legend(ncol=2, fontsize=8)
    axes[1, 0].plot(x_um, gnd / 1e14)
    axes[1, 0].set_ylabel(r"Nye GND ($10^{14}$ m$^{-2}$)")
    for family in range(4):
        axes[1, 1].plot(x_um, state.slip[family])
    axes[1, 1].set_ylabel(r"plastic slip $\gamma^a$")
    axes[2, 0].plot(x_um, angle)
    axes[2, 0].set_ylabel("lattice rotation (deg)")
    for family in range(4):
        axes[2, 1].plot(x_um, stress[family] / 1e6)
    axes[2, 1].set_ylabel("internal stress (MPa)")
    for axis in axes[-1]: axis.set_xlabel("x (micrometers)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_structure(path: Path, initial: SignedTransportState, final: SignedTransportState, parameters: SignedTransportParameters) -> None:
    k = 2.0 * np.pi * np.fft.rfftfreq(final.grid_points, d=parameters.domain_m / final.grid_points)
    def power(state):
        value = np.sum(np.abs(np.fft.rfft(state.mobile_plus_m2 - state.mobile_minus_m2, axis=1)) ** 2, axis=0)
        return value / np.max(value[1:])
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.semilogy(k[1:] / 1e6, power(initial)[1:], label="initial")
    axis.semilogy(k[1:] / 1e6, power(final)[1:], label="nonlinear final")
    axis.axvline(parameters.selected_wavenumber_m_inv / 1e6, color="black", linestyle="--", label=r"analytical $k_*$")
    axis.set_xlabel(r"wavenumber ($10^6$ m$^{-1}$)")
    axis.set_ylabel("normalized structure factor")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_convergence(path: Path, records: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    grid = [r for r in records if r["dt_s"] == 0.002]
    axes[0].plot([r["grid_points"] for r in grid], [r["gnd_rms_m2"] / 1e13 for r in grid], "o-")
    axes[0].set_xlabel("grid points"); axes[0].set_ylabel(r"GND RMS ($10^{13}$ m$^{-2}$)")
    time = [r for r in records if r["grid_points"] == 128]
    axes[1].plot([r["dt_s"] for r in time], [r["spectral_peak_fraction"] for r in time], "o-")
    axes[1].set_xlabel("time step (s)"); axes[1].set_ylabel("structure-factor peak fraction")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def main(output: Path, plot_directory: Path) -> None:
    p = SignedTransportParameters()
    states = {}
    convergence = []
    for n, dt in ((64, 0.002), (128, 0.002), (256, 0.002), (128, 0.004)):
        final = advance_pattern(initialize_signed_state(n, p), 1.0, dt, p)
        states[(n, dt)] = final
        convergence.append({"grid_points": n, "dt_s": dt, **wall_diagnostics(final, p)})
    primary = states[(128, 0.002)]
    primary_diag = wall_diagnostics(primary, p)
    grid_changes = relative_changes(wall_diagnostics(states[(128, 0.002)], p), wall_diagnostics(states[(256, 0.002)], p))
    timestep_changes = relative_changes(wall_diagnostics(states[(128, 0.004)], p), primary_diag)

    total_band_diag = wall_diagnostics(make_total_band(p, 128), p)
    zero_gnd_diag = wall_diagnostics(initialize_signed_state(128, p, perturbation=0.0), p)
    reaction_state, ledger = reaction_step(
        initialize_signed_state(128, p), 0.02, p,
        multiplication_rate_m2_s=2.0e14,
        annihilation_rate_s_inv=3.0,
        locking_rate_s_inv=1.5,
    )
    del reaction_state
    k = 2.0 * np.pi * np.fft.rfftfreq(256, d=p.domain_m / 256)
    fastest = float(k[int(np.argmax(dispersion_rate_s_inv(k, p)))])

    checks = {
        "four_positive_and_negative_BCC_families": primary.mobile_plus_m2.shape[0] == 4 and np.min(primary.mobile_plus_m2) >= 0.0 and np.min(primary.mobile_minus_m2) >= 0.0,
        "finite_dispersion_peak_matches_declared_spacing": abs(fastest - p.selected_wavenumber_m_inv) / p.selected_wavenumber_m_inv < 1e-12,
        "nonlinear_GND_wall_forms": primary_diag["physical_wall"],
        "wall_spacing_matches_declared_finite_mode": abs(primary_diag["wall_spacing_m"] - p.selected_wavelength_m) / p.selected_wavelength_m < 0.01,
        "total_density_band_is_not_a_wall": total_band_diag["classification"] == "total_density_band_only" and not total_band_diag["physical_wall"],
        "balanced_high_density_is_not_a_wall": zero_gnd_diag["gnd_rms_m2"] == 0.0 and not zero_gnd_diag["physical_wall"],
        "reaction_mobile_balance_closes": abs(ledger.mobile_balance_residual_m_inv) < 1e-8 * ledger.mobile_before_m_inv,
        "reaction_junction_balance_closes": abs(ledger.junction_balance_residual_m_inv) < 1e-8 * ledger.mobile_before_m_inv,
        "reactions_do_not_create_net_Burgers_content": abs(ledger.net_burgers_change_m_inv) < 1e-8 * ledger.mobile_before_m_inv,
        "each_family_signed_content_is_conserved_by_pair_reactions": ledger.maximum_family_signed_change_m_inv < 1e-8 * ledger.mobile_before_m_inv,
        "vector_Burgers_reaction_residual_closes": ledger.burgers_vector_residual < 1e-8 * ledger.mobile_before_m_inv * p.burgers_m,
        "grid_refinement_below_five_percent": max(grid_changes.values()) < 0.05,
        "timestep_refinement_below_five_percent": max(timestep_changes.values()) < 0.05,
        "collective_DD_memory_disabled": p.collective_scale == 0.0,
        "grain_count_fixed_and_no_phase_state": primary.physical_grain_count == 1 and not hasattr(primary, "grain_labels") and not hasattr(primary, "phase_fields"),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    fixture_passed = all(checks[name] for name in (
        "four_positive_and_negative_BCC_families", "reaction_mobile_balance_closes",
        "reaction_junction_balance_closes", "reactions_do_not_create_net_Burgers_content",
        "each_family_signed_content_is_conserved_by_pair_reactions",
        "vector_Burgers_reaction_residual_closes",
        "collective_DD_memory_disabled", "grain_count_fixed_and_no_phase_state",
    ))
    scientific_gate_passed = fixture_passed and all(checks.values())
    if not scientific_gate_passed:
        raise RuntimeError(f"Gate B checks failed: {checks}")

    plot_directory.mkdir(parents=True, exist_ok=True)
    initial = initialize_signed_state(128, p)
    plot_fields(plot_directory / "gate_B_fields.png", primary, p)
    plot_structure(plot_directory / "gate_B_structure_factor.png", initial, primary, p)
    plot_convergence(plot_directory / "gate_B_convergence.png", convergence)

    result = {
        "schema": "asb-drx-gate-B-signed-transport/v1",
        "fixture_passed": fixture_passed,
        "scientific_gate_passed": scientific_gate_passed,
        "scope": "generic signed-dislocation wall precursor; not material calibration, LAGB, DRX, or ASB",
        "sources": [
            {"citation": "Groma, Zaiser & Ispanovity, Phys. Rev. B 93, 214110 (2016)", "doi": "10.1103/PhysRevB.93.214110"},
            {"citation": "Wu et al., Phys. Rev. B 98, 054110 (2018)", "doi": "10.1103/PhysRevB.98.054110"},
        ],
        "parameters_SI": {
            "domain_m": p.domain_m, "burgers_m": p.burgers_m,
            "reference_density_per_family_m2": p.reference_density_m2,
            "mobility_m2_s": p.mobility_m2_s, "local_drive": p.local_drive,
            "nonlinear_saturation": p.nonlinear_saturation,
            "gradient_length2_m2": p.gradient_length2_m2,
            "long_range_m_inv2": p.long_range_m_inv2,
            "internal_stress_scale_Pa": p.internal_stress_scale_Pa,
            "declared_selected_wavelength_m": p.selected_wavelength_m,
            "collective_scale": p.collective_scale,
        },
        "checks": checks,
        "primary_wall": primary_diag,
        "controls": {"total_density_band": total_band_diag, "balanced_zero_GND": zero_gnd_diag},
        "reaction_ledger_m_inv": ledger.__dict__,
        "dispersion": {"analytical_k_star_m_inv": p.selected_wavenumber_m_inv, "discrete_fastest_k_m_inv": fastest},
        "convergence": {"records": convergence, "grid_relative_changes": grid_changes, "timestep_relative_changes": timestep_changes},
        "fields": {
            "x_m": (np.arange(primary.grid_points) * p.domain_m / primary.grid_points).tolist(),
            "total_density_m2": np.sum(primary.mobile_plus_m2 + primary.mobile_minus_m2, axis=0).tolist(),
            "signed_density_by_family_m2": (primary.mobile_plus_m2 - primary.mobile_minus_m2).tolist(),
            "plastic_slip_by_family": primary.slip.tolist(),
            "nye_tensor_m_inv": nye_tensor_1d(primary.plastic_distortion, p.domain_m).tolist(),
            "orientation": primary.orientation.tolist(),
            "internal_stress_by_family_Pa": signed_internal_stress_Pa(primary, p).tolist(),
        },
        "plots": ["gate_B_fields.png", "gate_B_structure_factor.png", "gate_B_convergence.png"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("output/gate_B_signed_transport.json"))
    parser.add_argument("--plot-directory", type=Path, default=Path("output/gate_B_plots"))
    args = parser.parse_args()
    main(args.output, args.plot_directory)
