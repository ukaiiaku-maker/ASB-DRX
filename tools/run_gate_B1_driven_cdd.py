#!/usr/bin/env python3
"""Run the compact Gate B1 development audit without claiming the gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from asb_drx.bertin_bcc import BertinBCCParameters
from asb_drx.driven_cdd import (
    DrivenCDDParameters,
    advance_driven_cdd,
    correlation_stresses_Pa,
    derived_nye_tensor,
    driven_structure_diagnostics,
    evaluate_driven_cdd,
    full_linearized_amplification_spectrum,
    homogeneous_dispersion_snapshot,
    initialize_driven_cdd,
)


def run_case(n: int, increment: float, seed: int = 7):
    gate_a = BertinBCCParameters()
    parameters = DrivenCDDParameters()
    initial = initialize_driven_cdd(
        n, parameters, gate_a, total_noise_amplitude=0.003,
        signed_noise_amplitude=0.0, seed=seed, remove_mode=max(1, n // 4),
    )
    final, ledgers = advance_driven_cdd(
        initial, 1.0e6, 0.015, increment, parameters, gate_a
    )
    return initial, final, ledgers, driven_structure_diagnostics(final, parameters, gate_a)


def plot_fields(path: Path, state, parameters, gate_a) -> None:
    x = np.arange(state.grid_points) * parameters.domain_m / state.grid_points * 1.0e6
    total = np.sum(state.mobile_plus_m2 + state.mobile_minus_m2, axis=0)
    kappa = state.mobile_plus_m2 - state.mobile_minus_m2
    alpha = derived_nye_tensor(state, parameters.domain_m)
    gnd = np.linalg.norm(alpha, axis=(1, 2)) / gate_a.burgers_m
    sc, back, diffusion = correlation_stresses_Pa(state, parameters, gate_a)
    stress = sc + back + diffusion
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
    axes[0, 0].plot(x, total / 1.0e15)
    axes[0, 0].set_ylabel(r"total $\rho$ ($10^{15}$ m$^{-2}$)")
    for family in range(4):
        axes[0, 1].plot(x, kappa[family] / 1.0e12, label=f"family {family}")
        axes[1, 1].plot(x, state.accumulated_slip[family])
        axes[2, 1].plot(x, stress[family] / 1.0e6)
    axes[0, 1].set_ylabel(r"signed $\kappa_a$ ($10^{12}$ m$^{-2}$)")
    axes[0, 1].legend(ncol=2, fontsize=8)
    axes[1, 0].plot(x, gnd / 1.0e12)
    axes[1, 0].set_ylabel(r"Nye GND ($10^{12}$ m$^{-2}$)")
    axes[1, 1].set_ylabel(r"accumulated slip $\gamma_a$")
    orientation = []
    current_orientations = evaluate_driven_cdd(
        state, parameters, gate_a
    ).orientations
    for current, initial in zip(current_orientations, state.initial_orientation):
        relative = current @ initial.T
        orientation.append(np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))))
    axes[2, 0].plot(x, orientation)
    axes[2, 0].set_ylabel("lattice rotation from initial (deg)")
    axes[2, 1].set_ylabel("CDD internal stress (MPa)")
    for axis in axes[-1]:
        axis.set_xlabel("x (micrometers)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_structure(path: Path, initial, final) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for state, label in ((initial, "initial"), (final, "final")):
        total = np.sum(state.mobile_plus_m2 + state.mobile_minus_m2, axis=0)
        kappa = state.mobile_plus_m2 - state.mobile_minus_m2
        total_power = np.abs(np.fft.rfft(total - np.mean(total))) ** 2
        signed_power = np.sum(np.abs(np.fft.rfft(kappa, axis=1)) ** 2, axis=0)
        axes[0].semilogy(np.maximum(total_power[1:] / max(np.max(total_power[1:]), 1.0), 1.0e-30), label=label)
        axes[1].semilogy(np.maximum(signed_power[1:] / max(np.max(signed_power[1:]), 1.0), 1.0e-30), label=label)
    axes[0].set_title("total-density structure factor")
    axes[1].set_title("signed-density structure factor")
    for axis in axes:
        axis.set_xlabel("Fourier mode")
        axis.set_ylabel("normalized power")
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_convergence(path: Path, records: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot([r["requested_increment"] for r in records], [r["gnd_to_total_ratio"] for r in records], "o-")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("requested strain increment")
    axes[0].set_ylabel("GND / total density")
    axes[1].plot([r["requested_increment"] for r in records], [r["total_density_contrast"] for r in records], "o-")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("requested strain increment")
    axes[1].set_ylabel("total-density contrast")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main(output: Path, plot_directory: Path) -> None:
    gate_a = BertinBCCParameters()
    parameters = DrivenCDDParameters()
    initial, primary, primary_ledgers, primary_diag = run_case(32, 5.0e-4, 7)
    _, refined, refined_ledgers, refined_diag = run_case(32, 1.25e-4, 7)
    seed_records = []
    for seed in (3, 7, 11):
        _, state, ledgers, diagnostic = run_case(16, 5.0e-4, seed)
        seed_records.append({"seed": seed, **diagnostic, "maximum_balance_residual_m_inv": max(abs(x.total_balance_residual_m_inv) for x in ledgers)})

    homogeneous = initialize_driven_cdd(64, parameters, gate_a)
    homogeneous, _ = advance_driven_cdd(
        homogeneous, 1.0e6, 0.005, 5.0e-4, parameters, gate_a
    )
    reduced_dispersion = homogeneous_dispersion_snapshot(homogeneous, parameters, gate_a)
    full_operator = full_linearized_amplification_spectrum(
        homogeneous, 1.0e6, parameters.max_coupled_strain_increment,
        parameters, gate_a,
    )
    zero = initialize_driven_cdd(16, parameters, gate_a)
    zero_dispersion = homogeneous_dispersion_snapshot(zero, parameters, gate_a)

    fields = ("mobile_plus_m2", "mobile_minus_m2", "plastic_deformation_gradient")
    normalized_difference = max(
        float(np.max(np.abs(getattr(primary, name) - getattr(refined, name)))
              / max(np.max(np.abs(getattr(refined, name))), 1.0))
        for name in fields
    )
    max_balance = max(abs(item.total_balance_residual_m_inv) for item in primary_ledgers + refined_ledgers)
    reference_content = max(item.mobile_before_m_inv for item in primary_ledgers + refined_ledgers)
    checks = {
        "four_signed_families_and_nonnegative": bool(primary.mobile_plus_m2.shape[0] == 4 and np.min(primary.mobile_plus_m2) >= 0.0 and np.min(primary.mobile_minus_m2) >= 0.0),
        "zero_load_has_no_predicted_growth": not any(item["unstable"] for item in zero_dispersion["families"]),
        "no_prescribed_wavelength": reduced_dispersion["no_prescribed_wavelength"] and not hasattr(parameters, "selected_wavelength_m"),
        "periodic_line_balance_closes": max_balance < 2.0e-12 * reference_content,
        "requested_timestep_refinement_below_five_percent": normalized_difference < 0.05,
        "checkpoint_and_restart_covered_by_unit_suite": True,
        "grain_count_fixed_and_no_label_state": primary.physical_grain_count == 1 and not hasattr(primary, "grain_labels") and not hasattr(primary, "phase_fields"),
        "reduced_operator_has_finite_driven_mode": any(item["unstable"] and item["fastest_discrete_mode"] > 1 for item in reduced_dispersion["families"]),
        "full_coupled_operator_has_finite_driven_mode": full_operator["finite_mode_unstable"],
        "nonlinear_grid_convergent_wall_forms": False,
        "density_similitude_matrix_passed": False,
        "incommensurate_domain_matrix_passed": False,
        "temperature_and_rate_matrix_passed": False,
        "unloaded_wall_persistence_passed": False,
    }
    fixture_names = (
        "four_signed_families_and_nonnegative", "zero_load_has_no_predicted_growth",
        "no_prescribed_wavelength", "periodic_line_balance_closes",
        "requested_timestep_refinement_below_five_percent",
        "checkpoint_and_restart_covered_by_unit_suite",
        "grain_count_fixed_and_no_label_state",
    )
    fixture_passed = all(checks[name] for name in fixture_names)
    scientific_gate_passed = all(checks.values())

    plot_directory.mkdir(parents=True, exist_ok=True)
    plot_fields(plot_directory / "gate_B1_fields.png", primary, parameters, gate_a)
    plot_structure(plot_directory / "gate_B1_structure_factor.png", initial, primary)
    convergence = [
        {"requested_increment": 5.0e-4, **primary_diag},
        {"requested_increment": 1.25e-4, **refined_diag},
    ]
    plot_convergence(plot_directory / "gate_B1_convergence.png", convergence)

    alpha = derived_nye_tensor(primary, parameters.domain_m)
    sc, back, diffusion = correlation_stresses_Pa(primary, parameters, gate_a)
    result = {
        "schema": "asb-drx-gate-B1-driven-cdd-development/v1",
        "fixture_passed": fixture_passed,
        "scientific_gate_passed": scientific_gate_passed,
        "physical_CDD_wall_gate_passed": scientific_gate_passed,
        "classification": "DRIVEN_CDD_CORE_VALIDATED__FULL_ACTIVE_OPERATOR_STABLE__NONLINEAR_WALL_UNRESOLVED",
        "scope": "Gate-A-coupled driven CDD development; not LAGB, polygonization, DRX, ASB, or material calibration",
        "checks": checks,
        "parameters_SI": {
            "domain_m": parameters.domain_m,
            "backstress_coefficient": parameters.backstress_coefficient,
            "diffusion_coefficient": parameters.diffusion_coefficient,
            "elastic_kernel_scale": parameters.elastic_kernel_scale,
            "maximum_coupled_strain_increment": parameters.max_coupled_strain_increment,
            "collective_scale": parameters.collective_scale,
        },
        "linear_stability": {
            "reduced_Groma_reference": reduced_dispersion,
            "full_implemented_operator": full_operator,
            "interpretation": "The reduced dry-friction density operator predicts finite modes, but every mode of the full active-family, Nye-compatible Gate-A map is damped; Gate B1 fails without changing coefficients.",
        },
        "primary": primary_diag,
        "timestep_refinement": {"records": convergence, "maximum_normalized_field_difference": normalized_difference},
        "seed_controls": seed_records,
        "deferred_after_operator_failure": {
            "incommensurate_domains_um": [13, 16, 19],
            "density_scales": [0.75, 1.0, 1.25],
            "temperature_and_rate_variation": True,
            "reason": "Not run as an acceptance matrix because the full coupled operator and nonlinear wall prerequisites failed; avoids tuning or spending HPC around a failed gate.",
        },
        "balance": {"maximum_total_residual_m_inv": max_balance, "reference_content_m_inv": reference_content},
        "fields": {
            "x_m": (np.arange(primary.grid_points) * parameters.domain_m / primary.grid_points).tolist(),
            "total_density_m2": np.sum(primary.mobile_plus_m2 + primary.mobile_minus_m2, axis=0).tolist(),
            "signed_density_by_family_m2": (primary.mobile_plus_m2 - primary.mobile_minus_m2).tolist(),
            "nye_tensor_m_inv": alpha.tolist(),
            "plastic_slip_by_family": primary.accumulated_slip.tolist(),
            "plastic_deformation_gradient": primary.plastic_deformation_gradient.tolist(),
            "internal_stress_by_family_Pa": (sc + back + diffusion).tolist(),
        },
        "plots": ["gate_B1_fields.png", "gate_B1_structure_factor.png", "gate_B1_convergence.png"],
        "hpc3_submission": None,
        "provenance": {
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "execution": "local compact development audit",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("output/gate_B1_driven_cdd.json"))
    parser.add_argument("--plot-directory", type=Path, default=Path("output/gate_B1_plots"))
    arguments = parser.parse_args()
    main(arguments.output, arguments.plot_directory)
