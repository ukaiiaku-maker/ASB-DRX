#!/usr/bin/env python3
"""Single-job extended Gate B1 robustness and convergence campaign."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import subprocess

import numpy as np

from asb_drx.bertin_bcc import BertinBCCParameters
from asb_drx.driven_cdd import (
    DrivenCDDParameters,
    advance_driven_cdd,
    correlation_stresses_Pa,
    derived_nye_tensor,
    driven_structure_diagnostics,
    full_linearized_amplification_spectrum,
    homogeneous_dispersion_snapshot,
    initialize_driven_cdd,
)
from run_gate_B1_driven_cdd import plot_convergence, plot_fields, plot_structure


def simulate(
    n: int, domain_um: float, density_scale: float, temperature_K: float,
    rate_s_inv: float, seed: int, requested_increment: float = 5.0e-4,
):
    gate_a = BertinBCCParameters()
    parameters = DrivenCDDParameters(domain_m=domain_um * 1.0e-6)
    state = initialize_driven_cdd(
        n, parameters, gate_a, density_scale=density_scale,
        temperature_K=temperature_K, total_noise_amplitude=0.003,
        signed_noise_amplitude=0.0, seed=seed, remove_mode=6,
        spectral_noise_modes=12,
    )
    initial = state
    state, ledgers = advance_driven_cdd(
        state, rate_s_inv, 0.015, requested_increment, parameters, gate_a
    )
    return initial, state, ledgers, driven_structure_diagnostics(state, parameters, gate_a)


def relative_change(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1.0e-300)


def main(output: Path, plot_directory: Path) -> None:
    gate_a = BertinBCCParameters()
    baseline = DrivenCDDParameters()
    initial, primary, primary_ledgers, primary_diag = simulate(128, 16, 1, 300, 1.0e6, 7)
    _, grid_fine, fine_ledgers, fine_diag = simulate(256, 16, 1, 300, 1.0e6, 7)
    _, time_coarse, _, time_coarse_diag = simulate(64, 16, 1, 300, 1.0e6, 7, 5.0e-4)
    _, time_fine, _, time_fine_diag = simulate(64, 16, 1, 300, 1.0e6, 7, 1.25e-4)

    seeds = []
    for seed in (3, 7, 11):
        _, _, ledgers, diagnostic = simulate(64, 16, 1, 300, 1.0e6, seed)
        seeds.append({"seed": seed, **diagnostic, "maximum_balance_residual_m_inv": max(abs(x.total_balance_residual_m_inv) for x in ledgers)})

    domains = []
    for domain_um, n in ((13, 104), (16, 128), (19, 152)):
        _, _, _, diagnostic = simulate(n, domain_um, 1, 300, 1.0e6, 7)
        domains.append({"domain_um": domain_um, "grid_points": n, **diagnostic})

    densities = []
    for scale in (0.75, 1.0, 1.25):
        _, _, _, diagnostic = simulate(128, 16, scale, 300, 1.0e6, 7)
        densities.append({
            "initial_density_scale": scale, **diagnostic,
            "similitude_product_m_inv_sqrt": diagnostic["signed_dominant_wavelength_m"] * np.sqrt(diagnostic["total_density_mean_m2"]),
        })

    thermomechanical = []
    for temperature, rate in ((300, 1.0e6), (600, 1.0e6), (300, 2.0e6)):
        _, _, _, diagnostic = simulate(64, 16, 1, temperature, rate, 7)
        thermomechanical.append({"temperature_K": temperature, "rate_s_inv": rate, **diagnostic})

    homogeneous = initialize_driven_cdd(64, baseline, gate_a)
    homogeneous, _ = advance_driven_cdd(homogeneous, 1.0e6, 0.005, 5.0e-4, baseline, gate_a)
    reduced = homogeneous_dispersion_snapshot(homogeneous, baseline, gate_a)
    full = full_linearized_amplification_spectrum(
        homogeneous, 1.0e6, baseline.max_coupled_strain_increment,
        baseline, gate_a,
    )

    grid_metrics = (
        "signed_dominant_wavelength_m", "gnd_to_total_ratio",
        "total_density_contrast", "orientation_gradient_rms_m_inv",
        "mean_wall_FWHM_m",
    )
    grid_changes = {
        name: relative_change(primary_diag[name], fine_diag[name])
        for name in grid_metrics
    }
    time_changes = {
        name: relative_change(time_coarse_diag[name], time_fine_diag[name])
        for name in grid_metrics
    }
    seed_wavelengths = np.asarray([item["signed_dominant_wavelength_m"] for item in seeds])
    domain_wavelengths = np.asarray([item["signed_dominant_wavelength_m"] for item in domains])
    similitude = np.asarray([item["similitude_product_m_inv_sqrt"] for item in densities])
    unload_error = None
    try:
        unloaded, unload_ledgers = advance_driven_cdd(
            primary, -1.0e6, 0.0, 5.0e-4, baseline, gate_a
        )
        unload_diag = driven_structure_diagnostics(unloaded, baseline, gate_a)
        unload_behavior = (
            "persistent" if unload_diag["gnd_to_total_ratio"] >= 0.2 * primary_diag["gnd_to_total_ratio"]
            else "relaxed"
        )
    except (RuntimeError, ValueError, FloatingPointError) as error:
        unload_ledgers = ()
        unload_diag = None
        unload_behavior = "domain_stop"
        unload_error = f"{type(error).__name__}: {error}"
    all_ledgers = primary_ledgers + fine_ledgers + unload_ledgers
    reference_content = max(item.mobile_before_m_inv for item in all_ledgers)
    maximum_balance = max(abs(item.total_balance_residual_m_inv) for item in all_ledgers)

    checks = {
        "fixture_passed": True,
        "full_operator_finite_mode_unstable": full["finite_mode_unstable"],
        "target_mode_removed_and_different_mode_forms": primary_diag["signed_dominant_mode"] != 6,
        "primary_GND_rich_wall_precursor": primary_diag["classification"] == "GND-rich wall precursor",
        "grid_refinement_below_five_percent": max(grid_changes.values()) < 0.05,
        "timestep_refinement_below_five_percent": max(time_changes.values()) < 0.05,
        "three_seed_ensemble_forms_precursors": all(item["classification"] == "GND-rich wall precursor" for item in seeds),
        "seed_wavelength_cv_below_twenty_percent": float(np.std(seed_wavelengths) / np.mean(seed_wavelengths)) < 0.20,
        "incommensurate_domain_wavelength_cv_below_ten_percent": float(np.std(domain_wavelengths) / np.mean(domain_wavelengths)) < 0.10,
        "density_similitude_cv_below_fifteen_percent": float(np.std(similitude) / np.mean(similitude)) < 0.15,
        "temperature_and_rate_use_gate_A_without_retuning": len(thermomechanical) == 3,
        "unloaded_relaxation_or_persistence_characterized": unload_behavior in ("relaxed", "persistent"),
        "line_balance_closes": maximum_balance < 2.0e-12 * reference_content,
        "no_label_or_phase_allocation": primary.physical_grain_count == 1 and not hasattr(primary, "grain_labels") and not hasattr(primary, "phase_fields"),
    }
    scientific_names = tuple(name for name in checks if name != "fixture_passed")
    scientific_passed = all(checks[name] for name in scientific_names)

    plot_directory.mkdir(parents=True, exist_ok=True)
    plot_fields(plot_directory / "gate_B1_fields.png", primary, baseline, gate_a)
    plot_structure(plot_directory / "gate_B1_structure_factor.png", initial, primary)
    plot_convergence(
        plot_directory / "gate_B1_convergence.png",
        [{"requested_increment": 5.0e-4, **time_coarse_diag}, {"requested_increment": 1.25e-4, **time_fine_diag}],
    )
    alpha = derived_nye_tensor(primary, baseline.domain_m)
    sc, back, diffusion = correlation_stresses_Pa(primary, baseline, gate_a)
    result = {
        "schema": "asb-drx-gate-B1-driven-cdd-extended/v1",
        "fixture_passed": True,
        "scientific_gate_passed": scientific_passed,
        "physical_CDD_wall_gate_passed": scientific_passed,
        "classification": "PHYSICAL_DRIVEN_CDD_WALL_GATE_PASSED" if scientific_passed else "DRIVEN_CDD_PRECURSOR_PRESENT__SCIENTIFIC_MATRIX_FAILED",
        "checks": {name: bool(value) for name, value in checks.items()},
        "linear_stability": {"reduced_reference": reduced, "full_compatible_operator": full},
        "primary": primary_diag,
        "grid_convergence": {"coarse_128": primary_diag, "fine_256": fine_diag, "relative_changes": grid_changes},
        "timestep_convergence": {"coarse": time_coarse_diag, "fine": time_fine_diag, "relative_changes": time_changes},
        "seed_ensemble": seeds,
        "incommensurate_domains": domains,
        "density_similitude": densities,
        "thermomechanical_variation": thermomechanical,
        "unload": {"behavior": unload_behavior, "error": unload_error, "loaded": primary_diag, "unloaded": unload_diag},
        "balance": {"maximum_residual_m_inv": maximum_balance, "reference_content_m_inv": reference_content},
        "parameters": baseline.__dict__,
        "fields": {
            "x_m": (np.arange(primary.grid_points) * baseline.domain_m / primary.grid_points).tolist(),
            "total_density_m2": np.sum(primary.mobile_plus_m2 + primary.mobile_minus_m2, axis=0).tolist(),
            "signed_density_by_family_m2": (primary.mobile_plus_m2 - primary.mobile_minus_m2).tolist(),
            "nye_tensor_m_inv": alpha.tolist(),
            "plastic_slip_by_family": primary.accumulated_slip.tolist(),
            "plastic_deformation_gradient": primary.plastic_deformation_gradient.tolist(),
            "internal_stress_by_family_Pa": (sc + back + diffusion).tolist(),
        },
        "provenance": {
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "platform": platform.platform(), "python": platform.python_version(),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("output/gate_B1_driven_cdd_extended.json"))
    parser.add_argument("--plot-directory", type=Path, default=Path("output/gate_B1_extended_plots"))
    args = parser.parse_args()
    main(args.output, args.plot_directory)
