#!/usr/bin/env python3
"""Decision-grade projected-Hessian and kinetic-dispersion audit for V19."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from full_model.production.arrhenius_kinetics import ActivatedProcess, EV_J
from full_model.production.dislocation_free_energy import DislocationFreeEnergyParameters
from full_model.production.predictive_signed_wall import (
    SignedWallParameters, SignedWallState, linearized_operator,
    projected_hessian, state_vector,
)
from full_model.production.wall_ordering_energy import WallOrderingParameters


def default_parameters():
    rho_scale = 5.0e14
    line = 0.5*45.0e9*(2.48e-10)**2
    process = lambda name, attempt: ActivatedProcess(name, attempt, 0.0, 1e9)
    return SignedWallParameters(
        rho_scale,
        DislocationFreeEnergyParameters(
            line, .06*line, 1e14, 1., rho_scale, 1.3, .3, .25),
        WallOrderingParameters(
            1.2*line*rho_scale*.3, rho_scale, 1.3, .3,
            .08*line*rho_scale, .25*line, .8*rho_scale, 2e-7),
        process("mobile_to_forest_lock", 1e7),
        process("forest_to_wall_trap", 1e7),
        process("wall_to_mobile_release", 1e7),
        process("neutral_pair_annihilation", 1e7),
        process("wall_order_relaxation", 1e7),
        .9*EV_J, .8*EV_J, 1.1*EV_J, 1.2*EV_J, .7*EV_J,
        1.5e9, exp_floor=.05, neutral_capture_area_m2=1e-16,
        order_energy_scale_J_m3=1e6,
        density_gradient_J_m3_m2=1e-7, order_gradient_J_m=2e-7,
        compatibility_J_m=1e-8, diffusivity_m2_s=1e-12,
        orientation_mobility_m3_J_s=1e-5, plastic_spin_rate_s=10.)


def normalized_state(total_ratio, order, families=4):
    # Reservoir fractions sum to one and both signs are initially balanced.
    fractions = (.10, .10, .20, .20, .20, .20)
    arrays = [np.full(families, total_ratio*f/families) for f in fractions]
    return state_vector(SignedWallState(*arrays, np.asarray(order)))


def run():
    p = default_parameters()
    velocities = np.array([[30., 0.], [0., 30.], [-30., 0.], [0., -30.]])
    wavelengths = np.geomspace(.1e-6, 20e-6, 96)
    states = []
    any_physical_instability = False
    for ratio in (.7, 1.0, 1.3, 1.6):
        for order in (.05, .35, .65, .95):
            x = normalized_state(ratio, order)
            raw, projected, modes = projected_hessian(x, p)
            growth_zero = float(np.max(np.real(np.linalg.eigvals(
                linearized_operator(x, [0.0, 0.0], velocities, 8e8, 1100., p,
                                    local_hessian=raw)))))
            growth = []
            for wavelength in wavelengths:
                L = linearized_operator(
                    x, [2*np.pi/wavelength, 0.0], velocities,
                    8e8, 1100., p, local_hessian=raw)
                growth.append(float(np.max(np.real(np.linalg.eigvals(L)))))
            growth = np.asarray(growth)
            imax = int(np.argmax(growth))
            # The local functional has exactly flat sign-partition directions;
            # finite-difference roundoff gives them O(1e-2 1/s) eigenvalues.
            # One inverse second is a deliberately conservative instability
            # threshold, over two orders above that measured neutral-mode noise.
            tolerance = 1.0
            positive = growth > tolerance
            unstable_at_zero = bool(growth_zero > tolerance)
            cutoff_only = bool((not unstable_at_zero) and positive[0]
                               and not np.any(positive[1:]))
            finite_band = bool((not unstable_at_zero)
                               and np.any(positive[1:-1]) and not cutoff_only)
            any_physical_instability |= unstable_at_zero or finite_band
            fastest_is_zero = bool(growth_zero >= growth[imax])
            states.append({
                "total_density_ratio": ratio,
                "wall_order": order,
                "raw_minimum_eigenvalue_J_m3": float(np.linalg.eigvalsh(raw)[0]),
                "projected_minimum_eigenvalue_J_m3": float(np.linalg.eigvalsh(projected)[0]),
                "projected_modes": modes,
                "growth_rate_at_k0_s_inv": growth_zero,
                "unstable_at_k0": unstable_at_zero,
                "maximum_growth_rate_s_inv": float(max(growth_zero, growth[imax])),
                "fastest_wavelength_m": (None if fastest_is_zero
                                          else float(wavelengths[imax])),
                "fastest_mode": ("k=0 / infinite wavelength" if fastest_is_zero
                                   else "finite sampled wavelength"),
                "positive_growth_finite_band": finite_band,
                "grid_cutoff_only": cutoff_only,
                "dispersion": [{"wavelength_m": float(w), "max_real_growth_s_inv": float(g)}
                               for w, g in zip(wavelengths, growth)],
            })
    return {
        "schema": "asb-drx/v19-projected-dynamics/v1",
        "source_operator": "full_model.production.predictive_signed_wall",
        "variables": "four families x (mobile+/-, forest+/-, wall+/-), q_wall, theta",
        "constraints": ["total line for transfer channels",
                        "signed Burgers content per family",
                        "nonnegativity tangent cone"],
        "kinetic_branch": {
            "temperature": "isothermal 1100 K perturbations held fixed",
            "mechanics": "fixed macrostress 800 MPa; orientation/plastic-spin coupling retained",
            "transport": "opposite signed advection plus physical diffusion",
            "reaction_barriers": "EXP-floor enthalpy minus T times activation entropy",
            "thermal_feedback_in_linearization": False,
            "stress_feedback_in_linearization": False,
        },
        "projected_dynamic_instability_identified": bool(any_physical_instability),
        "candidate_finite_wavelength_mode_identified": bool(any(
            state["positive_growth_finite_band"] for state in states)),
        "growth_rate_classification_tolerance_s_inv": 1.0,
        "scientific_gate_passed": False,
        "states": states,
        "claim_limit": "A linear accessible mode is a precursor hypothesis; full-mechanics formation and release persistence remain required.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"output": str(args.output),
                      "states": len(result["states"]),
                      "projected_dynamic_instability_identified":
                      result["projected_dynamic_instability_identified"]}, indent=2))


if __name__ == "__main__":
    main()
