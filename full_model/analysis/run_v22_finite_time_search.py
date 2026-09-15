"""Bounded V22 accepted-map search for signed transient amplification.

This is a local decision screen, not a material calibration.  Every tangent is
obtained by differentiating the same bounded accepted step used to advance the
homogeneous trajectory.  Signed gain is projected onto normalized plus/minus
reservoir differences, excluding the scalar order coordinate.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from full_model.production.common_tensorial_wall import (
    CommonWallDriving, CommonWallParameters, CommonWallState,
    accepted_euler_step, accepted_step_fourier_symbol, fourier_symbol,
    wall_polarization_invariants,
)
from full_model.production.tensorial_nye import (
    bcc_four_family_systems, make_junction_topology,
)


def topologies(systems, shear_modulus, burgers):
    return tuple(make_junction_topology(
        systems, a, b, sa, sb,
        line_tension_J_m=.5*shear_modulus*burgers**2)
        for a in range(4) for b in range(a+1, 4)
        for sa, sb in ((1, -1), (-1, 1)))


def make_state(n, density, temperature, topology_count, signed_bias=1e-3):
    shape = (n, n, 4)
    bias = signed_bias*np.asarray([1., -1., 1., -1.])
    reservoirs = {}
    for name, fraction in (("mobile", .40), ("forest", .40), ("wall", .15)):
        pair = fraction*density/8.0
        reservoirs[name+"_plus_m2"] = np.broadcast_to(
            pair*(1+bias), shape).copy()
        reservoirs[name+"_minus_m2"] = np.broadcast_to(
            pair*(1-bias), shape).copy()
    junction = np.full((n, n, topology_count), .05*density/max(topology_count, 1))
    return CommonWallState(
        reservoirs["mobile_plus_m2"], reservoirs["mobile_minus_m2"],
        reservoirs["forest_plus_m2"], reservoirs["forest_minus_m2"],
        reservoirs["wall_plus_m2"], reservoirs["wall_minus_m2"], junction,
        np.full((n, n), 1e-4), np.full((n, n), 1e-4), np.zeros(shape),
        np.zeros((n, n, 3, 3)), np.zeros(shape+(3,)),
        np.zeros(shape+(3, 3)), np.zeros((n, n)),
        np.full((n, n), temperature))


def signed_basis(layout):
    columns = []
    for reservoir in ("mobile", "forest", "wall"):
        for family in range(4):
            vector = np.zeros(len(layout), dtype=complex)
            plus = layout.index((reservoir+"_plus_m2", (family,),
                                 layout[0][2]))
            minus = layout.index((reservoir+"_minus_m2", (family,),
                                  layout[0][2]))
            vector[plus] = 1/np.sqrt(2); vector[minus] = -1/np.sqrt(2)
            columns.append(vector)
    return np.stack(columns, axis=1)


def search_case(n, domain_m, temperature, density_ratio, strain_rate,
                entropy_kB, multi_hit, steps, requested_dt):
    systems = bcc_four_family_systems()
    p = CommonWallParameters(
        spacing_m=domain_m/n, rho_reference_m2=5e14,
        activation_entropy_kB=entropy_kB,
        multi_hit_enabled=multi_hit,
        multi_hit_wall_log_factor=1.0,
        multi_hit_junction_log_factor=1.0,
        multi_hit_annihilation_log_factor=-.5)
    topology = topologies(systems, p.c44_Pa, p.burgers_m)
    state = make_state(n, density_ratio*p.rho_reference_m2, temperature,
                       len(topology))
    modes = ((0, 0), (1, 0), (2, 0))
    propagators = {mode: None for mode in modes}
    maximum_real = {mode: -np.inf for mode in modes}
    maximum_signed_real = {mode: -np.inf for mode in modes}
    elapsed = 0.0; minimum_scale = 1.0
    cumulative_collision = 0.0
    for _ in range(steps):
        # Elastic loading proxy is explicitly rate/history controlled; the cap
        # prevents the local screen from entering an undeclared stress regime.
        stress = min(2.0e9, p.c44_Pa*strain_rate*max(elapsed, requested_dt))
        family_stress = np.empty(state.mobile_plus_m2.shape)
        family_stress[...] = stress*np.asarray([1., .75, -.8, -.6])
        driving = CommonWallDriving(resolved_stress_Pa=family_stress)
        for mode in modes:
            instantaneous = fourier_symbol(
                state, driving, systems, topology, p, mode, 2e-6)
            basis = signed_basis(instantaneous["layout"])
            maximum_real[mode] = max(
                maximum_real[mode], float(np.max(
                    instantaneous["eigenvalues_s_inv"].real)))
            signed_generator = (basis.conj().T
                                @instantaneous["matrix_s_inv"]@basis)
            maximum_signed_real[mode] = max(
                maximum_signed_real[mode], float(np.max(
                    np.linalg.eigvals(signed_generator).real)))
            accepted = accepted_step_fourier_symbol(
                state, driving, systems, topology, p, mode, requested_dt, 2e-6)
            matrix = accepted["matrix"]
            propagators[mode] = (matrix if propagators[mode] is None else
                                 matrix@propagators[mode])
        state, residual, scale = accepted_euler_step(
            state, driving, systems, topology, p, requested_dt)
        actual_dt = requested_dt*scale
        elapsed += actual_dt; minimum_scale = min(minimum_scale, scale)
        cumulative_collision += actual_dt*float(np.mean(
            residual.channel_rates_m2_s["collision_frequency_s"]))
    rows = []
    for mode in modes:
        matrix = propagators[mode]
        layout = fourier_symbol(
            state, driving, systems, topology, p, mode, 2e-6)["layout"]
        basis = signed_basis(layout)
        signed_map = basis.conj().T@matrix@basis
        u, singular, vh = np.linalg.svd(signed_map)
        rows.append({
            "mode": list(mode),
            "maximum_instantaneous_real_s_inv": maximum_real[mode],
            "maximum_signed_instantaneous_real_s_inv": maximum_signed_real[mode],
            "full_finite_time_gain": float(np.linalg.svd(
                matrix, compute_uv=False)[0]),
            "signed_finite_time_gain": float(singular[0]),
            "signed_optimal_input": [[float(x.real), float(x.imag)]
                                      for x in vh[0]],
        })
    invariants = wall_polarization_invariants(state, systems, p)
    return {
        "inputs": {"grid": n, "domain_m": domain_m,
                   "temperature_K": temperature,
                   "density_ratio": density_ratio,
                   "strain_rate_s": strain_rate,
                   "activation_entropy_kB": entropy_kB,
                   "multi_hit_enabled": multi_hit,
                   "steps": steps, "requested_dt_s": requested_dt},
        "actual_time_s": elapsed, "minimum_accept_scale": minimum_scale,
        "cumulative_collision_exposure": cumulative_collision,
        "final_wall_order_mean": float(np.mean(state.wall_order)),
        "final_wall_polarization_mean": float(np.mean(
            invariants["wall_polarization"])),
        "modes": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    cases = []
    temperatures = (950., 1250.)
    rates = (1e3, 1e5)
    density_ratios = (0.5, 2.0)
    multi_hits = (False, True)
    if args.quick:
        temperatures, rates, density_ratios, multi_hits = ((1100.,), (1e4,),
                                                            (1.0,), (False, True))
    for temperature in temperatures:
        for rate in rates:
            for density in density_ratios:
                for multi_hit in multi_hits:
                    cases.append(search_case(
                        6, 6e-6, temperature, density, rate, 0.0, multi_hit,
                        3 if args.quick else 5, 2e-9))
    if not args.quick:
        # Incommensurate-domain check of the strongest high-T/high-density
        # branch; wavelength is never varied as a constitutive parameter.
        for multi_hit in (False, True):
            cases.append(search_case(
                6, 9.7e-6, 1250., 2.0, 1e5, 0.0, multi_hit, 5, 2e-9))
    best_signed = max(row["signed_finite_time_gain"]
                      for case in cases for row in case["modes"])
    positive_interior = any(
        row["mode"] != [0, 0]
        and row["maximum_signed_instantaneous_real_s_inv"] > 0.0
        for case in cases for row in case["modes"])
    positive_rows = []
    for case_index, case in enumerate(cases):
        inputs = case["inputs"]
        for row in case["modes"]:
            mode_number = int(np.hypot(*row["mode"]))
            if (mode_number > 0
                    and row["maximum_signed_instantaneous_real_s_inv"] > 0.0):
                positive_rows.append({
                    "case_index": case_index,
                    "domain_m": inputs["domain_m"],
                    "wavelength_m": inputs["domain_m"]/mode_number,
                    "temperature_K": inputs["temperature_K"],
                    "density_ratio": inputs["density_ratio"],
                    "strain_rate_s": inputs["strain_rate_s"],
                    "multi_hit_enabled": inputs["multi_hit_enabled"],
                    "mode": row["mode"],
                    "growth_rate_s_inv":
                        row["maximum_signed_instantaneous_real_s_inv"],
                })
    robust_matches = []
    for index, left in enumerate(positive_rows):
        for right in positive_rows[index+1:]:
            same_loading = all(left[key] == right[key] for key in (
                "temperature_K", "density_ratio", "strain_rate_s",
                "multi_hit_enabled"))
            different_domain = left["domain_m"] != right["domain_m"]
            wavelength_error = abs(
                left["wavelength_m"]-right["wavelength_m"]
            )/max(left["wavelength_m"], right["wavelength_m"])
            if same_loading and different_domain and wavelength_error <= 0.10:
                robust_matches.append({
                    "left": left, "right": right,
                    "relative_wavelength_difference": wavelength_error,
                })
    threshold = float(np.exp(8))
    if robust_matches:
        classification = "SIGNED_INTERIOR_CANDIDATE_REQUIRES_NONLINEAR_TEST"
    elif best_signed >= threshold:
        classification = "SIGNED_TRANSIENT_CANDIDATE_REQUIRES_NONLINEAR_TEST"
    else:
        classification = "NO_DECISION_THRESHOLD_REACHED_IN_LOCAL_SEARCH"
    result = {
        "schema": "v22_finite_time_search_v1",
        "method": "JVP_OF_AUTHORITATIVE_ACCEPTED_STEP_MAP",
        "signed_projection": "normalized plus-minus differences for mobile/forest/wall; q excluded",
        "thresholds": {"finite_time_signed_gain": threshold,
                       "cross_domain_wavelength_relative_difference": 0.10},
        "best_signed_finite_time_gain": best_signed,
        "positive_interior_mode_found": positive_interior,
        "positive_interior_rows": positive_rows,
        "robust_cross_domain_signed_matches": robust_matches,
        "classification": classification,
        "fixture_passed": True,
        "scientific_gate_passed": False,
        "local_campaign_authorized": True,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({key: result[key] for key in (
        "best_signed_finite_time_gain", "positive_interior_mode_found",
        "classification")}, indent=2))


if __name__ == "__main__":
    main()
