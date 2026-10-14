#!/usr/bin/env python3
"""Discriminate finite-wavelength growth in the production transport operator.

The projection deliberately holds non-mobile reservoirs fixed.  It is the
mobile signed-transport sub-block of the authoritative common-state Jacobian,
not a second reduced constitutive implementation.  A positive result is thus
necessary evidence for the production mechanism, but is not by itself a wall
or recrystallization claim.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from full_model.analysis.run_v66_interaction_stability import _state
from full_model.production.common_tensorial_wall import (
    CommonWallDriving, CommonWallParameters, CommonWallState,
    wall_residual,
)
from full_model.production.tensorial_nye import (
    bcc_four_family_systems, make_junction_topology,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"full_model"/"verification"/"v66_production_groma_tangent"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_parameters(spacing_m):
    base = CommonWallParameters(spacing_m=spacing_m, transport_scheme="spectral")
    poisson = base.c12_Pa/(base.c11_Pa+base.c12_Pa)
    reduced_modulus = base.c44_Pa/(2.0*np.pi*(1.0-poisson))
    gb2 = reduced_modulus*base.burgers_m**2
    return replace(
        base, groma_mobile_correlation_energy_J_m=1.25*gb2,
        groma_gnd_correlation_energy_J_m=.5*gb2,
        groma_sign_resolved_transport_enabled=True,
        mobile_correlation_diffusivity_m2_s=0.0), gb2


def mobile_symbol(state, driving, systems, topologies, parameters, mode):
    """Return full and transport-only signed-mobile production projections."""
    n = state.orientation_rad.shape[0]
    x = np.arange(n)[:, None]/n
    phase = 2.0*np.pi*mode*x
    cosine = np.broadcast_to(np.cos(phase), (n, n))
    sine = np.broadcast_to(np.sin(phase), (n, n))
    norm = float(np.mean(cosine*cosine))
    coordinates = tuple(
        (name, family) for name in ("mobile_plus_m2", "mobile_minus_m2")
        for family in range(len(systems)))
    full_matrix = np.zeros((len(coordinates), len(coordinates)), complex)
    transport_matrix = np.zeros_like(full_matrix)
    for column, (input_name, input_family) in enumerate(coordinates):
        arrays = {name: np.zeros_like(value) for name, value in state.__dict__.items()}
        arrays[input_name][..., input_family] = parameters.rho_reference_m2*cosine
        direction = CommonWallState(**arrays)
        step = 1e-6*max(
            float(np.max(np.abs(getattr(state, input_name)))), 1.0
        )/parameters.rho_reference_m2
        plus = replace(state, **{
            name: value+step*np.asarray(getattr(direction, name))
            for name, value in state.__dict__.items()})
        minus = replace(state, **{
            name: value-step*np.asarray(getattr(direction, name))
            for name, value in state.__dict__.items()})
        rp = wall_residual(
            plus, driving, systems, topologies, parameters,
            differentiation_state=True)
        rm = wall_residual(
            minus, driving, systems, topologies, parameters,
            differentiation_state=True)
        for row, (output_name, output_family) in enumerate(coordinates):
            full = (getattr(rp.state_rate, output_name)[..., output_family]
                    -getattr(rm.state_rate, output_name)[..., output_family]
                    )/(2.0*step*parameters.rho_reference_m2)
            channel = ("transport_mobile_plus_rate_m2_s" if
                       output_name == "mobile_plus_m2" else
                       "transport_mobile_minus_rate_m2_s")
            transport = (
                rp.channel_rates_m2_s[channel][..., output_family]
                -rm.channel_rates_m2_s[channel][..., output_family]
                )/(2.0*step*parameters.rho_reference_m2)
            full_matrix[row, column] = (
                np.mean(full*cosine)/norm-1j*np.mean(full*sine)/norm)
            transport_matrix[row, column] = (
                np.mean(transport*cosine)/norm
                -1j*np.mean(transport*sine)/norm)
    full_eigenvalues, full_eigenvectors = np.linalg.eig(full_matrix)
    transport_eigenvalues, transport_eigenvectors = np.linalg.eig(
        transport_matrix)
    return (coordinates, full_eigenvalues, full_eigenvectors,
            transport_eigenvalues, transport_eigenvectors)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # The reduced oracle predicts an order-two-micrometre fastest wavelength
    # at this density.  A 4.8 micrometre periodic box contains both longer and
    # shorter resolved competitors, so an interior mode can be discriminated.
    n = 24
    dx = 2.0e-7
    systems = bcc_four_family_systems()
    topologies = (make_junction_topology(
        systems, 0, 1, 1, -1, line_tension_J_m=0.0),)
    state = _state(n)
    parameters, gb2 = candidate_parameters(dx)
    conditions = {}
    for stress_Pa in (0.0, 5.0e7, 6.0e7, 8.0e7, 1.0e8,
                      1.8e8, 5.0e8, 1.0e9, 2.0e9):
        records = []
        for mode in range(1, n//2):
            (coordinates, full_eigenvalues, full_eigenvectors,
             transport_eigenvalues, transport_eigenvectors) = mobile_symbol(
                state, CommonWallDriving(resolved_stress_Pa=np.full(
                    state.mobile_plus_m2.shape, stress_Pa)),
                systems, topologies, parameters, mode)
            leading = int(np.argmax(full_eigenvalues.real))
            vector = full_eigenvectors[:, leading]
            transport_leading = int(np.argmax(transport_eigenvalues.real))
            transport_vector = transport_eigenvectors[:, transport_leading]
            positive_weight = float(np.sum(np.abs(vector[:4])**2))
            negative_weight = float(np.sum(np.abs(vector[4:])**2))
            records.append({
                "mode_index": mode,
                "wavelength_m": n*dx/mode,
                "leading_real_rate_s": float(full_eigenvalues[leading].real),
                "leading_imaginary_rate_s": float(full_eigenvalues[leading].imag),
                "transport_leading_real_rate_s": float(
                    transport_eigenvalues[transport_leading].real),
                "transport_leading_imaginary_rate_s": float(
                    transport_eigenvalues[transport_leading].imag),
                "positive_population_weight": positive_weight,
                "negative_population_weight": negative_weight,
                "dominant_coordinate": list(coordinates[
                    int(np.argmax(np.abs(vector)))]),
                "transport_dominant_coordinate": list(coordinates[
                    int(np.argmax(np.abs(transport_vector)))]),
            })
        conditions[str(int(stress_Pa))] = records

    figure = OUT/"production_mobile_tangent.png"
    fig, ax = plt.subplots(figsize=(5.5, 3.8), constrained_layout=True)
    for stress, records in conditions.items():
        ax.plot([1e-6/item["wavelength_m"] for item in records],
                [item["transport_leading_real_rate_s"] for item in records], marker="o",
                label=f"{float(stress)/1e6:g} MPa")
    ax.axhline(0.0, color="black", linewidth=.7)
    ax.set_xlabel(r"wave number (cycles $\mu$m$^{-1}$)")
    ax.set_ylabel(r"leading mobile tangent rate (s$^{-1}$)")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(figure, dpi=180)
    plt.close(fig)

    maxima = {
        stress: max(records, key=lambda item: item[
            "transport_leading_real_rate_s"])
        for stress, records in conditions.items()}
    finite_selected = {
        stress: bool(item["transport_leading_real_rate_s"] > 0.0
                     and item["mode_index"] not in (1, n//2-1))
        for stress, item in maxima.items()}
    result = {
        "schema": "v66-production-groma-mobile-tangent-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_scope": (
            "signed-mobile sub-block of authoritative production residual; "
            "other common-state coordinates held fixed"),
        "grid": [n, n], "spacing_m": dx,
        "base_temperature_K": float(state.temperature_K[0, 0]),
        "base_mobile_density_per_sign_family_m2": float(
            state.mobile_plus_m2[0, 0, 0]),
        "parameters": {
            "A": 1.25, "D": .5, "reduced_modulus_b2_J_m": gb2,
            "arrhenius_glide_barrier_eV": parameters.glide_barrier_eV,
            "exp_a": parameters.exp_a, "exp_n": parameters.exp_n,
            "exp_floor": parameters.exp_floor,
            "taylor_alpha": parameters.taylor_alpha,
        },
        "conditions": conditions,
        "maximum_by_stress": maxima,
        "finite_interior_mode_selected": finite_selected,
        "claim_boundary": (
            "a positive mobile tangent is neither nonlinear saturation, a "
            "GND-bearing wall, a subgrain, nor recrystallization"),
        "artifacts": {str(figure.relative_to(ROOT)): _sha(figure)},
    }
    path = OUT/"result.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"maximum_by_stress": maxima,
                      "finite_interior_mode_selected": finite_selected}, indent=2))


if __name__ == "__main__":
    main()
