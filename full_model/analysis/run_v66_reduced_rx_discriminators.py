#!/usr/bin/env python3
"""Execute the V66 H1/H2 reduced recrystallization discriminators."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from full_model.production.structural_boundary import (
    NetworkState,
    StructuralBoundaryParameters,
    StructuralBoundaryState,
    absorb_lattice_content,
    exp_floor_event_rate_Hz,
    network_energy_J_m2,
    read_shockley_slope_J_m2_rad,
    structural_boundary_energy_J_m2,
    transfer_network_structural_content,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "full_model" / "verification" / "v66_reduced_rx"


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _absorption_derivative(theta_deg, rho_m2, p, quantum_m1=1.0e5):
    state = StructuralBoundaryState(
        rho_m2, 0.0, math.radians(theta_deg)/p.burgers_m)
    result = absorb_lattice_content(
        state, absorbed_plus_m1=quantum_m1, absorbed_minus_m1=0.0,
        compatible=True, parameters=p, energy_tolerance_J_m2=0.0)
    return result.delta_helmholtz_J_m2/quantum_m1


def _control_cases(p):
    theta = math.radians(2.0)
    state = StructuralBoundaryState(2e14, 2e14, theta/p.burgers_m)
    amount = math.radians(0.10)/p.burgers_m
    cases = {
        "compatible_positive": dict(plus=amount, minus=0.0, compatible=True),
        "sign_balanced": dict(plus=amount, minus=amount, compatible=True),
        "incompatible_positive": dict(plus=amount, minus=0.0, compatible=False),
        "no_incoming": dict(plus=0.0, minus=0.0, compatible=True),
    }
    out = {}
    for name, case in cases.items():
        result = absorb_lattice_content(
            state, absorbed_plus_m1=case["plus"],
            absorbed_minus_m1=case["minus"],
            compatible=case["compatible"], parameters=p,
            energy_tolerance_J_m2=0.0)
        out[name] = {
            "classification": result.classification,
            "delta_helmholtz_J_m2": result.delta_helmholtz_J_m2,
            "structural_change_m1": result.structural_change_m1,
            "misorientation_change_deg": math.degrees(
                p.burgers_m*result.structural_change_m1),
            "extrinsic_line_change_m1": result.extrinsic_line_change_m1,
            "annihilated_line_m1": result.annihilated_line_m1,
            "line_balance_residual_m1": result.line_balance_residual_m1,
            "accepted": result.accepted,
        }
    return out


def _h1_map(p):
    theta_deg = np.geomspace(0.15, 12.0, 90)
    rho_m2 = np.geomspace(1e12, 2e16, 100)
    derivative = np.empty((len(theta_deg), len(rho_m2)))
    for i, theta in enumerate(theta_deg):
        for j, rho in enumerate(rho_m2):
            derivative[i, j] = _absorption_derivative(theta, rho, p)
    affinity = np.maximum(-derivative, 0.0)
    temperatures = (700.0, 900.0, 1100.0)
    rate = {str(int(T)): exp_floor_event_rate_Hz(affinity, T, p)
            for T in temperatures}
    return theta_deg, rho_m2, derivative, rate


def _h2_curves(p):
    theta0_deg = 2.5
    delta_deg = np.linspace(-2.45, 2.45, 401)
    base = NetworkState(tuple(map(math.radians, (0.0, theta0_deg,
                                                  2.0*theta0_deg))))
    energy = np.empty_like(delta_deg)
    for i, delta in enumerate(delta_deg):
        state = NetworkState((base.orientation_rad[0],
                              base.orientation_rad[1]+math.radians(delta),
                              base.orientation_rad[2]))
        energy[i] = network_energy_J_m2(state, p)
    energy -= network_energy_J_m2(base, p)
    theta0 = math.radians(theta0_deg)
    curvature = (-2.0*p.high_angle_energy_J_m2
                 /(p.transition_misorientation_rad*theta0))
    stabilizing_k = -curvature
    delta_rad = np.radians(delta_deg)
    constrained = {
        "unconstrained": energy,
        "half_critical_accommodation": energy+0.25*stabilizing_k*delta_rad**2,
        "one_and_half_critical_accommodation": (
            energy+0.75*stabilizing_k*delta_rad**2),
    }
    perturbation = math.radians(0.1)/p.burgers_m
    plus = transfer_network_structural_content(base, perturbation, p)
    minus = transfer_network_structural_content(base, -perturbation, p)
    return delta_deg, constrained, {
        "theta0_deg": theta0_deg,
        "unconstrained_curvature_J_m2_rad2": curvature,
        "critical_accommodation_stiffness_J_m2_rad2": stabilizing_k,
        "positive_perturbation_delta_J_m2": plus.delta_helmholtz_J_m2,
        "negative_perturbation_delta_J_m2": minus.delta_helmholtz_J_m2,
        "positive_balance_residual_m1": plus.balance_residual_m1,
        "negative_balance_residual_m1": minus.balance_residual_m1,
    }


def _geometry_dilution_map(p):
    """Line supply needed to prevent theta dilution on a growing circle."""
    radius_um = np.geomspace(0.1, 20.0, 100)
    radial_speed_um_s = np.geomspace(1e-3, 1e3, 100)
    theta = math.radians(5.0)
    R = radius_um[:, None]*1e-6
    speed = radial_speed_um_s[None, :]*1e-6
    # theta_dot = b/L Ndot - theta/L Ldot, L=2*pi*R.  The threshold is
    # Ndot = theta*Ldot/b = theta*2*pi*v/b, independent of R for a circle.
    required_line_rate_m_s = theta*2.0*math.pi*speed/p.burgers_m
    return radius_um, radial_speed_um_s, np.broadcast_to(
        required_line_rate_m_s, (len(radius_um), len(radial_speed_um_s)))


def _plot_h1(theta, rho, derivative, rates, path):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7), constrained_layout=True)
    scale = 1e9*derivative
    image = axes[0].pcolormesh(
        rho, theta, scale, shading="auto", cmap="coolwarm",
        vmin=-max(abs(np.nanpercentile(scale, 3)), abs(np.nanpercentile(scale, 97))),
        vmax=max(abs(np.nanpercentile(scale, 3)), abs(np.nanpercentile(scale, 97))))
    axes[0].contour(rho, theta, derivative, levels=(0.0,), colors="black",
                    linewidths=1.0)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"compatible lattice density, $\rho_+$ (m$^{-2}$)")
    axes[0].set_ylabel(r"initial misorientation, $\theta$ (deg)")
    cbar = fig.colorbar(image, ax=axes[0])
    cbar.set_label(r"$dF/d\Lambda$ (nJ m$^{-1}$)")

    for T, rate in rates.items():
        j = int(np.argmin(np.abs(rho-2e14)))
        axes[1].plot(theta, rate[:, j], label=f"{T} K")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"initial misorientation, $\theta$ (deg)")
    axes[1].set_ylabel(r"EXP-floor event rate (s$^{-1}$)")
    axes[1].legend(frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_h2(delta, curves, path):
    fig, ax = plt.subplots(figsize=(5.2, 3.7), constrained_layout=True)
    labels = {
        "unconstrained": "boundary energy only",
        "half_critical_accommodation": "0.5 critical accommodation",
        "one_and_half_critical_accommodation": "1.5 critical accommodation",
    }
    for name, value in curves.items():
        ax.plot(delta, 1e3*value, label=labels[name])
    ax.axhline(0.0, color="black", linewidth=.7)
    ax.set_xlabel(r"middle-region perturbation, $\delta$ (deg)")
    ax.set_ylabel(r"energy change (mJ m$^{-2}$)")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p = StructuralBoundaryParameters()
    controls = _control_cases(p)
    theta, rho, derivative, rates = _h1_map(p)
    delta, curves, h2 = _h2_curves(p)
    radius, speed, supply = _geometry_dilution_map(p)
    h1_figure = OUT/"h1_absorption_map.png"
    h2_figure = OUT/"h2_network_redistribution.png"
    _plot_h1(theta, rho, derivative, rates, h1_figure)
    _plot_h2(delta, curves, h2_figure)

    favorable = derivative < 0.0
    result = {
        "schema": "v66-reduced-rx-discriminators-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "straight low-angle tilt boundary and three-region common-axis network",
        "material_claim": "generic mechanism discriminator; not calibrated",
        "parameters": asdict(p),
        "controls": controls,
        "H1": {
            "theta_deg_range": [float(theta[0]), float(theta[-1])],
            "rho_m2_range": [float(rho[0]), float(rho[-1])],
            "fraction_energy_favorable": float(np.mean(favorable)),
            "dF_dLambda_J_m_extrema": [float(np.min(derivative)),
                                        float(np.max(derivative))],
            "rate_Hz_at_theta2deg_rho2e14": {
                T: float(value[np.argmin(abs(theta-2.0)),
                               np.argmin(abs(rho-2e14))])
                for T, value in rates.items()},
            "prediction_confirmed": (
                controls["compatible_positive"]["misorientation_change_deg"] > 0
                and controls["sign_balanced"]["misorientation_change_deg"] == 0
                and controls["incompatible_positive"]["misorientation_change_deg"] == 0),
        },
        "H2": h2,
        "geometry_dilution": {
            "theta_deg": 5.0,
            "radius_um_range": [float(radius[0]), float(radius[-1])],
            "radial_speed_um_s_range": [float(speed[0]), float(speed[-1])],
            "required_structural_line_rate_m_s_range": [
                float(np.min(supply)), float(np.max(supply))],
            "formula": "Ndot_required = theta * 2*pi*v / b",
        },
        "scientific_decision": {
            "H1_supported_in_reduced_scope": bool(
                controls["compatible_positive"]["delta_helmholtz_J_m2"] < 0),
            "H2_redistribution_supported_in_reduced_scope": bool(
                h2["positive_perturbation_delta_J_m2"] < 0
                and h2["negative_perturbation_delta_J_m2"] < 0),
            "production_integration_qualified": False,
            "reason": (
                "compatible signed absorption can lower complete reduced energy and change "
                "misorientation; Read-Shockley concavity makes conservative redistribution "
                "energy lowering unless accommodation exceeds the derived critical stiffness. "
                "Kinetic accessibility and tensorial production coupling remain to be tested."),
        },
        "artifacts": {},
    }
    result_path = OUT/"result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    result["artifacts"] = {
        str(path.relative_to(ROOT)): _sha(path)
        for path in (h1_figure, h2_figure, result_path)
    }
    # The result cannot contain its own stable hash.  Record figure hashes and
    # a pre-manifest result hash, explicitly labelled.
    result["artifacts"].pop(str(result_path.relative_to(ROOT)))
    result["pre_manifest_result_sha256"] = _sha(result_path)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps(result["scientific_decision"], indent=2))


if __name__ == "__main__":
    main()
