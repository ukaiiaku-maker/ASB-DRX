#!/usr/bin/env python3
"""Nonlinear signed-transport test of the V66 production Groma branch.

This first nonlinear discriminator disables reservoir reactions while retaining
the authoritative production flux, Arrhenius/Taylor kinetics, signed plastic
slip, beta-p, Mura--Nye, and lattice-spin updates.  The corresponding complete-
reaction trajectory is a separate required control; this script must not be
used to claim a wall or RX by itself.
"""

from __future__ import annotations

from dataclasses import fields, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from full_model.analysis.run_v66_interaction_stability import _state
from full_model.analysis.run_v66_production_groma_tangent import candidate_parameters
from full_model.production.common_tensorial_wall import (
    CommonWallDriving, CommonWallState, wall_free_energy_density_J_m3,
    resolved_driving_components, wall_residual,
)
from full_model.production.nonlocal_elasticity import elastic_energy_density
from full_model.production.tensorial_nye import (
    bcc_four_family_systems, make_junction_topology,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"full_model"/"verification"/"v66_production_groma_nonlinear_mechanical"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _map(state, operation):
    return CommonWallState(**{
        item.name: operation(item.name, np.asarray(getattr(state, item.name)))
        for item in fields(state)})


def _combine(state, rate, scale):
    return _map(state, lambda name, value: value+scale*getattr(rate, name))


def _rhs(state, driving, systems, topologies, parameters):
    residual = wall_residual(
        state, driving, systems, topologies, parameters,
        differentiation_state=True)
    full = residual.state_rate
    zero = lambda value: np.zeros_like(value)
    rate = CommonWallState(
        residual.channel_rates_m2_s["transport_mobile_plus_rate_m2_s"],
        residual.channel_rates_m2_s["transport_mobile_minus_rate_m2_s"],
        zero(state.forest_plus_m2), zero(state.forest_minus_m2),
        zero(state.wall_plus_m2), zero(state.wall_minus_m2),
        zero(state.junction_m2), zero(state.wall_order),
        zero(state.multi_hit_coordination), full.slip,
        full.beta_p, full.alignment_m2, full.family_nye_m1,
        full.orientation_rad, zero(state.temperature_K))
    raw = residual.channel_rates_m2_s["raw_stress_Pa"]
    mechanical_power = float(np.mean(np.sum(
        raw*full.slip, axis=2)))
    defect_rate = float(np.mean(
        residual.channel_rates_m2_s["transport_free_energy_rate_W_m3"]))
    return rate, mechanical_power, mechanical_power-defect_rate


def _rk4(state, dt, driving, systems, topologies, parameters):
    k1, p1, h1 = _rhs(state, driving, systems, topologies, parameters)
    k2, p2, h2 = _rhs(
        _combine(state, k1, .5*dt), driving, systems, topologies, parameters)
    k3, p3, h3 = _rhs(
        _combine(state, k2, .5*dt), driving, systems, topologies, parameters)
    k4, p4, h4 = _rhs(
        _combine(state, k3, dt), driving, systems, topologies, parameters)
    increment = _map(k1, lambda name, value: (
        value+2.0*getattr(k2, name)+2.0*getattr(k3, name)+getattr(k4, name))/6.0)
    return (_combine(state, increment, dt),
            dt*(p1+2*p2+2*p3+p4)/6.0,
            dt*(h1+2*h2+2*h3+h4)/6.0)


def _observables(state, spacing_m):
    mobile = state.mobile_plus_m2+state.mobile_minus_m2
    signed = state.mobile_plus_m2-state.mobile_minus_m2
    total = np.sum(mobile, axis=2)
    fluctuation = total-np.mean(total)
    spectrum = np.abs(np.fft.fftn(fluctuation))**2
    spectrum[0, 0] = 0.0
    index = np.unravel_index(int(np.argmax(spectrum)), spectrum.shape)
    signed_index = tuple(
        item if item <= total.shape[axis]//2 else item-total.shape[axis]
        for axis, item in enumerate(index))
    cycles = np.sqrt(sum(item*item for item in signed_index))
    wavelength = (None if cycles == 0.0 else
                  total.shape[0]*spacing_m/cycles)
    family_polarization = np.divide(
        signed, mobile, out=np.zeros_like(signed), where=mobile > 0.0)
    nye = np.sum(state.family_nye_m1, axis=2)
    return {
        "mean_total_mobile_density_m2": float(np.mean(total)),
        "density_contrast": float(np.std(total)/np.mean(total)),
        "family_polarization_rms": float(np.sqrt(np.mean(
            family_polarization*family_polarization))),
        "family_polarization_max_abs": float(np.max(np.abs(
            family_polarization))),
        "nye_rms_m1": float(np.sqrt(np.mean(nye*nye))),
        "orientation_span_deg": float(np.ptp(state.orientation_rad)*180/np.pi),
        "peak_mode": list(map(int, signed_index)),
        "peak_wavelength_m": None if wavelength is None else float(wavelength),
    }


def _elastic_energy(state, driving, systems, topologies, parameters):
    resolved = resolved_driving_components(
        state, driving, systems, topologies, parameters)
    return float(np.mean(elastic_energy_density(
        resolved["stress_tensor_Pa"], resolved["compatible_strain"],
        resolved["eigenstrain"])))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n = 24
    spacing = 2.0e-7
    systems = bcc_four_family_systems()
    topologies = (make_junction_topology(
        systems, 0, 1, 1, -1, line_tension_J_m=0.0),)
    parameters, gb2 = candidate_parameters(spacing)
    state = replace(_state(n), temperature_K=np.full((n, n), 900.0))
    rng = np.random.default_rng(66024)
    plus_noise = rng.normal(size=state.mobile_plus_m2.shape)
    minus_noise = rng.normal(size=state.mobile_minus_m2.shape)
    plus_noise -= np.mean(plus_noise, axis=(0, 1), keepdims=True)
    minus_noise -= np.mean(minus_noise, axis=(0, 1), keepdims=True)
    state = replace(
        state, mobile_plus_m2=state.mobile_plus_m2*(1.0+1e-3*plus_noise),
        mobile_minus_m2=state.mobile_minus_m2*(1.0+1e-3*minus_noise))
    # Fixed total shear strain gives families 2/3 about 60 MPa initially and,
    # unlike a prescribed resolved stress, retains beta-p/elastic feedback.
    engineering_shear = 1.65e-3
    driving = CommonWallDriving(mean_strain=np.array(
        [[0.0, .5*engineering_shear], [.5*engineering_shear, 0.0]]))
    state.validate(systems, topologies)
    initial_content = np.sum(
        state.mobile_plus_m2+state.mobile_minus_m2, axis=(0, 1))
    initial_energy = float(np.mean(wall_free_energy_density_J_m3(
        state, parameters, topologies, systems)))
    initial_elastic = _elastic_energy(
        state, driving, systems, topologies, parameters)
    records = [{"time_s": 0.0, **_observables(state, spacing)}]
    time_s = 0.0
    next_record = 500.0
    # Energy convergence is substantially stricter than positivity for the
    # finite-amplitude stage; ten seconds keeps the endpoint first-law audit
    # informative rather than relying on the 50 s linear-stability limit.
    dt = 10.0
    work = 0.0
    heat = 0.0
    rejected = 0
    termination_reason = "target_time_reached"
    while time_s < 12000.0-1e-12:
        trial_dt = min(dt, 12000.0-time_s, next_record-time_s)
        try:
            candidate, work_increment, heat_increment = _rk4(
                state, trial_dt, driving, systems, topologies, parameters)
            mobile_min = min(float(np.min(candidate.mobile_plus_m2)),
                             float(np.min(candidate.mobile_minus_m2)))
            relative_increment = max(
                float(np.max(np.abs(candidate.mobile_plus_m2-state.mobile_plus_m2)
                             /state.mobile_plus_m2)),
                float(np.max(np.abs(candidate.mobile_minus_m2-state.mobile_minus_m2)
                             /state.mobile_minus_m2)))
            if mobile_min <= 0.0 or relative_increment > .03:
                raise FloatingPointError("inadmissible explicit transport trial")
            candidate.validate(systems, topologies)
        except (FloatingPointError, ValueError):
            dt *= .5
            rejected += 1
            if dt < 1e-3:
                termination_reason = (
                    "loss_of_unprojected_positive_continuum_state_below_1_ms_step")
                break
            continue
        state = candidate
        time_s += trial_dt
        work += work_increment
        heat += heat_increment
        if relative_increment < .005:
            dt = min(10.0, 1.25*dt)
        if abs(time_s-next_record) < 1e-9:
            records.append({"time_s": time_s, **_observables(state, spacing)})
            print(json.dumps(records[-1]), flush=True)
            next_record += 500.0

    if records[-1]["time_s"] != time_s:
        records.append({"time_s": time_s, **_observables(state, spacing)})

    final_content = np.sum(
        state.mobile_plus_m2+state.mobile_minus_m2, axis=(0, 1))
    final_energy = float(np.mean(wall_free_energy_density_J_m3(
        state, parameters, topologies, systems)))
    final_elastic = _elastic_energy(
        state, driving, systems, topologies, parameters)
    content_error = np.max(np.abs(final_content-initial_content))
    content_scale = np.max(initial_content)
    first_law_residual = work-(final_energy-initial_energy)-heat
    endpoint_first_law_residual = (
        -(final_elastic-initial_elastic)-(final_energy-initial_energy)-heat)

    figure = OUT/"nonlinear_transport_history.png"
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.2), constrained_layout=True)
    t = np.array([item["time_s"] for item in records])
    axes[0].semilogy(t, [item["density_contrast"] for item in records])
    axes[0].set(xlabel="time (s)", ylabel="total-density contrast")
    axes[1].semilogy(t, [item["family_polarization_rms"] for item in records])
    axes[1].set(xlabel="time (s)", ylabel="family polarization RMS")
    axes[2].plot(t, [1e6*(item["peak_wavelength_m"] or np.nan)
                     for item in records], marker=".")
    axes[2].set(xlabel="time (s)", ylabel="peak wavelength (micrometres)")
    fig.savefig(figure, dpi=180)
    plt.close(fig)

    result = {
        "schema": "v66-production-groma-nonlinear-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "reaction-disabled production signed transport with compatible "
            "kinematics and nonlocal fixed-strain elastic feedback"),
        "grid": [n, n], "spacing_m": spacing, "temperature_K": 900.0,
        "fixed_engineering_shear": engineering_shear, "seed": 66024,
        "initial_noise_fraction": 1.0e-3,
        "parameters": {"A": 1.25, "D": .5,
                       "reduced_modulus_b2_J_m": gb2},
        "records": records,
        "ledger": {
            "completed_time_s": time_s,
            "termination_reason": termination_reason,
            "maximum_relative_family_content_error": float(
                content_error/content_scale),
            "external_work_J_m3": work,
            "interpretation_of_external_work": (
                "integrated plastic power / recoverable elastic release at fixed strain"),
            "recoverable_elastic_energy_change_J_m3": final_elastic-initial_elastic,
            "defect_free_energy_change_J_m3": final_energy-initial_energy,
            "irreversible_heat_J_m3": heat,
            "first_law_residual_J_m3": first_law_residual,
            "relative_first_law_residual": float(abs(first_law_residual)/max(
                abs(work), abs(final_energy-initial_energy), abs(heat), 1.0)),
            "endpoint_fixed_strain_first_law_residual_J_m3": (
                endpoint_first_law_residual),
            "endpoint_fixed_strain_relative_first_law_residual": float(
                abs(endpoint_first_law_residual)/max(
                    abs(final_elastic-initial_elastic),
                    abs(final_energy-initial_energy), abs(heat), 1.0)),
            "rejected_steps": rejected,
        },
        "decision": {
            "finite_wavelength_nonlinear_amplification": bool(
                records[-1]["density_contrast"] > 2.0*records[0]["density_contrast"]),
            "gnd_bearing_wall_qualified": False,
            "recrystallization_qualified": False,
            "reason": "reaction-disabled mechanism discriminator cannot qualify a wall or RX",
        },
        "claim_boundary": (
            "no reservoir reactions, front motion, grain allocation, or material calibration"),
        "artifacts": {str(figure.relative_to(ROOT)): _sha(figure)},
    }
    path = OUT/"result.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"initial": records[0], "final": records[-1],
                      "ledger": result["ledger"],
                      "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
