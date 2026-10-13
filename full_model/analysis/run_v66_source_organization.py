#!/usr/bin/env python3
"""Compare GB/TJ-only and matched random dislocation sources in common transport."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from full_model.production.arrhenius_dislocation_sources import (
    ArrheniusSourceParameters,
    apply_sign_balanced_pair_source,
    boundary_tj_site_field,
    pair_source_rate_m2_s,
    randomized_matched_site_field,
)
from full_model.production.common_tensorial_wall import (
    CommonWallDriving,
    CommonWallParameters,
    CommonWallState,
    accepted_euler_step,
)
from full_model.production.tensorial_nye import bcc_four_family_systems


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"full_model"/"verification"/"v66_source_organization"


def _sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _periodic_supports(n, length_m):
    x = np.arange(n)[:, None]*length_m/n
    y = np.arange(n)[None, :]*length_m/n
    seeds = np.array(((.18, .20), (.72, .18), (.22, .73), (.70, .70)))*length_m
    distances = []
    for sx, sy in seeds:
        dx = np.minimum(abs(x-sx), length_m-abs(x-sx))
        dy = np.minimum(abs(y-sy), length_m-abs(y-sy))
        distances.append(dx*dx+dy*dy)
    distances = np.stack(distances)
    # The scale gives a roughly two-cell diffuse ownership transition.
    logits = -distances/(2.0*length_m*(2.0*length_m/n))
    logits -= np.max(logits, axis=0)
    weights = np.exp(logits)
    return weights/np.sum(weights, axis=0)


def _initial_state(n, temperature_K):
    shape = (n, n, 4)
    background = np.full(shape, 1e8)
    zeros = np.zeros(shape)
    return CommonWallState(
        background.copy(), background.copy(), zeros.copy(), zeros.copy(),
        zeros.copy(), zeros.copy(), np.zeros((n, n, 0)),
        np.zeros((n, n)), np.zeros((n, n)), np.zeros(shape),
        np.zeros((n, n, 3, 3)), np.zeros((n, n, 4, 3)),
        np.zeros((n, n, 4, 3, 3)), np.zeros((n, n)),
        np.full((n, n), temperature_K))


def _metrics(state, site, boundary):
    total = np.sum(
        state.mobile_plus_m2+state.mobile_minus_m2
        +state.forest_plus_m2+state.forest_minus_m2
        +state.wall_plus_m2+state.wall_minus_m2, axis=2)
    signed = np.sum(state.mobile_plus_m2-state.mobile_minus_m2, axis=2)
    gnd = np.sqrt(np.sum(state.family_nye_m1**2, axis=(2, 3, 4)))
    orientation = state.orientation_rad
    def corr(a, b):
        aa = np.ravel(a)-np.mean(a); bb = np.ravel(b)-np.mean(b)
        return float(np.sum(aa*bb)/np.sqrt(np.sum(aa*aa)*np.sum(bb*bb)))
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(total-np.mean(total))))**2
    center = tuple(value//2 for value in total.shape)
    spectrum[center] = 0.0
    peak = np.unravel_index(np.argmax(spectrum), spectrum.shape)
    k_index = np.hypot(peak[0]-center[0], peak[1]-center[1])
    return {
        "mean_total_density_m2": float(np.mean(total)),
        "total_density_cv": float(np.std(total)/np.mean(total)),
        "rms_signed_mobile_m2": float(np.sqrt(np.mean(signed*signed))),
        "rms_nye_m1": float(np.sqrt(np.mean(gnd*gnd))),
        "rms_orientation_rad": float(np.sqrt(np.mean(orientation*orientation))),
        "density_source_correlation": corr(total, site),
        "density_boundary_correlation": corr(total, boundary),
        "structure_factor_peak_index": float(k_index),
        "total_density_m2": total,
        "signed_mobile_m2": signed,
        "nye_norm_m1": gnd,
        "orientation_rad": orientation,
    }


def _run(mode, site, boundary, p_source, *, n, length_m, temperature_K,
         stress_Pa, duration_s, steps):
    systems = bcc_four_family_systems(p_source.burgers_m)
    state = _initial_state(n, temperature_K)
    dx = length_m/n
    parameters = CommonWallParameters(
        spacing_m=dx, transport_scheme="upwind",
        mobile_correlation_diffusivity_m2_s=2e-13,
        multiplication_coefficient=0.0, wall_order_enabled=False,
        lock_barrier_eV=5.0, unlock_barrier_eV=5.0,
        wall_barrier_eV=5.0, annihilation_barrier_eV=5.0,
        junction_barrier_eV=5.0, order_barrier_eV=5.0,
        thermal_diffusivity_m2_s=0.0)
    speed = np.full((n, n, 4), 2.0e-5)
    driving = CommonWallDriving(glide_speed_m_s=speed,
                                resolved_stress_Pa=np.full_like(speed, stress_Pa))
    source_rate = pair_source_rate_m2_s(
        site, stress_Pa, temperature_K, p_source)
    dt = duration_s/steps
    source_balance = 0.0
    minimum_scale = 1.0
    for _ in range(steps):
        state, source = apply_sign_balanced_pair_source(
            state, source_rate, dt, source_mode=mode)
        source_balance = max(source_balance, abs(source.line_balance_residual_m2))
        state, _, scale = accepted_euler_step(
            state, driving, systems, (), parameters, dt)
        minimum_scale = min(minimum_scale, scale)
    result = _metrics(state, site, boundary)
    result.update(
        source_line_balance_max_abs_m2=source_balance,
        minimum_transport_acceptance_scale=minimum_scale,
        integrated_pair_source_m2=float(np.mean(source_rate)*duration_s),
        dt_s=dt,
    )
    return result


def _plot(cases, site_fields, path):
    fig, axes = plt.subplots(2, 4, figsize=(12.0, 6.0), constrained_layout=True)
    names = ("gb_tj_only", "random_bulk")
    for row, name in enumerate(names):
        case = cases[name]
        panels = (
            (site_fields[name], "normalized source sites", "viridis"),
            (case["total_density_m2"]/1e14,
             r"total density ($10^{14}$ m$^{-2}$)", "magma"),
            (case["nye_norm_m1"]/1e6, r"Nye norm ($10^6$ m$^{-1}$)", "cividis"),
            (np.degrees(case["orientation_rad"]), "orientation (deg)", "coolwarm"),
        )
        for col, (value, label, cmap) in enumerate(panels):
            image = axes[row, col].imshow(value.T, origin="lower", cmap=cmap,
                                          interpolation="nearest")
            axes[row, col].set_xlabel("x cell")
            axes[row, col].set_ylabel("y cell")
            if col == 0:
                axes[row, col].text(
                    .02, .98, "GB/TJ sites" if row == 0 else "random bulk sites",
                    transform=axes[row, col].transAxes, ha="left", va="top",
                    color="white", fontsize=9)
            cbar = fig.colorbar(image, ax=axes[row, col], shrink=.82)
            cbar.set_label(label)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n = 48
    length_m = 4.8e-6
    temperature_K = 900.0
    stress_Pa = 1.8e8
    duration_s = 2.0e-3
    steps = 200
    supports = _periodic_supports(n, length_m)
    boundary = boundary_tj_site_field(supports, tj_site_weight=2.0)
    random = randomized_matched_site_field(boundary, seed=66)
    p_source = ArrheniusSourceParameters(
        grain_size_m=length_m/2.0, pair_increment_m2=1.0e14,
        tj_site_weight=2.0)
    sites = {"gb_tj_only": boundary, "random_bulk": random}
    cases = {
        name: _run(name, site, boundary, p_source, n=n, length_m=length_m,
                   temperature_K=temperature_K, stress_Pa=stress_Pa,
                   duration_s=duration_s, steps=steps)
        for name, site in sites.items()
    }
    figure = OUT/"source_organization_fields.png"
    _plot(cases, sites, figure)
    serial_cases = {
        name: {key: value for key, value in case.items()
               if not isinstance(value, np.ndarray)}
        for name, case in cases.items()
    }
    result = {
        "schema": "v66-source-organization-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "source-location discriminator using common signed transport",
        "production_qualification": False,
        "reason_not_production_qualified": (
            "the Hall-Petch hazard is physical, but its sign-balanced pair transition "
            "kernel and source work/heat ledger remain a V66 hypothesis"),
        "source_parameters": asdict(p_source),
        "numerics": {
            "grid": [n, n], "length_m": length_m,
            "temperature_K": temperature_K, "resolved_stress_Pa": stress_Pa,
            "duration_s": duration_s, "steps": steps,
            "common_transport": "upwind production residual",
            "random_control": "exact permutation of GB/TJ site weights",
        },
        "cases": serial_cases,
        "decision": {
            "matched_integrated_source": bool(np.isclose(
                cases["gb_tj_only"]["integrated_pair_source_m2"],
                cases["random_bulk"]["integrated_pair_source_m2"], rtol=2e-15)),
            "location_changes_organization": bool(
                abs(cases["gb_tj_only"]["density_boundary_correlation"]
                    -cases["random_bulk"]["density_boundary_correlation"]) > .1),
            "physical_wall_claim": False,
            "interpretation": (
                "source localization alone creates distinguishable density and GND "
                "heterogeneity, but finite-time structural-wall qualification requires "
                "the absorption/network reaction and complete source-energy transaction"),
        },
        "artifacts": {
            str(figure.relative_to(ROOT)): _sha(figure),
        },
    }
    result_path = OUT/"result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps(result["decision"], indent=2))


if __name__ == "__main__":
    main()
