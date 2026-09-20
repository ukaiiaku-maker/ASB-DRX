#!/usr/bin/env python3
"""Locate the first cross-grid short-wave discrepancy in one production Mura event."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from full_model.analysis.run_v34_finite_coupled_response import resolved_bicrystal
from full_model.analysis.run_v36_recurrent_physical_response import driving_at_time
from full_model.production.common_tensorial_wall import resolved_driving_components
from full_model.production.tensorial_nye import (
    nye_from_plastic_distortion, rotated_system_fields,
)
from full_model.production.v24_mechanical_wall import accepted_v24_mechanical_step
from full_model.production.wall_topology_supply import (
    accepted_compatible_mura_transport_capture_step,
)


LENGTH_M = 3.2e-6
DT_S = 4.8828125e-7


def rms(value):
    return float(np.sqrt(np.mean(np.abs(value)**2)))


def coefficients(value, half_width):
    n = value.shape[0]
    spectrum = np.fft.fftshift(
        np.fft.fftn(value, axes=(0, 1))/(n*n), axes=(0, 1))
    center = n//2
    return spectrum[center-half_width:center+half_width+1,
                    center-half_width:center+half_width+1]


def compare(a, b, half_width):
    ca = coefficients(a, half_width); cb = coefficients(b, half_width)
    scale = max(rms(ca), rms(cb), 1e-300)
    return {
        "complex_coefficient_relative_rms": rms(ca-cb)/scale,
        "n128_band_rms": rms(ca), "n192_band_rms": rms(cb),
        "n128_whole_rms": rms(a), "n192_whole_rms": rms(b),
        "whole_rms_amplitude_relative_difference": (
            abs(rms(a)-rms(b))/max(rms(a), rms(b), 1e-300)),
    }


def spectrum(field, bands=(8, 16, 24, 32, 48, 63)):
    n = field.shape[0]
    full = np.fft.fftshift(
        np.fft.fftn(field, axes=(0, 1))/(n*n), axes=(0, 1))
    total = float(np.sum(np.abs(full)**2))
    result = []
    for half in bands:
        if 2*half+1 > n:
            continue
        block = coefficients(field, half)
        result.append({
            "half_width": half,
            "retained_squared_norm_fraction": (
                float(np.sum(np.abs(block)**2))/max(total, 1e-300)),
        })
    return result


def run_grid(n):
    context = resolved_bicrystal(
        grid=n, length_m=LENGTH_M, interface_width_m=4e-7,
        temperature_K=1100.0, child_line_fraction=.35)
    state0 = context["state"].mechanical
    driving = driving_at_time(n, .01, "hold", 0.0, .5*DT_S)
    drive = resolved_driving_components(
        state0.common, driving, context["systems"], context["topologies"],
        context["wall_parameters"])
    _, slip_directions, _ = rotated_system_fields(
        context["systems"], state0.common.orientation_rad)
    velocity_plus = drive["speed_m_s"][..., None]*slip_directions
    started = time.perf_counter()
    state1, ledger = accepted_v24_mechanical_step(
        state0, driving, context["capture_support"], context["systems"],
        context["topologies"], context["wall_parameters"],
        context["extensive_parameters"], context["topology_kinetics"], DT_S,
        topology_route_enabled=False,
        mura_transport_operator="compatible_dealiased")
    elapsed = time.perf_counter()-started
    capture = ledger["transport_capture"]
    accepted_dt = float(ledger["accepted_dt_s"])
    beta_rate = (state1.common.beta_p-state0.common.beta_p)/accepted_dt
    curl_rate = nye_from_plastic_distortion(
        beta_rate, context["spacing_m"])
    fields = {
        "capture_support": context["capture_support"].astype(float),
        "raw_stress": ledger["raw_stress_Pa"],
        "effective_stress": ledger["effective_stress_Pa"],
        "speed": drive["speed_m_s"],
        "velocity_plus": velocity_plus,
        "initial_alignment_plus": state0.reservoir_alignment.mobile_plus_m2,
        "swept_plus": capture["swept_product_plus_m_s"],
        "swept_minus": capture["swept_product_minus_m_s"],
        "alignment_rate_plus": capture["alignment_rate_plus_m2_s"],
        "alignment_rate_minus": capture["alignment_rate_minus_m2_s"],
        "captured_line": sum(capture["sign"][sign]["captured_line_m2"]
                             for sign in ("plus", "minus")),
        "line_stretching": sum(
            capture["sign"][sign]["mura_line_stretching_m2"]
            for sign in ("plus", "minus")),
        "beta_rate": beta_rate,
        "curl_nye_rate": curl_rate,
        "orientation_rate": (
            state1.common.orientation_rad-state0.common.orientation_rad)
            /accepted_dt,
    }
    ordering = ledger["ordering_thermodynamics"]
    return fields, {
        "grid": n, "spacing_m": context["spacing_m"],
        "requested_dt_s": DT_S, "accepted_dt_s": accepted_dt,
        "wall_seconds": elapsed,
        "event_scale": float(ledger["mura_event_scale"]),
        "family_event_scales": [float(x) for x in
                                 ledger["mura_family_event_scales"]],
        "operator": ledger["mura_transport_operator"],
        "ordering_dispatch": ordering["stiff_dispatch"],
        "ordering_integration_method": ordering["integration_method"],
        "ordering_active_degrees_of_freedom": ordering[
            "active_degrees_of_freedom"],
        "ordering_maximum_attempt_exposure": ordering[
            "maximum_attempt_exposure"],
        "ordering_accessibility_passed": ordering.get(
            "asymptotic_state_accessibility_passed"),
        "field_spectra": {name: spectrum(value) for name, value in fields.items()},
    }


def capture_length_sensitivity():
    prepared = {}
    for n in (128, 192):
        context = resolved_bicrystal(
            grid=n, length_m=LENGTH_M, interface_width_m=4e-7,
            temperature_K=1100.0, child_line_fraction=.35)
        state = context["state"].mechanical
        driving = driving_at_time(n, .01, "hold", 0.0, .5*DT_S)
        drive = resolved_driving_components(
            state.common, driving, context["systems"], context["topologies"],
            context["wall_parameters"])
        _, slip_directions, _ = rotated_system_fields(
            context["systems"], state.common.orientation_rad)
        velocity = drive["speed_m_s"][..., None]*slip_directions
        prepared[n] = context, state, velocity
    rows = []
    for length in (0.0, 5e-8, 1e-7, 2e-7, 4e-7, 8e-7):
        captured = {}
        for n, (context, state, velocity) in prepared.items():
            _, _, ledger = accepted_compatible_mura_transport_capture_step(
                state.density, state.reservoir_alignment, velocity, -velocity,
                context["capture_support"], context["systems"],
                state.common.orientation_rad, context["spacing_m"], DT_S,
                context["topologies"],
                capture_deposition_length_m=length)
            captured[n] = sum(
                ledger["sign"][sign]["captured_line_m2"]
                for sign in ("plus", "minus"))
        rows.append({
            "capture_deposition_length_m": length,
            "halfwidth8_error": compare(
                captured[128], captured[192], 8)[
                    "complex_coefficient_relative_rms"],
            "halfwidth24_error": compare(
                captured[128], captured[192], 24)[
                    "complex_coefficient_relative_rms"],
        })
    return {
        "registered_before_outcomes_m": [0.0, 5e-8, 1e-7, 2e-7, 4e-7, 8e-7],
        "selection_basis": (
            "400 nm resolves the pre-existing 450 nm trap half-width; the "
            "capture depth is distinct from mesh and atomistic core radius"),
        "rows": rows,
    }


def manufactured_origin_audit():
    mode = (5, 3)
    records = {}
    corrected = {}
    for n in (128, 192):
        dx = LENGTH_M/n
        # Production analytic fixtures use the same physical origin -L/2 on
        # both grids; the alternate cell-center convention is tested too.
        x = -0.5*LENGTH_M+np.arange(n)*dx
        y = -0.5*LENGTH_M+np.arange(n)*dx
        field = np.cos(2*np.pi*(mode[0]*x[:, None]+mode[1]*y[None, :])/LENGTH_M)
        c = coefficients(field, 8)
        records[str(n)] = [float(c[8+mode[0], 8+mode[1]].real),
                           float(c[8+mode[0], 8+mode[1]].imag)]
        xc = -0.5*LENGTH_M+(np.arange(n)+.5)*dx
        yc = -0.5*LENGTH_M+(np.arange(n)+.5)*dx
        fc = np.cos(2*np.pi*(mode[0]*xc[:, None]+mode[1]*yc[None, :])/LENGTH_M)
        coefficient = coefficients(fc, 8)[8+mode[0], 8+mode[1]]
        phase = np.exp(-1j*2*np.pi*(mode[0]+mode[1])*.5/n)
        coefficient *= phase
        corrected[str(n)] = [float(coefficient.real), float(coefficient.imag)]
    return {
        "production_coordinate_origin_m": -0.5*LENGTH_M,
        "mode": list(mode), "node_aligned_coefficients": records,
        "cell_centered_coefficients_after_known_origin_correction": corrected,
        "optimized_field_shift_used": False,
    }


def main():
    fields128, row128 = run_grid(128)
    fields192, row192 = run_grid(192)
    names = tuple(fields128)
    comparisons = {name: {str(half): compare(
        fields128[name], fields192[name], half)
        for half in (8, 16, 24, 32, 48, 63)} for name in names}
    ordered = (
        "capture_support", "initial_alignment_plus", "raw_stress",
        "effective_stress", "speed", "velocity_plus", "swept_plus",
        "alignment_rate_plus", "captured_line", "line_stretching",
        "beta_rate", "curl_nye_rate", "orientation_rate")
    first = next((name for name in ordered
                  if comparisons[name]["24"][
                      "complex_coefficient_relative_rms"] > .05), None)
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip()
    payload = {
        "schema": "asb-drx/v46/mura-suboperator-audit/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": source,
        "physical_domain_m": LENGTH_M,
        "grids": {"128": row128, "192": row192},
        "coordinate_and_symbol_audit": manufactured_origin_audit(),
        "capture_length_sensitivity": capture_length_sensitivity(),
        "comparisons": comparisons,
        "causal_order_for_diagnosis": list(ordered),
        "first_field_above_five_percent_at_halfwidth24": first,
        "interpretation": (
            "diagnostic fields are recorded in causal order; a discrepancy in "
            "an input prevents assigning later disagreement to the later operator"),
        "production_state_filtered": False,
    }
    output = Path("full_model/verification/v46_mura_suboperator_audit.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps({
        "first": first,
        "curl_error_h24": comparisons["curl_nye_rate"]["24"][
            "complex_coefficient_relative_rms"],
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
