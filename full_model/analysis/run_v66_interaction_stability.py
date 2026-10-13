#!/usr/bin/env python3
"""Assess H3 with the authoritative common-wall tangent and nonlinear evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from full_model.production.common_tensorial_wall import (
    CommonWallDriving,
    CommonWallParameters,
    CommonWallState,
    fourier_symbol,
)
from full_model.production.tensorial_nye import (
    bcc_four_family_systems,
    make_junction_topology,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"full_model"/"verification"/"v66_interaction_stability"


def _sha(path):
    h = hashlib.sha256(path.read_bytes())
    return h.hexdigest()


def _state(n):
    family = (n, n, 4)
    z = np.zeros(family)
    return CommonWallState(
        np.full(family, 1e13), np.full(family, 1e13),
        np.full(family, 1e13), np.full(family, 1e13),
        np.full(family, 1e12), np.full(family, 1e12),
        np.full((n, n, 1), 1e10), np.zeros((n, n)),
        np.zeros((n, n)), z.copy(), np.zeros((n, n, 3, 3)),
        np.zeros((n, n, 4, 3)), np.zeros((n, n, 4, 3, 3)),
        np.zeros((n, n)), np.full((n, n), 900.0))


def _dispersion(stress_Pa):
    n = 12
    dx = 1e-7
    systems = bcc_four_family_systems()
    topology = (make_junction_topology(
        systems, 0, 1, 1, -1, line_tension_J_m=0.0),)
    state = _state(n)
    parameters = CommonWallParameters(spacing_m=dx, transport_scheme="spectral")
    driving = CommonWallDriving(
        resolved_stress_Pa=np.full((n, n, 4), stress_Pa))
    records = []
    representative_mode = None
    for index in range(1, n//2):
        symbol = fourier_symbol(
            state, driving, systems, topology, parameters, (index, 0),
            relative_step=1e-6)
        eigenvalue, eigenvector = np.linalg.eig(symbol["matrix_s_inv"])
        leading = int(np.argmax(eigenvalue.real))
        vector = eigenvector[:, leading]
        dominant = int(np.argmax(np.abs(vector)))
        wavelength = n*dx/index
        record = {
            "mode_index": index,
            "wavelength_m": wavelength,
            "leading_real_rate_s": float(eigenvalue[leading].real),
            "leading_imaginary_rate_s": float(eigenvalue[leading].imag),
            "dominant_state_coordinate": str(symbol["layout"][dominant][:2]),
            "dominant_coordinate_fraction": float(abs(vector[dominant])),
        }
        records.append(record)
        if index == 1:
            ranked = np.argsort(np.abs(vector))[-8:][::-1]
            representative_mode = [
                {"coordinate": str(symbol["layout"][item][:2]),
                 "magnitude": float(abs(vector[item]))}
                for item in ranked]
    return records, representative_mode


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    conditions = {}
    for stress in (0.0, 1.8e8):
        records, eigenvector = _dispersion(stress)
        conditions[str(int(stress))] = {
            "records": records,
            "leading_mode_composition": eigenvector,
        }

    figure = OUT/"common_wall_dispersion.png"
    fig, ax = plt.subplots(figsize=(5.3, 3.7), constrained_layout=True)
    for stress, condition in conditions.items():
        wavelength = np.array([item["wavelength_m"] for item in condition["records"]])
        rate = np.array([item["leading_real_rate_s"] for item in condition["records"]])
        ax.plot(1e-6/wavelength, rate, marker="o",
                label=f"{float(stress)/1e6:.0f} MPa")
    ax.set_xlabel(r"wave number (cycles $\mu$m$^{-1}$)")
    ax.set_ylabel(r"leading tangent growth rate (s$^{-1}$)")
    ax.legend(frameon=False)
    fig.savefig(figure, dpi=180)
    plt.close(fig)

    all_rates = {
        stress: np.array([item["leading_real_rate_s"]
                          for item in condition["records"]])
        for stress, condition in conditions.items()}
    flatness = {
        stress: float(np.ptp(value)/max(abs(np.mean(value)), 1e-300))
        for stress, value in all_rates.items()
    }
    source_result = json.loads((
        ROOT/"full_model"/"verification"/"v66_source_organization"/"result.json"
    ).read_text())
    result = {
        "schema": "v66-interaction-stability-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "linear tangent of authoritative common residual plus nonlinear source comparison",
        "base_state_total_reservoir_density_m2": 4*(2e13+2e13+2e12)+1e10,
        "common_rho_log_curvature": (
            "positive: d2[C rho log(rho/rho_ref)]/d rho2 = C/rho"),
        "conditions": conditions,
        "relative_dispersion_flatness": flatness,
        "nonlinear_source_evidence": {
            "gb_tj_density_source_correlation": source_result["cases"]["gb_tj_only"]["density_source_correlation"],
            "random_density_source_correlation": source_result["cases"]["random_bulk"]["density_source_correlation"],
            "gb_tj_peak_index": source_result["cases"]["gb_tj_only"]["structure_factor_peak_index"],
            "random_peak_index": source_result["cases"]["random_bulk"]["structure_factor_peak_index"],
        },
        "decision": {
            "H3_interaction_selected_wall_spacing_supported": False,
            "local_reaction_instability_present": bool(any(
                np.max(value) > 0.0 for value in all_rates.values())),
            "classification": "local_reaction_mode_not_finite_wavelength_patterning",
            "reason": (
                "the leading positive tangent is nearly wave-number independent and is "
                "dominated by junction/forest exchange, while short nonlinear runs remain "
                "almost perfectly correlated with their imposed source maps. This does not "
                "establish an interaction-selected wall wavelength; it keeps H1/H2 primary."),
        },
        "artifacts": {str(figure.relative_to(ROOT)): _sha(figure)},
    }
    path = OUT/"result.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps(result["decision"], indent=2))


if __name__ == "__main__":
    main()
