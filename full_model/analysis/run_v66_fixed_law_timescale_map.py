#!/usr/bin/env python3
"""Fixed-law Arrhenius competition map for H3 formation and recovery."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from full_model.analysis.run_v66_interaction_stability import _state
from full_model.analysis.run_v66_production_groma_tangent import (
    candidate_parameters, mobile_symbol,
)
from full_model.production.common_tensorial_wall import (
    CommonWallDriving, wall_residual,
)
from full_model.production.tensorial_nye import (
    bcc_four_family_systems, make_junction_topology,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"full_model"/"verification"/"v66_fixed_law_timescale_map"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n = 24
    spacing = 2.0e-7
    systems = bcc_four_family_systems()
    topologies = (make_junction_topology(
        systems, 0, 1, 1, -1, line_tension_J_m=0.0),)
    parameters, gb2 = candidate_parameters(spacing)
    parameters = replace(
        parameters, polarized_wall_absorption_enabled=True,
        wall_absorption_polarization_power=1.0)
    temperatures = (700.0, 900.0, 1100.0, 1300.0, 1500.0)
    stresses = (40e6, 50e6, 60e6, 70e6, 80e6)
    strain_window = 1.0e-2
    target_efolds = 5.0
    records = []
    for temperature in temperatures:
        state = replace(
            _state(n), temperature_K=np.full((n, n), temperature))
        # Exactly sign-balanced reference for local reaction competition.
        balanced = replace(
            state,
            mobile_minus_m2=state.mobile_plus_m2.copy(),
            forest_minus_m2=state.forest_plus_m2.copy(),
            wall_minus_m2=state.wall_plus_m2.copy())
        # A prospective 10% family polarization measures the separately
        # implemented compatible-absorption timescale without selecting its
        # spatial amplitude from a PF result.
        polarized = replace(balanced, mobile_plus_m2=(
            1.1*balanced.mobile_plus_m2), mobile_minus_m2=(
            .9*balanced.mobile_minus_m2))
        for stress in stresses:
            driving = CommonWallDriving(resolved_stress_Pa=np.full(
                state.mobile_plus_m2.shape, stress))
            mode_rows = []
            for mode in range(1, n//2):
                (_, full_values, _, transport_values, _) = mobile_symbol(
                    balanced, driving, systems, topologies, parameters, mode)
                mode_rows.append({
                    "mode": mode,
                    "transport_growth_s_inv": float(np.max(
                        transport_values.real)),
                    "full_mobile_projected_growth_s_inv": float(np.max(
                        full_values.real)),
                })
            selected = max(mode_rows, key=lambda item: item[
                "transport_growth_s_inv"])
            selected_mode = selected["mode"]
            growth = selected["transport_growth_s_inv"]
            concurrent_growth = selected[
                "full_mobile_projected_growth_s_inv"]
            full_selected = max(mode_rows, key=lambda item: item[
                "full_mobile_projected_growth_s_inv"])
            residual = wall_residual(
                balanced, driving, systems, topologies, parameters)
            channels = residual.channel_rates_m2_s
            mobile_scale = float(np.mean(
                balanced.mobile_plus_m2+balanced.mobile_minus_m2))
            forest_scale = float(np.mean(
                balanced.forest_plus_m2+balanced.forest_minus_m2))
            wall_scale = float(np.mean(
                balanced.wall_plus_m2+balanced.wall_minus_m2))
            local_fractional_rates = {
                "lock_net_s_inv": float(np.mean(np.abs(
                    channels["lock_plus"])+np.abs(
                    channels["lock_minus"]))/mobile_scale),
                "wall_exchange_net_s_inv": float(np.mean(np.abs(
                    channels["wall_plus"])+np.abs(
                    channels["wall_minus"]))/max(forest_scale, wall_scale)),
                "annihilation_s_inv": float(np.mean(
                    2.0*channels["annihilation_pairs"])/mobile_scale),
                "junction_net_s_inv": float(np.mean(np.abs(
                    channels["junction"]))/max(forest_scale, 1.0)),
            }
            gross_turnover_rates = {
                "lock_gross_s_inv": float(np.mean(
                    channels["lock_plus_turnover"]
                    +channels["lock_minus_turnover"])/mobile_scale),
                "wall_exchange_gross_s_inv": float(np.mean(
                    channels["wall_plus_turnover"]
                    +channels["wall_minus_turnover"])
                    /max(forest_scale, wall_scale)),
                "junction_gross_s_inv": float(np.mean(
                    channels["junction_turnover"])/max(forest_scale, 1.0)),
            }
            polarized_residual = wall_residual(
                polarized, driving, systems, topologies, parameters)
            absorbed = (polarized_residual.channel_rates_m2_s["wall_plus"]
                        +polarized_residual.channel_rates_m2_s["wall_minus"])
            signed_scale = float(np.mean(np.abs(
                polarized.mobile_plus_m2-polarized.mobile_minus_m2)))
            absorption_rate = float(np.mean(np.maximum(absorbed, 0.0))
                                    /max(signed_scale, 1.0))
            critical_rate = max(growth, 0.0)*strain_window/target_efolds
            records.append({
                "temperature_K": temperature,
                "resolved_stress_Pa": stress,
                "selected_mode": selected_mode,
                "selected_wavelength_m": n*spacing/selected_mode,
                "transport_growth_rate_s_inv": growth,
                "full_mobile_projected_growth_at_transport_mode_s_inv": (
                    concurrent_growth),
                "reaction_shift_at_transport_mode_s_inv": (
                    concurrent_growth-growth),
                "full_mobile_projected_selected_mode": full_selected["mode"],
                "full_mobile_projected_selected_growth_s_inv": full_selected[
                    "full_mobile_projected_growth_s_inv"],
                "five_efold_time_s": (None if growth <= 0.0 else target_efolds/growth),
                "one_percent_window_maximum_shear_rate_s_inv": critical_rate,
                "local_fractional_rates": local_fractional_rates,
                "gross_turnover_rates": gross_turnover_rates,
                "transport_retention_ratio": (
                    concurrent_growth/max(growth, 1e-300)),
                "ten_percent_polarization_absorption_rate_s_inv": absorption_rate,
            })

    figure = OUT/"timescale_competition.png"
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5), constrained_layout=True)
    for temperature in temperatures:
        subset = [row for row in records if row["temperature_K"] == temperature]
        x = [row["resolved_stress_Pa"]*1e-6 for row in subset]
        axes[0].plot(x, [row["transport_growth_rate_s_inv"] for row in subset],
                     marker="o", label=f"{temperature:g} K")
        axes[1].plot(x, [row[
            "full_mobile_projected_growth_at_transport_mode_s_inv"]
            for row in subset], marker="o", label=f"{temperature:g} K")
    axes[0].axhline(0.0, color="black", linewidth=.7)
    axes[0].set(xlabel="resolved stress (MPa)",
                ylabel=r"finite-mode growth rate (s$^{-1}$)")
    axes[1].axhline(0.0, color="black", linewidth=.7)
    axes[1].set(xlabel="resolved stress (MPa)",
                ylabel=r"full mobile-projected rate (s$^{-1}$)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(figure, dpi=180)
    plt.close(fig)

    positive = [row for row in records if row["transport_growth_rate_s_inv"] > 0.0]
    best_concurrent = max(records, key=lambda row: row[
        "full_mobile_projected_growth_at_transport_mode_s_inv"])
    fastest_growth = max(records, key=lambda row: row[
        "transport_growth_rate_s_inv"])
    result = {
        "schema": "v66-fixed-law-timescale-map-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "grid": [n, n], "spacing_m": spacing,
        "temperatures_K": list(temperatures),
        "resolved_stresses_Pa": list(stresses),
        "fixed_parameters": {
            "A": 1.25, "D": .5, "reduced_modulus_b2_J_m": gb2,
            "glide_barrier_eV": parameters.glide_barrier_eV,
            "lock_barrier_eV": parameters.lock_barrier_eV,
            "wall_barrier_eV": parameters.wall_barrier_eV,
            "annihilation_barrier_eV": parameters.annihilation_barrier_eV,
            "junction_barrier_eV": parameters.junction_barrier_eV,
            "order_barrier_eV": parameters.order_barrier_eV,
            "exp_a": parameters.exp_a, "exp_n": parameters.exp_n,
            "exp_floor": parameters.exp_floor,
            "taylor_alpha": parameters.taylor_alpha,
        },
        "loading_exposure_rule": {
            "strain_window": strain_window,
            "target_efolds": target_efolds,
            "maximum_shear_rate": "lambda * strain_window / target_efolds",
        },
        "records": records,
        "extrema": {"fastest_transport_growth": fastest_growth,
                    "best_concurrent_mobile_projection": best_concurrent},
        "decision": {
            "positive_transport_conditions": len(positive),
            "positive_concurrent_mobile_projection": bool(
                best_concurrent[
                    "full_mobile_projected_growth_at_transport_mode_s_inv"] > 0.0),
            "concurrent_interior_mode_selected": bool(
                best_concurrent["full_mobile_projected_selected_mode"]
                not in (1, n//2-1)),
            "nonlinear_concurrent_run_authorized": bool(
                best_concurrent[
                    "full_mobile_projected_growth_at_transport_mode_s_inv"] > 0.0
                and best_concurrent["full_mobile_projected_selected_mode"]
                not in (1, n//2-1)),
        },
        "claim_boundary": (
            "homogeneous fixed-state timescale discriminator; rate exposure is "
            "an analytical residence estimate, not nonlinear wall or RX evidence"),
        "artifacts": {str(figure.relative_to(ROOT)): _sha(figure)},
    }
    (OUT/"result.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"extrema": result["extrema"],
                      "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
