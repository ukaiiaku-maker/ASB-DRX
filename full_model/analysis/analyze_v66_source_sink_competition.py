#!/usr/bin/env python3
"""Map fixed-law Hall--Petch source competition with neutral GB/TJ sinks.

The source geometry follows Azizi--Dillon (Acta Materialia 316 (2026)
122461): ``N=b^2/d^2``, ``phi=sqrt(d/b)``, and one-event strain ``b/d``.
The source-density rate is therefore ``hazard/d^2`` per sign and family.
The neutral-sink rate is the V66 thermal EXP-floor hypothesis.  The resulting
``rho_eq=source_density_rate/sink_rate`` is an upper bound because the common
transaction can reject source events that cannot pay their line-storage cost.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from full_model.production.arrhenius_dislocation_sources import (
    ArrheniusSourceParameters, hall_petch_nucleation_hazard_Hz,
)
from full_model.production.arrhenius_kinetics import (
    ActivatedProcess, EV_J, activated_rate_array_s, exp_floor_enthalpy_j,
)
from full_model.production.multigrain_source_transaction import (
    NeutralBoundarySinkParameters,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT/"full_model"/"verification"/"v66_source_sink_competition"


def sink_rate_s(temperature_K, parameters):
    process = ActivatedProcess(
        "neutral-boundary-pair-sink", parameters.attempt_frequency_s,
        0.0, parameters.attempt_frequency_s)
    enthalpy = exp_floor_enthalpy_j(
        0.0, parameters.activation_enthalpy_eV*EV_J,
        parameters.critical_stress_Pa, parameters.exp_a, parameters.exp_n,
        parameters.barrier_floor_fraction)
    return activated_rate_array_s(process, enthalpy, temperature_K)


def calculate():
    temperatures = np.asarray([650.0, 800.0, 1000.0, 1200.0])
    stresses = np.linspace(0.0, 600.0e6, 241)
    grain_sizes = np.asarray([0.5e-6, 1.0e-6, 2.4e-6, 5.0e-6, 10.0e-6])
    sink = NeutralBoundarySinkParameters()
    rows = []
    surfaces = {}
    for grain_size in grain_sizes:
        source = ArrheniusSourceParameters(grain_size_m=float(grain_size))
        concentration = np.sqrt(grain_size/source.burgers_m)
        floor_stress = ((source.activation_enthalpy_eV-source.barrier_floor_eV)
                        *EV_J/(concentration
                              *source.intrinsic_activation_volume_m3))
        rho_equilibrium = np.empty((len(temperatures), len(stresses)))
        event_rate = np.empty_like(rho_equilibrium)
        for i, temperature in enumerate(temperatures):
            event_rate[i] = hall_petch_nucleation_hazard_Hz(
                stresses, temperature, source)
            density_rate = event_rate[i]/grain_size**2
            annihilation_rate = float(sink_rate_s(temperature, sink))
            rho_equilibrium[i] = density_rate/annihilation_rate
            floor_event_rate = float(hall_petch_nucleation_hazard_Hz(
                floor_stress, temperature, source))
            rows.append({
                "grain_size_m": float(grain_size),
                "temperature_K": float(temperature),
                "source_floor_stress_Pa": float(floor_stress),
                "thermal_sink_rate_s": annihilation_rate,
                "floor_source_event_rate_Hz_per_family": floor_event_rate,
                "floor_source_density_rate_m2_s_per_sign_family": (
                    floor_event_rate/grain_size**2),
                "floor_source_shear_rate_s_per_family": (
                    floor_event_rate*source.burgers_m/grain_size),
                "upper_bound_equilibrium_mobile_density_m2_per_sign_family": (
                    floor_event_rate/grain_size**2/annihilation_rate),
                "sink_half_life_s": float(np.log(2.0)/annihilation_rate),
            })
        surfaces[f"d_{grain_size:.6e}_m"] = {
            "event_rate_Hz_per_family": event_rate.tolist(),
            "upper_bound_equilibrium_mobile_density_m2_per_sign_family": (
                rho_equilibrium.tolist()),
        }
    return temperatures, stresses, grain_sizes, sink, rows, surfaces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    temperatures, stresses, grain_sizes, sink, rows, surfaces = calculate()

    payload = {
        "schema": "v66-source-sink-competition-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_reference": {
            "citation": ("Azizi and Dillon, Acta Materialia 316 (2026) "
                         "122461, doi:10.1016/j.actamat.2026.122461"),
            "closure": "nucleation-controlled Hall-Petch",
            "site_multiplicity": "b^2/d^2",
            "stress_concentration": "sqrt(d/b)",
            "event_strain": "b/d",
            "density_increment_per_event_per_sign_family_m2": "1/d^2",
        },
        "sink_parameters": asdict(sink),
        "stress_Pa": stresses.tolist(),
        "temperature_K": temperatures.tolist(),
        "grain_size_m": grain_sizes.tolist(),
        "summary": rows,
        "surfaces": surfaces,
        "interpretation": {
            "scope": ("fixed-state rate competition and upper bound only; "
                      "accepted source activity can be lower after complete-energy "
                      "qualification"),
            "spatial_site_fields": ("unit-mean GB/TJ and matched-random fields "
                                    "share these integrated rates"),
            "no_wall_claim": True,
            "no_material_calibration_claim": True,
        },
    }
    (args.out/"result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)+"\n")

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7), constrained_layout=True)
    for temperature in temperatures:
        selected = [row for row in rows if row["temperature_K"] == temperature]
        axes[0].loglog(
            grain_sizes*1e6,
            [row["upper_bound_equilibrium_mobile_density_m2_per_sign_family"]
             for row in selected], marker="o", label=f"{temperature:.0f} K")
    axes[0].set_xlabel("grain size, $d$ ($\\mu$m)")
    axes[0].set_ylabel("upper-bound $\\rho_{eq}$ per sign/family (m$^{-2}$)")
    axes[0].legend(frameon=False)
    selected = [row for row in rows if row["grain_size_m"] == 2.4e-6]
    axes[1].semilogy(
        temperatures, [1.0/row["floor_source_event_rate_Hz_per_family"]
                       for row in selected], marker="o",
        label="HP event waiting time")
    axes[1].semilogy(
        temperatures, [row["sink_half_life_s"] for row in selected],
        marker="s", label="neutral-pair half-life")
    axes[1].set_xlabel("temperature (K)")
    axes[1].set_ylabel("fixed-state time scale at $d=2.4$ $\\mu$m (s)")
    axes[1].legend(frameon=False)
    fig.savefig(args.out/"source_sink_competition.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"result": str(args.out/"result.json"),
                      "figure": str(args.out/"source_sink_competition.png"),
                      "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
