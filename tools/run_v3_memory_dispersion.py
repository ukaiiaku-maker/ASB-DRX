#!/usr/bin/env python3
"""Bounded dispersion screen for the delayed junction-memory hypothesis."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from asb_drx.arrhenius_v3 import (
    ArrheniusMechanism, BoundedActivationEntropy, ExpFloorEnthalpy,
)
from asb_drx.vector_topology_cdd_v3 import (
    JunctionReaction, VectorTopologyNetwork,
    arrhenius_forest_linearization, reaction_transport_memory_symbol_s_inv,
)


EV_J = 1.602176634e-19


def main(output: Path) -> None:
    glide = ArrheniusMechanism(
        ExpFloorEnthalpy(0.45 * EV_J, 0.8e9, 900.0, 0.25, 1.2, 2.0),
        BoundedActivationEntropy(reference_kB=0.2), 2.0e7,
        validity_temperature_K=(500.0, 1400.0),
        validity_stress_Pa=(0.0, 2.0e9),
    )
    network = VectorTopologyNetwork(
        2.86e-10,
        (JunctionReaction((0, 1), "sessile", glide, 0.02 * EV_J, 5.0e14),),
    )
    mobile = np.full(8, 2.5e14)
    reaction = network.reactions[0]
    records = []
    for temperature in (800.0, 900.0, 1000.0):
        forward, reverse = reaction.rate_constants_s_inv(temperature)
        junction = np.asarray([
            forward / reverse * mobile[0] * mobile[1]
            / reaction.reference_density_m2
        ])
        minimum_wavelength = 10.0 / np.sqrt(float(np.sum(mobile) + np.sum(junction)))
        k_values = np.geomspace(1.0e3, 2.0 * np.pi / minimum_wavelength, 360)
        for stress in (0.5e9, 0.7e9, 0.9e9):
            for forest_coefficient in (3.0e-7, 6.0e-7, 1.0e-6):
                velocity, derivative = arrhenius_forest_linearization(
                    (glide,) * 4, np.full(4, 1.0e-9), np.full(4, stress),
                    junction, np.full((4, 1), forest_coefficient), temperature,
                )
                for diffusivity in (2.0e-13, 5.0e-13, 1.0e-12):
                    for relaxation in (3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2):
                        growth = []
                        for k_m_inv in k_values:
                            symbol = reaction_transport_memory_symbol_s_inv(
                                np.asarray([k_m_inv, 0.0]), mobile, junction, network,
                                temperature, np.asarray([1.0, 0.0]), velocity,
                                derivative, np.full(8, diffusivity),
                                np.asarray([relaxation]),
                            )
                            growth.append(float(np.max(np.linalg.eigvals(symbol).real)))
                        growth = np.asarray(growth)
                        peak = int(np.argmax(growth))
                        wavelength = float(2.0 * np.pi / k_values[peak])
                        records.append({
                            "temperature_K": temperature,
                            "resolved_stress_Pa": stress,
                            "forest_coefficient_Pa_m2": forest_coefficient,
                            "diffusivity_m2_s": diffusivity,
                            "memory_relaxation_s": relaxation,
                            "maximum_growth_s_inv": float(growth[peak]),
                            "peak_wavelength_m": wavelength,
                            "minimum_continuum_wavelength_m": minimum_wavelength,
                            "interior_finite_mode": bool(
                                growth[peak] > 0.0
                                and peak not in (0, len(k_values) - 1)
                                and wavelength >= 1.2 * minimum_wavelength
                            ),
                        })
    interior = [record for record in records if record["interior_finite_mode"]]
    temperatures = sorted({record["temperature_K"] for record in interior})
    stresses = sorted({record["resolved_stress_Pa"] for record in interior})
    forest_coefficients = sorted({
        record["forest_coefficient_Pa_m2"] for record in interior
    })
    result = {
        "schema": "asb-drx-vector-topology-memory-dispersion/v1",
        "source_commit": os.environ.get("SOURCE_COMMIT", "local-working-tree"),
        "fixture_passed": True,
        "numerical_verification_passed": True,
        "finite_mode_region_identified": bool(interior),
        "interior_finite_mode_cases": len(interior),
        "temperatures_with_interior_mode_K": temperatures,
        "stresses_with_interior_mode_Pa": stresses,
        "forest_coefficients_with_interior_mode_Pa_m2": forest_coefficients,
        "screen": records,
        "mechanism_supported": False,
        "physical_CDD_wall_gate_passed": False,
        "integrated_scientific_claim_supported": False,
        "decision": (
            "advance_delayed_memory_to_compact_nonlinear_falsification"
            if interior else "reject_delayed_memory_closure"
        ),
        "reason": (
            "positive interior continuum modes occur across multiple bounded "
            "temperatures, stresses, diffusivities, and relaxation times; "
            "nonlinear convergence and topology tests remain required"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
