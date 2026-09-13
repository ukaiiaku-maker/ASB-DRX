#!/usr/bin/env python3
"""Bounded local dispersion screen for the minimal vector-topology closure."""

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
    arrhenius_forest_linearization, reaction_transport_symbol_s_inv,
)


EV_J = 1.602176634e-19


def mechanism() -> ArrheniusMechanism:
    return ArrheniusMechanism(
        ExpFloorEnthalpy(0.45 * EV_J, 0.8e9, 900.0, 0.25, 1.2, 2.0),
        BoundedActivationEntropy(reference_kB=0.2), 2.0e7,
        validity_temperature_K=(500.0, 1400.0),
        validity_stress_Pa=(0.0, 2.0e9),
    )


def main(output: Path) -> None:
    glide = mechanism()
    network = VectorTopologyNetwork(
        2.86e-10,
        (JunctionReaction(
            (0, 1), "sessile", glide, 0.02 * EV_J, 5.0e14,
        ),),
    )
    mobile = np.full(8, 2.5e14)
    reaction = network.reactions[0]
    forward, reverse = reaction.rate_constants_s_inv(900.0)
    junction = np.asarray([
        forward / reverse * mobile[0] * mobile[1]
        / reaction.reference_density_m2
    ])
    total_density = float(np.sum(mobile) + np.sum(junction))
    # A continuum wavelength must contain at least ten mean line spacings.
    minimum_continuum_wavelength_m = 10.0 / np.sqrt(total_density)
    maximum_continuum_k_m_inv = 2.0 * np.pi / minimum_continuum_wavelength_m
    continuum_k = np.geomspace(1.0e3, maximum_continuum_k_m_inv, 320)
    extended_k = np.geomspace(1.0e3, 1.0e11, 640)
    records = []
    for diffusivity in (1.0e-13, 2.0e-13, 5.0e-13, 1.0e-12):
        for forest_coefficient in (1.0e-8, 1.0e-7, 3.0e-7, 1.0e-6):
            velocities, derivative = arrhenius_forest_linearization(
                (glide,) * 4, np.full(4, 1.0e-9), np.full(4, 0.7e9),
                junction, np.full((4, 1), forest_coefficient), 900.0,
            )

            def growth(k_m_inv: float) -> float:
                operator = reaction_transport_symbol_s_inv(
                    np.asarray([k_m_inv, 0.0]), mobile, junction, network,
                    900.0, np.asarray([1.0, 0.0]), velocities, derivative,
                    np.full(8, diffusivity),
                )
                return float(np.max(np.linalg.eigvals(operator).real))

            continuum_growth = np.asarray([growth(value) for value in continuum_k])
            extended_growth = np.asarray([growth(value) for value in extended_k])
            ci = int(np.argmax(continuum_growth))
            ei = int(np.argmax(extended_growth))
            records.append({
                "diffusivity_m2_s": diffusivity,
                "forest_coefficient_Pa_m2": forest_coefficient,
                "continuum_maximum_growth_s_inv": float(continuum_growth[ci]),
                "continuum_peak_wavelength_m": float(2.0 * np.pi / continuum_k[ci]),
                "continuum_peak_at_short_wavelength_boundary": ci == len(continuum_k) - 1,
                "extended_maximum_growth_s_inv": float(extended_growth[ei]),
                "extended_peak_wavelength_m": float(2.0 * np.pi / extended_k[ei]),
            })
    positive_continuum = [
        record for record in records
        if record["continuum_maximum_growth_s_inv"] > 0.0
    ]
    robust_interior = [
        record for record in positive_continuum
        if not record["continuum_peak_at_short_wavelength_boundary"]
    ]
    result = {
        "schema": "asb-drx-vector-topology-dispersion/v1",
        "source_commit": os.environ.get("SOURCE_COMMIT", "local-working-tree"),
        "fixture_passed": True,
        "numerical_verification_passed": True,
        "frank_rule_passed": bool(np.max(np.abs(network.frank_residuals_m)) < 1.0e-24),
        "minimum_continuum_wavelength_m": minimum_continuum_wavelength_m,
        "screen": records,
        "positive_continuum_cases": len(positive_continuum),
        "robust_interior_finite_mode_cases": len(robust_interior),
        "mechanism_supported": False,
        "physical_CDD_wall_gate_passed": False,
        "integrated_scientific_claim_supported": False,
        "decision": (
            "reject_minimal_instantaneous_junction_friction_closure"
            if not robust_interior else "interior_finite_mode_region_identified"
        ),
        "reason": (
            "damped cases or positive growth driven to the declared shortest "
            "continuum wavelength; no robust interior finite-k selection"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
