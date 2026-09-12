#!/usr/bin/env python3
"""Compact Mission-v3 Arrhenius and staggered-CDD model comparison."""

from __future__ import annotations

import json
import math
import os
import platform
from pathlib import Path

import numpy as np

from asb_drx.arrhenius_v3 import (
    ArrheniusMechanism,
    BoundedActivationEntropy,
    ExpFloorEnthalpy,
    PARAMETER_CLASSIFICATION,
)
from asb_drx.cdd_flux_v3 import (
    FLUX_PARAMETER_CLASSIFICATION,
    correlation_rhs,
    discrete_mode_growth_rates_s_inv,
)
from asb_drx.physical_noise import periodic_physical_noise


EV_J = 1.602176634e-19


def source_provenance() -> str:
    fields = os.environ.get("HPC3_RUN_ID", "").split("-")
    if len(fields) >= 3 and len(fields[1]) == 7:
        return fields[1]
    return os.environ.get("SOURCE_COMMIT", "local-working-tree")


def arrhenius_family_records() -> list[dict[str, object]]:
    enthalpy = ExpFloorEnthalpy(
        1.5 * EV_J, 1.2e9, 1000.0, 0.2, 2.0, 2.5,
        enthalpy_temperature_coefficient=0.3,
        stress_temperature_coefficient=0.1,
    )
    families = (
        ("A_zero_entropy", BoundedActivationEntropy()),
        ("B_positive_entropy", BoundedActivationEntropy(reference_kB=2.0)),
        ("C_negative_entropy", BoundedActivationEntropy(reference_kB=-2.0)),
        ("D_temperature_dependent_H0", BoundedActivationEntropy(reference_kB=0.5)),
        (
            "E_bounded_temperature_entropy",
            BoundedActivationEntropy(
                reference_kB=0.5, temperature_amplitude_kB=1.0,
                temperature_width_K=150.0, reference_temperature_K=1000.0,
            ),
        ),
    )
    records = []
    for name, entropy in families:
        mechanism = ArrheniusMechanism(
            enthalpy, entropy, 1.0e12,
            validity_temperature_K=(700.0, 1300.0),
            validity_stress_Pa=(0.0, 3.0e9),
        )
        samples = []
        valid = True
        for temperature in (800.0, 1000.0, 1200.0):
            for stress in (0.0, 0.4e9, 0.8e9, 1.2e9):
                try:
                    free = mechanism.activation_free_energy_J(stress, temperature)
                    rate = mechanism.net_rate_s_inv(stress, temperature)
                    volume = enthalpy.activation_volume_m3(stress, temperature)
                except ValueError:
                    valid = False
                    continue
                valid = valid and free >= 0.0 and volume >= 0.0 and stress * rate >= 0.0
                samples.append({
                    "temperature_K": temperature,
                    "stress_Pa": stress,
                    "activation_enthalpy_J": enthalpy.enthalpy_J(stress, temperature),
                    "activation_entropy_kB": entropy.entropy_kB(temperature),
                    "activation_free_energy_J": free,
                    "activation_volume_m3": volume,
                    "net_rate_s_inv": rate,
                })
        records.append({"family": name, "valid": bool(valid), "samples": samples})
    return records


def flux_candidate_record(face_density: str) -> dict[str, object]:
    spectra = {}
    for n in (64, 128, 256):
        spectra[str(n)] = discrete_mode_growth_rates_s_inv(
            n, 16.0e-6, 5.0e14, 2.0e-13, 40.0e9, 2.86e-10,
            backstress_coefficient=1.0, diffusion_coefficient=1.0,
            face_density=face_density,
        )
    rng = np.random.default_rng(17)
    plus = 5.0e14 * (0.5 + 0.01 * rng.normal(size=(2, 128)))
    minus = 5.0e14 * (0.5 + 0.01 * rng.normal(size=(2, 128)))
    rhs_plus, rhs_minus = correlation_rhs(
        plus, minus, np.full_like(plus, 2.0e-13), np.full(128, 40.0e9),
        2.86e-10, 16.0e-6 / 128,
        backstress_coefficient=1.0, diffusion_coefficient=1.0,
        face_density=face_density,
    )
    maximum_balance = max(
        float(np.max(np.abs(np.sum(rhs_plus, axis=1)))),
        float(np.max(np.abs(np.sum(rhs_minus, axis=1)))),
    )
    reference_rate = max(
        float(np.max(np.sum(np.abs(rhs_plus), axis=1))),
        float(np.max(np.sum(np.abs(rhs_minus), axis=1))),
    )
    nyquist_damped = all(
        item["total_s_inv"][-1] < 0.0 and item["signed_s_inv"][-1] < 0.0
        for item in spectra.values()
    )
    return {
        "candidate": face_density,
        "periodic_balance_residual_m2_s_inv": maximum_balance,
        "periodic_balance_relative_residual": maximum_balance / reference_rate,
        "nyquist_damped": nyquist_damped,
        "spectra": spectra,
    }


def physical_noise_record() -> dict[str, object]:
    domain = 16.0e-6
    coarse = periodic_physical_noise(64, domain, 0.5e-6, 9)
    fine = periodic_physical_noise(128, domain, 0.5e-6, 9)
    restrictions = float(np.max(np.abs(coarse - fine[::2])))
    boxes = {}
    for length_um, n in ((13, 104), (16, 128), (19, 152)):
        field = periodic_physical_noise(n, length_um * 1.0e-6, 0.5e-6, 9)
        modes = int(np.count_nonzero(np.abs(np.fft.rfft(field)) > 1.0e-9))
        boxes[str(length_um)] = {"grid_points": n, "resolved_modes": modes}
    return {
        "correlation_length_m": 0.5e-6,
        "cutoff_wavenumber_m_inv": 8.0e6,
        "same_box_restriction_max_abs": restrictions,
        "box_spectra": boxes,
    }


def main(output: Path) -> None:
    arrhenius = arrhenius_family_records()
    fluxes = [flux_candidate_record(name) for name in ("arithmetic", "logarithmic")]
    noise = physical_noise_record()
    hard = {
        "arrhenius_barrier_rate_and_dissipation_valid": all(x["valid"] for x in arrhenius),
        "both_fluxes_close_periodic_balance": all(x["periodic_balance_relative_residual"] < 2.0e-15 for x in fluxes),
        "both_fluxes_remove_nyquist_null": all(x["nyquist_damped"] for x in fluxes),
        "physical_noise_is_grid_restriction": noise["same_box_restriction_max_abs"] < 3.0e-14,
    }
    result = {
        "schema": "asb-drx-mission-v3-model-comparison/v1",
        "fixture_passed": all(hard.values()),
        "numerical_verification_passed": all(hard.values()),
        "mechanism_supported": False,
        "integrated_scientific_claim_supported": False,
        "predictive_validation_supported": False,
        "decision": (
            "LOGARITHMIC_FACE_FORM_PREFERRED_FOR_INTEGRATION_DUE_TO_DISCRETE_CHAIN_RULE; "
            "FULL_SIGNED_KINEMATIC_COUPLING_REMAINS_UNQUALIFIED"
        ),
        "hard_invariants": hard,
        "arrhenius_families": arrhenius,
        "flux_candidates": fluxes,
        "physical_noise": noise,
        "parameter_classification": {
            "arrhenius": PARAMETER_CLASSIFICATION,
            "flux": FLUX_PARAMETER_CLASSIFICATION,
        },
        "provenance": {
            "source_commit": source_provenance(),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main(Path("output/mission_v3_model_comparison.json"))
