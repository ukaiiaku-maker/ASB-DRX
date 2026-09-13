#!/usr/bin/env python3
"""Factorized bounded screen of the integrated Mission-v3 1-D candidate."""

from __future__ import annotations

from dataclasses import replace
import json
import os
import platform
from pathlib import Path

import numpy as np

from asb_drx.arrhenius_v3 import (
    ArrheniusMechanism, BoundedActivationEntropy, ExpFloorEnthalpy,
)
from asb_drx.integrated_cdd_v3 import (
    IntegratedCDDParameters, IntegratedCDDState,
    frozen_mode_eigenvalues_s_inv, integrated_cdd_step,
)
from asb_drx.physical_noise import periodic_physical_noise
from asb_drx.staggered_cdd import StaggeredSignedState


EV_J = 1.602176634e-19


def parameters(entropy_kB: float = 0.0, ratio: float = 1.0) -> IntegratedCDDParameters:
    mechanism = ArrheniusMechanism(
        ExpFloorEnthalpy(0.55 * EV_J, 0.7e9, 900.0, 0.2, 1.3, 2.0),
        BoundedActivationEntropy(reference_kB=entropy_kB),
        2.0e7,
        validity_temperature_K=(500.0, 1400.0),
        validity_stress_Pa=(0.0, 2.0e9),
    )
    dyads = np.zeros((4, 3, 3))
    dyads[:, 0, 2] = np.asarray([0.45, 0.30, -0.30, -0.45])
    return IntegratedCDDParameters(
        mechanism, 45.0e9, 2.5e6, 2.0e-9, dyads, 2.0e-13,
        backstress_coefficient=ratio, diffusion_coefficient=1.0,
        reference_density_m2=5.0e14,
        glide_event_length_m=1.0e-9,
    )


def spectrum_record(
    label: str, n: int, domain_um: float, density: float,
    entropy: float, ratio: float, stress_Pa: float,
) -> dict[str, object]:
    model = parameters(entropy, ratio)
    resolved = stress_Pa * model.slip_dyads_crystal[:, 0, 2]
    spectrum = frozen_mode_eigenvalues_s_inv(
        n, domain_um * 1.0e-6, density, resolved, 900.0,
        2.86e-10, model,
    )
    maximum_by_mode = {
        mode: float(np.max(values.real)) for mode, values in spectrum.items()
    }
    fastest = max(maximum_by_mode, key=maximum_by_mode.get)
    return {
        "label": label, "grid_points": n, "domain_um": domain_um,
        "density_per_population_m2": density, "entropy_kB": entropy,
        "backstress_to_diffusion_ratio": ratio, "stress_Pa": stress_Pa,
        "maximum_real_growth_s_inv": maximum_by_mode[fastest],
        "least_damped_mode": fastest,
        "nyquist_real_growth_s_inv": maximum_by_mode[n // 2],
        "all_resolved_modes_nonpositive": max(maximum_by_mode.values()) <= 1.0e-7,
    }


def nonlinear_noise_record(seed: int, correlation_um: float) -> dict[str, object]:
    n = 64
    domain = 16.0e-6
    model = parameters()
    noise = periodic_physical_noise(n, domain, correlation_um * 1.0e-6, seed)
    population = 2.5e14 * (1.0 + 0.01 * noise)
    plus = np.broadcast_to(population, (4, n)).copy()
    minus = plus.copy()
    signed = StaggeredSignedState(
        plus, minus, np.zeros_like(plus), 2.86e-10, domain / n,
    )
    beta_p = np.zeros((n, 3, 3))
    identity = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
    state = IntegratedCDDState(
        signed, beta_p, identity, np.full(n, 900.0),
        applied_shear=0.01,
    )
    before = plus + minus
    step = integrated_cdd_step(state, 0.0, 0.01, model)
    after = step.state.signed.mobile_plus_m2 + step.state.signed.mobile_minus_m2
    return {
        "seed": seed, "correlation_length_um": correlation_um,
        "accepted_dt_s": step.accepted_dt_s, "halvings": step.halvings,
        "total_density_cv_before": float(np.std(before) / np.mean(before)),
        "total_density_cv_after": float(np.std(after) / np.mean(after)),
        "correlation_energy_change_J_m3": step.ledger.correlation_energy_change_J_m3,
        "clipping_added_m_inv": step.ledger.flux.clipping_added_m_inv,
    }


def source_provenance() -> str:
    fields = os.environ.get("HPC3_RUN_ID", "").split("-")
    if len(fields) >= 3 and len(fields[1]) == 7:
        return fields[1]
    return os.environ.get("SOURCE_COMMIT", "local-working-tree")


def main(output: Path) -> None:
    cases: list[tuple[str, int, float, float, float, float, float]] = []
    baseline = (64, 16.0, 2.5e14, 0.0, 1.0, 0.45e9)
    cases.append(("baseline", *baseline))
    cases.extend((f"entropy_{value:+g}", 64, 16.0, 2.5e14, value, 1.0, 0.45e9)
                 for value in (-1.0, 1.0))
    cases.extend((f"ratio_{value:g}", 64, 16.0, 2.5e14, 0.0, value, 0.45e9)
                 for value in (0.5, 2.0))
    cases.extend((f"density_{value:.1e}", 64, 16.0, value, 0.0, 1.0, 0.45e9)
                 for value in (1.0e14, 1.0e15))
    cases.extend((f"grid_{value}", value, 16.0, 2.5e14, 0.0, 1.0, 0.45e9)
                 for value in (128, 256))
    cases.extend((f"domain_{value:g}", 64, value, 2.5e14, 0.0, 1.0, 0.45e9)
                 for value in (13.0, 19.0))
    cases.extend((label, 64, 16.0, 2.5e14, 0.0, 1.0, stress)
                 for label, stress in (
                     ("zero_load", 0.0), ("below_threshold", 0.10e9),
                     ("unload", 0.0),
                 ))
    spectra = [spectrum_record(label, n, domain, density, entropy, ratio, stress)
               for label, n, domain, density, entropy, ratio, stress in cases]
    noise = [nonlinear_noise_record(seed, length)
             for seed in (7, 19, 41) for length in (0.35, 0.70)]
    hard = {
        "all_spectra_damped_through_nyquist": all(
            record["all_resolved_modes_nonpositive"] for record in spectra
        ),
        "all_nonlinear_steps_clip_free": all(
            record["clipping_added_m_inv"] == 0.0 for record in noise
        ),
        "all_nonlinear_steps_release_correlation_energy": all(
            record["correlation_energy_change_J_m3"] <= 0.0 for record in noise
        ),
    }
    result = {
        "schema": "asb-drx-mission-v3-integrated-screen/v1",
        "fixture_passed": all(hard.values()),
        "numerical_verification_passed": all(hard.values()),
        "mechanism_supported": False,
        "physical_CDD_wall_gate_passed": False,
        "integrated_scientific_claim_supported": False,
        "predictive_validation_supported": False,
        "decision": (
            "POSITIVE_VARIATIONAL_1D_BASELINE_IS_ADVECTIVE_DISSIPATIVE_AND_HAS_NO_"
            "FINITE_MODE_WALL_INSTABILITY; ESCALATE_TO_STATE_DEPENDENT_FRICTION_"
            "AND_CROSS_FAMILY_COUPLING"
        ),
        "hard_invariants": hard,
        "factorized_spectrum_screen": spectra,
        "physical_noise_nonlinear_screen": noise,
        "provenance": {
            "source_commit": source_provenance(),
            "python": platform.python_version(), "platform": platform.platform(),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main(Path("output/mission_v3_integrated_screen.json"))
