#!/usr/bin/env python3
"""Machine-readable V45 finite/transient/asymptotic ordering qualification."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from full_model.production.extensive_wall import accepted_ordering_step
from tests.test_v40_ordering_finite_time import _compact_active_case


NAMES = ("wall_tangle_plus_m2", "wall_tangle_minus_m2",
         "wall_ordered_plus_m2", "wall_ordered_minus_m2")


def step(case, parameters, exposure):
    state, density, systems, topologies, _, target, stress, attempt = case
    return accepted_ordering_step(
        density, systems, topologies, state.common.orientation_rad, target,
        stress, state.common.temperature_K, parameters, exposure/attempt)


def errors(left, right):
    rows = {}
    for name in NAMES:
        a = np.asarray(getattr(left, name)); b = np.asarray(getattr(right, name))
        rows[name] = {
            "maximum_absolute_m2": float(np.max(np.abs(a-b))),
            "maximum_relative_to_pair_peak": float(
                np.max(np.abs(a-b))/max(float(np.max(np.abs(a))),
                                        float(np.max(np.abs(b))), 1.0)),
        }
    return rows


def main():
    case = _compact_active_case(); parameters = case[4]
    dense = replace(
        parameters, ordering_integration_method="finite_time_bdf",
        ordering_finite_time_backend="dense_bdf_oracle")
    hybrid = replace(
        parameters, ordering_integration_method="implicit_backward_euler",
        ordering_finite_time_backend="matrix_free_exponential_rosenbrock",
        ordering_asymptotic_minimum_attempt_exposure=1.000001e-3)
    dense_transient = step(case, dense, 1e-6)
    matrix_transient = step(case, replace(
        hybrid, ordering_integration_method="finite_time_bdf"), 1e-6)
    dense_overlap = step(case, dense, 1e-3)
    asymptotic_overlap = step(case, hybrid, 1.000001e-3)
    retained = json.loads(Path(
        "full_model/verification/v45_resumed_n128_endpoint.json").read_text())
    payload = {
        "schema": "asb-drx/v45/ordering-qualification/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "governing_equation": (
            "qdot_s = -a(x)*tanh(event_length*(mu_ordered_s-"
            "mu_tangle_s)/(2*kB*T)); rho_ordered_s=q_s*rho_total_s; "
            "0<=q_s<=1 for each Burgers family and sign"),
        "spatial_operator": (
            "chemical-potential Hessian acts globally by exact periodic FFT; "
            "no physical-space Jacobian sparsity is asserted"),
        "finite_backend": {
            "name": "matrix_free_exponential_rosenbrock",
            "transient_attempt_exposure": 1e-6,
            "dense_oracle_errors": errors(
                matrix_transient[0], dense_transient[0]),
            "integration_method": matrix_transient[1]["integration_method"],
            "complete_elapsed_time_s": matrix_transient[1][
                "complete_elapsed_time_s"],
            "discarded_reaction_time_s": matrix_transient[1][
                "discarded_reaction_time_s"],
        },
        "finite_asymptotic_overlap": {
            "dense_attempt_exposure": 1e-3,
            "asymptotic_attempt_exposure": 1.000001e-3,
            "errors": errors(asymptotic_overlap[0], dense_overlap[0]),
            "asymptotic_endpoint_remainder_relative": asymptotic_overlap[1][
                "implicit_max_scaled_residual"],
            "finite_time_error_semantics": (
                "overlap error is measured against the dense finite endpoint; "
                "stationary remainder is reported separately and is not "
                "relabeled as finite-time error"),
        },
        "retained_n128": {
            "source_sha": retained["source_sha"],
            "active_degrees_of_freedom": retained[
                "ordering_active_degrees_of_freedom"],
            "dense_jacobian_bytes_avoided": retained[
                "ordering_dense_jacobian_bytes_avoided"],
            "wall_seconds": retained["wall_seconds"],
            "dispatches": retained["ordering_dispatches"],
            "front_repeated": retained["front_repeated"],
            "front_heat_redeposited": retained["front_heat_redeposited"],
        },
        "quarantined_candidates": [
            {"source": "25fe320", "backend": "projected Newton/Picard",
             "classification": "FAILED_NUMERICAL_ACTIVE_SET_NONDESCENT",
             "residual": 2.0319145066043491e-4,
             "best_trial_residual": 2.8614302163838297e-4},
            {"source": "05008f3", "backend": "trust-region backward Euler",
             "classification": "FAILED_NUMERICAL_NONROOT_XTOL",
             "residual": 1.0910270371381611e-4},
            {"backend": "adaptive explicit DOP853",
             "classification": "FAILED_COST_STIFFNESS_LIMITED_COMPACT_FIXTURE"},
        ],
        "production_dispatch": {
            "finite_below_attempt_exposure": 1.000001e-3,
            "asymptotic_at_or_above_attempt_exposure": 1.000001e-3,
            "no_dense_production_jacobian": True,
        },
        "material_calibration_claimed": False,
    }
    output = Path("full_model/verification/v45_ordering_qualification.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps({
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "maximum_overlap_relative_error": max(
            row["maximum_relative_to_pair_peak"]
            for row in payload["finite_asymptotic_overlap"]["errors"].values()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
