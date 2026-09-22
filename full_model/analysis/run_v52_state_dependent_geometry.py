#!/usr/bin/env python3
"""Qualification evidence for state-sampled two-face geometry kinetics."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from full_model.analysis.run_v50_subcell_qualification import fixture
from full_model.production.state_dependent_subcell import (
    accepted_state_dependent_subcell_x_faces, state_dependent_face_rates,
)
from full_model.production.v24_mechanical_wall import (
    mechanical_checkpoint_arrays, mechanical_from_checkpoint_arrays,
)
from tests.test_v50_production_subcell_geometry import intrinsic_kinetics


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(a, b, scale):
    return float(abs(float(a)-float(b))/max(abs(float(scale)), 1e-300))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state, data = fixture(32)
    directions = {"lower_x": 1.0, "upper_x": -1.0}
    call = (data[4], data[5], data[1], data[6], data[7],
            intrinsic_kinetics())
    quadrature = {}
    for order in (8, 16, 32):
        quadrature[str(order)] = state_dependent_face_rates(
            state, directions, *call, quadrature_order=order)
    homogeneous_relative = {
        face: relative(quadrature["8"][face]["generalized_rate_s"],
                       quadrature["32"][face]["generalized_rate_s"],
                       quadrature["32"][face]["generalized_rate_s"])
        for face in directions
    }
    _, y = np.indices(state.common.temperature_K.shape)
    heterogeneous = replace(state, common=replace(
        state.common,
        temperature_K=900.0+400.0*(y/(y.shape[1]-1))**2))
    heterogeneous_rates = state_dependent_face_rates(
        heterogeneous, directions, *call, quadrature_order=32)

    duration = 1e-6
    whole, whole_ledger = accepted_state_dependent_subcell_x_faces(
        state, directions, *call, duration, quadrature_order=8)
    split = state; split_ledgers = []
    for _ in range(2):
        split, ledger = accepted_state_dependent_subcell_x_faces(
            split, directions, *call, duration/2, quadrature_order=8)
        split_ledgers.append(ledger)
    lower_scale = abs(float(whole.subcell_geometry.lower_left_m[0]
                            -state.subcell_geometry.lower_left_m[0]))
    upper_scale = abs(float(whole.subcell_geometry.upper_right_m[0]
                            -state.subcell_geometry.upper_right_m[0]))
    subdivision = {
        "whole_accepted": bool(whole_ledger["accepted"]),
        "split_accepted": bool(all(row["accepted"] for row in split_ledgers)),
        "whole_rate_refresh_count": whole_ledger["rate_refresh_count"],
        "split_rate_refresh_count": sum(
            row["rate_refresh_count"] for row in split_ledgers),
        "lower_face_displacement_relative_difference": relative(
            split.subcell_geometry.lower_left_m[0],
            whole.subcell_geometry.lower_left_m[0], lower_scale),
        "upper_face_displacement_relative_difference": relative(
            split.subcell_geometry.upper_right_m[0],
            whole.subcell_geometry.upper_right_m[0], upper_scale),
        "mean_temperature_difference_K": float(
            np.mean(split.common.temperature_K)
            -np.mean(whole.common.temperature_K)),
        "whole_elapsed_time_s": whole_ledger["consumed_duration_s"],
        "split_elapsed_time_s": sum(
            row["consumed_duration_s"] for row in split_ledgers),
    }

    first, first_ledger = accepted_state_dependent_subcell_x_faces(
        state, directions, *call, 2e-7, quadrature_order=8)
    restored = mechanical_from_checkpoint_arrays(
        mechanical_checkpoint_arrays(first), data[4], data[5])
    continuous, second_a = accepted_state_dependent_subcell_x_faces(
        first, directions, *call, 2e-7, quadrature_order=8)
    restarted, second_b = accepted_state_dependent_subcell_x_faces(
        restored, directions, *call, 2e-7, quadrature_order=8)
    restart_exact = all(np.array_equal(
        value, mechanical_checkpoint_arrays(restarted)[name])
        for name, value in mechanical_checkpoint_arrays(continuous).items())

    payload = {
        "schema": "asb-drx/v52/state-dependent-shared-geometry/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "event_measure": "one_exchanged_species_per_climb_event_per_face",
        "local_state_sampling": (
            "periodic bilinear field sampling at Gauss sites on each actual "
            "straight face"),
        "rigid_face_velocity_closure": (
            "physical-site-weighted mean of local nonlinear Arrhenius-affinity "
            "rates times the microscopic event jump"),
        "quadrature": quadrature,
        "homogeneous_q8_q32_relative_rate_difference": homogeneous_relative,
        "heterogeneous_temperature_control": heterogeneous_rates,
        "rate_at_mean_inputs_is_not_mean_local_rate": {
            face: bool(not np.isclose(
                row["generalized_rate_s"],
                row["rate_at_weighted_mean_inputs_s"], rtol=1e-8))
            for face, row in heterogeneous_rates.items()
        },
        "common_clock_subdivision": subdivision,
        "restart": {
            "first_accepted": bool(first_ledger["accepted"]),
            "continuous_second_accepted": bool(second_a["accepted"]),
            "restarted_second_accepted": bool(second_b["accepted"]),
            "exact": bool(restart_exact),
        },
        "numerical_cap_handling": (
            "common subinterval terminates at the first quarter-cell rate "
            "refresh cap; both face rates are recomputed from the new state"),
        "scope_limit": (
            "partial one-face stall remains fail-closed pending a dedicated "
            "one-face shared owner; no serial clock fallback"),
        "chemical_work_J_per_defect": 0.0,
        "prepared_geometry_not_spontaneous_lagb": True,
        "drx_claimed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps({
        "sha256": digest(args.output),
        "restart_exact": restart_exact,
        "subdivision": subdivision,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
