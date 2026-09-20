#!/usr/bin/env python3
"""State-dependent finite-rate accessibility and production routing audit."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from full_model.analysis.run_v24_mechanical_supply import build_case
from full_model.production.extensive_wall import accepted_ordering_step
from full_model.production.v24_mechanical_wall import accepted_v24_mechanical_step
from tests.test_v40_ordering_finite_time import _compact_active_case


def counterexample():
    state, density, systems, topologies, parameters, target, stress, attempt = (
        _compact_active_case())
    totals = {sign: (getattr(density, f"wall_tangle_{sign}_m2")
                     +getattr(density, f"wall_ordered_{sign}_m2"))
              for sign in ("plus", "minus")}
    initial = replace(
        density,
        wall_tangle_plus_m2=.75*totals["plus"],
        wall_ordered_plus_m2=.25*totals["plus"],
        wall_tangle_minus_m2=.75*totals["minus"],
        wall_ordered_minus_m2=.25*totals["minus"])
    guarded = replace(
        parameters, nye_match_coefficient_J_m=0.0,
        ordered_gradient_J_m3=0.0,
        ordered_excess_J_m=parameters.disordered_excess_J_m+1e-10,
        ordering_integration_method="implicit_backward_euler",
        ordering_finite_time_backend="matrix_free_exponential_rosenbrock",
        ordering_asymptotic_minimum_attempt_exposure=5e-4)
    duration = .001/attempt
    updated, ledger, _ = accepted_ordering_step(
        initial, systems, topologies, state.common.orientation_rad, target,
        stress, state.common.temperature_K, guarded, duration)
    signs = {}
    for sign in ("plus", "minus"):
        active = totals[sign] > 0.0
        q = np.divide(getattr(updated, f"wall_ordered_{sign}_m2"),
                      totals[sign], out=np.zeros_like(totals[sign]),
                      where=active)
        density_change = (getattr(updated, f"wall_ordered_{sign}_m2")
                          -getattr(initial, f"wall_ordered_{sign}_m2"))
        signs[sign] = {
            "minimum_final_active_q": float(np.min(q[active])),
            "maximum_abs_delta_q": float(np.max(np.abs(q[active]-.25))),
            "maximum_abs_ordered_density_change_m2": float(
                np.max(np.abs(density_change))),
            "integrated_abs_line_change_per_thickness": float(
                np.sum(np.abs(density_change))*parameters.spacing_m**2),
            "maximum_abs_burgers_moment_change_m1": float(
                np.max(np.abs(density_change))*2.48e-10),
            "inactive_pool_remains_exact_zero": bool(np.array_equal(
                getattr(updated, f"wall_ordered_{sign}_m2")[~active], 0.0)),
        }
    return {
        "initial_q": .25, "local_attempt_exposure": .001,
        "unreachable_equilibrium_q": 0.0,
        "necessary_lower_bound_on_final_q": .249,
        "duration_s": duration, "signs": signs,
        "dispatch": ledger["stiff_dispatch"],
        "accessibility_passed": ledger["asymptotic_state_accessibility_passed"],
        "maximum_violation": ledger[
            "asymptotic_accessibility_maximum_violation"],
        "finite_time_kinetic_accuracy_certified": ledger[
            "finite_time_kinetic_accuracy_certified_by_this_solve"],
    }


def actual_production_state():
    (state, driving, _, support, systems, topologies, parameters, extensive,
     kinetics, _) = build_case(
         32, length_m=3.2e-6, periodic_nye_consistent=True)
    requested_dt = 4.8828125e-7
    updated, ledger = accepted_v24_mechanical_step(
        state, driving, support, systems, topologies, parameters, extensive,
        kinetics, requested_dt, topology_route_enabled=False,
        mura_transport_operator="compatible_dealiased")
    ordering = ledger["ordering_thermodynamics"]
    return {
        "grid": 32, "accepted_dt_s": ledger["accepted_dt_s"],
        "requested_dt_s": requested_dt,
        "dispatch": ordering.get("stiff_dispatch", "exact_zero_pool"),
        "integration_method": ordering["integration_method"],
        "maximum_attempt_exposure": ordering.get("maximum_attempt_exposure", 0.0),
        "minimum_active_local_attempt_exposure": ordering.get(
            "minimum_active_local_attempt_exposure"),
        "proposed_maximum_fraction_change": ordering.get(
            "asymptotic_proposed_maximum_fraction_change"),
        "accessibility_maximum_violation": ordering.get(
            "asymptotic_accessibility_maximum_violation"),
        "complete_elapsed_time_s": ordering["complete_elapsed_time_s"],
        "discarded_reaction_time_s": ordering["discarded_reaction_time_s"],
        "active_degrees_of_freedom": ordering.get(
            "active_degrees_of_freedom", 0),
        "state_changed": bool(not np.array_equal(
            state.common.beta_p, updated.common.beta_p)),
    }


def main():
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip()
    scoped_diff = subprocess.run([
        "git", "diff", "--quiet",
        "2d3320a71642671397376208c6cfdef17d77b015",
        "a9c4f0b47eca3d261a9c731bdcd0d18410485ab2", "--",
        "full_model/production", "tests"], check=False)
    payload = {
        "schema": "asb-drx/v46/ordering-applicability/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": source,
        "governing_bound": "|q_i(t+H)-q_i(t)| <= integral a_i dt",
        "bound_scope": (
            "necessary local sign/family finite-speed accessibility; not a "
            "sufficient finite-time endpoint error certificate"),
        "counterexample": counterexample(),
        "actual_first_operation": actual_production_state(),
        "v45_n128_sub4_sub8_scoped_source_equivalent": bool(
            scoped_diff.returncode == 0),
        "v45_source_pair": [
            "a9c4f0b47eca3d261a9c731bdcd0d18410485ab2",
            "2d3320a71642671397376208c6cfdef17d77b015"],
        "zero_pool_semantics": (
            "no fraction coordinate is formed; density and signed moment remain exact zero"),
        "material_calibration_claimed": False,
    }
    payload["classification"] = (
        "STATE_INACCESSIBLE_ASYMPTOTIC_ENDPOINT_REJECTED_FINITE_TIME_USED"
        if (not payload["counterexample"]["accessibility_passed"]
            and payload["counterexample"][
                "finite_time_kinetic_accuracy_certified"])
        else "FAILED_STATE_QUALIFICATION")
    output = Path("full_model/verification/v46_ordering_applicability.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps({
        "classification": payload["classification"],
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest()},
        sort_keys=True))


if __name__ == "__main__":
    main()
