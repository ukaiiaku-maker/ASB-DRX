#!/usr/bin/env python3
"""V48 matched-path derivative, subcell symmetry, and affinity-rate evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from full_model.analysis.run_v46_geometry_representation import (
    REPRESENTATION_LENGTH_M, energy_terms_J, prepare_represented_block,
)
from full_model.analysis.run_v47_geometry_force import propose, stable_event
from full_model.production.arrhenius_kinetics import ActivatedProcess, EV_J
from full_model.production.v24_mechanical_wall import (
    V43GeometryKinetics, accepted_geometry_plaquette_transaction,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def energy_increment(state, data, cell, extent):
    return stable_event(state, propose(state, data, cell, extent), data,
                        extent)["stable_complete_change_J"]


def derivative_audit(state, data, cell):
    extents = np.asarray((.0125, .025, .05, .1, .2))
    increments = np.asarray([
        energy_increment(state, data, cell, float(s)) for s in extents])
    design = np.stack((extents, extents**2), axis=1)
    linear, quadratic = np.linalg.lstsq(design, increments, rcond=None)[0]
    fitted = design@np.asarray((linear, quadratic))
    fit_relative = float(np.max(np.abs(fitted-increments))/max(
        np.max(np.abs(increments)), 1e-300))
    rows = []
    for location in (.025, .05, .1):
        h = 1e-4
        numerical = (energy_increment(state, data, cell, location+h)
                     -energy_increment(state, data, cell, location-h))/(2*h)
        analytic = linear+2*quadratic*location
        rows.append({
            "extent": location,
            "quadratic_path_derivative_J_per_extent": analytic,
            "centered_same_state_derivative_J_per_extent": numerical,
            "relative_difference": abs(numerical-analytic)/max(
                abs(numerical), abs(analytic), 1e-300),
        })
    return {
        "path": "same plaquette, family, sign, line shape and thickness",
        "extents": extents.tolist(),
        "increments_J": increments.tolist(),
        "quadratic_linear_coefficient_J": float(linear),
        "quadratic_coefficient_J": float(quadratic),
        "quadratic_fit_max_relative_residual": fit_relative,
        "zero_extent_one_sided_force_N": float(-linear/data[9]),
        "finite_state_derivative_comparisons": rows,
        "secant_constancy_required": False,
    }


def subcell_translation(state, data, start, width):
    baseline = energy_terms_J(state, data)["total"]
    rows = []
    for fraction in (.125, .25, .5, .75, 1.0):
        candidate = state
        for j in range(start, start+width):
            candidate = propose(candidate, data, (start, j), -fraction)
            candidate = propose(candidate, data, (start+width, j), fraction)
        energy = energy_terms_J(candidate, data)["total"]
        rows.append({
            "fraction_of_cell": fraction,
            "displacement_m": fraction*data[9],
            "energy_change_J": energy-baseline,
        })
    scale = max(abs(baseline), 1e-300)
    return {
        "rows": rows,
        "maximum_abs_energy_change_over_stored_energy": max(
            abs(row["energy_change_J"])/scale for row in rows),
        "integer_shift_energy_change_J": rows[-1]["energy_change_J"],
        "scope": (
            "fractional occupancy translation path in a homogeneous periodic "
            "background; distinct from an exact integer np.roll permutation"),
    }


def affinity_rows(state, data, cell):
    rows = []
    for temperature in (800.0, 1100.0, 1400.0):
        local = state
        from dataclasses import replace
        local = replace(local, common=replace(
            local.common,
            temperature_K=np.full_like(local.common.temperature_K,
                                       temperature)))
        for mu in (1.1e-19, 1.43e-19, 1.7e-19, 3.0e-19):
            kinetics = V43GeometryKinetics(
                ActivatedProcess("v48-geometry-affinity", 1e9),
                enthalpy_J=.2*EV_J, critical_stress_Pa=1e9,
                chemical_species="vacancy",
                chemical_potential_J_per_defect=mu,
                atomic_volume_m3_per_atom=1.8e-29,
                exchange_stoichiometry_defects_per_atom=1.0,
                continuum_representation_length_m=REPRESENTATION_LENGTH_M)
            _, ledger = accepted_geometry_plaquette_transaction(
                local, {"cell": cell, "family": 0, "burgers_sign": 1,
                        "proposed_extent": .1},
                data[4], data[5], data[1], data[6], data[7], kinetics, 1e-9)
            rows.append({key: ledger.get(key) for key in (
                "classification", "accepted", "activation_enthalpy_J",
                "arrhenius_unbiased_rate_s", "affinity_biased_rate_s",
                "available_energy_per_event_J", "physical_event_count",
                "downhill_activity", "accepted_extent",
                "observed_accepted_velocity_m_s")})
            rows[-1].update(temperature_K=temperature,
                            chemical_potential_J_per_defect=mu)
    return rows


def main():
    output = Path("full_model/verification/v48_geometry_qualification.json")
    state, data, start, width = prepare_represented_block(64)
    cell = (start+width, start)
    retained = json.loads(Path(
        "full_model/verification/v47_geometry_force.json").read_text())
    refinement = retained["fixed_scale_refinement"]
    rate_rows = affinity_rows(state, data, cell)
    rate_verified = bool(
        any(row["accepted"] for row in rate_rows)
        and any(not row["accepted"] for row in rate_rows)
        and any(row["downhill_activity"] is not None
                and 0.0 < row["downhill_activity"] < .99
                for row in rate_rows)
        and all(row["activation_enthalpy_J"] is not None
                and row["activation_enthalpy_J"] > 0.0
                for row in rate_rows))
    record = {
        "schema": "asb-drx/v48/geometry-qualification/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "representation_length_m": REPRESENTATION_LENGTH_M,
        "matched_strip_refinement": refinement,
        "matched_strip_refinement_source": (
            "immutable V47 fixed-400-nm rows; not recomputed because repeated "
            "n128 plaquette construction is an avoidable local bottleneck"),
        "same_state_derivative": derivative_audit(state, data, cell),
        "subcell_translation": subcell_translation(state, data, start, width),
        "affinity_rate_rows": rate_rows,
        "matched_strip_sign_converged": False,
        "rate_implementation_verified": rate_verified,
        "geometry_observable_numerically_qualified": False,
        "classification": (
            "AFFINITY_RATE_VERIFIED_MATCHED_STRIP_SIGN_UNRESOLVED"
            if rate_verified else
            "AFFINITY_RATE_NOT_DISCRIMINATED_MATCHED_STRIP_SIGN_UNRESOLVED"),
        "drx_claimed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True)+"\n")
    print(json.dumps({
        "classification": record["classification"],
        "derivative_fit_relative": record["same_state_derivative"][
            "quadratic_fit_max_relative_residual"],
        "sha256": digest(output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
