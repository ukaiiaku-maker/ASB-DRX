#!/usr/bin/env python3
"""Matched-coordinate input attribution for the production EXP-floor map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from full_model.analysis.audit_v63_phase_sensitive_fields import (
    centered_band, common_quadrature, digest, norm, normalized_coefficients,
)
from full_model.analysis.postprocess_v59_physical_asb import _wall_parameters
from full_model.analysis.run_v58_three_grain_production import _load_checkpoint
from full_model.production.common_tensorial_wall import (
    CommonWallDriving, exp_floor_rate, resolved_driving_components,
)
from full_model.production.multigrain_production import (
    _owner_drivings_from_common_stress,
)
from full_model.production.tensorial_nye import bcc_four_family_systems


def project(field: np.ndarray, half_width: int, quadrature_n: int) -> np.ndarray:
    return common_quadrature(centered_band(
        normalized_coefficients(field), half_width), quadrature_n)


def owner_drivers(path: Path) -> tuple[dict, dict, object]:
    state, runtime, step, gamma, _, configuration, provenance = _load_checkpoint(path)
    n = int(configuration["n"])
    spacing = float(configuration["length_m"])/n
    wall = _wall_parameters(configuration, spacing)
    systems = bcc_four_family_systems()
    strain = np.array([[0.0, .5*gamma], [.5*gamma, 0.0]])
    drivings = _owner_drivings_from_common_stress(
        state, CommonWallDriving(mean_strain=strain), systems, wall)
    rows = {name: [] for name in (
        "raw_stress_Pa", "chemical_backstress_Pa",
        "thermodynamic_stress_Pa", "taylor_resistance_Pa",
        "effective_stress_Pa", "speed_m_s")}
    for owner, driving in zip(state.owners, drivings):
        resolved = resolved_driving_components(owner, driving, systems, (), wall)
        for name in rows:
            rows[name].append(np.asarray(resolved[name], dtype=float))
    values = {name: np.stack(fields, axis=2) for name, fields in rows.items()}
    values["temperature_K"] = np.stack(
        [owner.temperature_K for owner in state.owners], axis=2)[..., None]
    values["supports"] = np.moveaxis(np.asarray(state.supports), 0, 2)
    metadata = {
        "path": str(path.resolve()), "sha256": digest(path), "step": step,
        "physical_time_s": runtime.ledger.physical_time_s, "gamma": gamma,
        "n": n, "spacing_m": spacing, "grain_ids": list(state.grain_ids),
        "source_commit": (provenance or {}).get("source_commit"),
    }
    return values, metadata, wall


def effective_stress(raw, chemical, resistance, regularization):
    thermodynamic = raw-chemical
    smooth = np.sqrt(thermodynamic*thermodynamic+resistance*resistance
                     +regularization*regularization)
    return thermodynamic*(1.0-resistance/smooth)


def speed_from_inputs(raw, chemical, resistance, temperature, wall):
    effective = effective_stress(
        raw, chemical, resistance, wall.taylor_regularization_Pa)
    activation = exp_floor_rate(
        effective, temperature, wall.glide_barrier_eV, wall
    )/wall.attempt_frequency_s
    return (wall.glide_speed_attempt_m_s*activation
            *np.tanh(effective/wall.critical_stress_Pa)), effective


def weighted_relative(left, right, weight) -> float:
    w = np.maximum(np.asarray(weight, dtype=float), 0.0)
    difference = np.sqrt(float(np.sum(w*(left-right)**2, dtype=np.longdouble)))
    scale = max(
        np.sqrt(float(np.sum(w*left*left, dtype=np.longdouble))),
        np.sqrt(float(np.sum(w*right*right, dtype=np.longdouble))), 1e-300)
    return difference/scale


def run(left: Path, right: Path, output: Path,
        half_width: int, quadrature_n: int) -> dict:
    a, ma, wall_a = owner_drivers(left)
    b, mb, wall_b = owner_drivers(right)
    if ma["grain_ids"] != mb["grain_ids"] or ma["step"] != mb["step"]:
        raise ValueError("owner identity or checkpoint step differs")
    parameter_names = (
        "glide_barrier_eV", "attempt_frequency_s", "glide_speed_attempt_m_s",
        "critical_stress_Pa", "exp_a", "exp_n", "exp_floor",
        "taylor_regularization_Pa")
    if any(getattr(wall_a, name) != getattr(wall_b, name)
           for name in parameter_names):
        raise ValueError("rate-map parameters differ")
    pa = {name: project(value, half_width, quadrature_n)
          for name, value in a.items()}
    pb = {name: project(value, half_width, quadrature_n)
          for name, value in b.items()}
    temperature_a = np.broadcast_to(pa["temperature_K"], pa["raw_stress_Pa"].shape)
    temperature_b = np.broadcast_to(pb["temperature_K"], pb["raw_stress_Pa"].shape)
    support_a = np.broadcast_to(pa["supports"][..., None], pa["raw_stress_Pa"].shape)
    support_b = np.broadcast_to(pb["supports"][..., None], pb["raw_stress_Pa"].shape)
    weight = .5*(np.maximum(support_a, 0.0)+np.maximum(support_b, 0.0))
    inputs_a = dict(raw=pa["raw_stress_Pa"],
                    chemical=pa["chemical_backstress_Pa"],
                    resistance=pa["taylor_resistance_Pa"],
                    temperature=temperature_a)
    inputs_b = dict(raw=pb["raw_stress_Pa"],
                    chemical=pb["chemical_backstress_Pa"],
                    resistance=pb["taylor_resistance_Pa"],
                    temperature=temperature_b)
    recomputed_a, effective_a = speed_from_inputs(**inputs_a, wall=wall_a)
    recomputed_b, effective_b = speed_from_inputs(**inputs_b, wall=wall_a)
    projected_actual_a = pa["speed_m_s"]
    projected_actual_b = pb["speed_m_s"]
    baseline_error = weighted_relative(recomputed_a, recomputed_b, weight)
    substitutions = {}
    for name in inputs_a:
        mixed = dict(inputs_a); mixed[name] = inputs_b[name]
        speed, effective = speed_from_inputs(**mixed, wall=wall_a)
        error_to_b = weighted_relative(speed, recomputed_b, weight)
        substitutions[name] = {
            "rate_change_from_left_relative_l2": weighted_relative(
                speed, recomputed_a, weight),
            "remaining_rate_error_to_right_relative_l2": error_to_b,
            "fraction_of_baseline_error_removed": (
                baseline_error-error_to_b)/max(baseline_error, 1e-300),
            "effective_stress_change_from_left_relative_l2": weighted_relative(
                effective, effective_a, weight),
        }
    direct_effective_with_left_temperature = (
        wall_a.glide_speed_attempt_m_s
        *exp_floor_rate(pb["effective_stress_Pa"], temperature_a,
                        wall_a.glide_barrier_eV, wall_a)
        /wall_a.attempt_frequency_s
        *np.tanh(pb["effective_stress_Pa"]/wall_a.critical_stress_Pa))
    result = {
        "schema": "asb-drx-v63-exp-floor-input-attribution-v1",
        "left": ma, "right": mb,
        "common_band_half_width": half_width,
        "common_quadrature_n": quadrature_n,
        "coordinate_and_sign_semantics": (
            "same periodic origin, no translation optimization; signed raw, "
            "chemical, thermodynamic, and effective owner/family stresses are retained"),
        "active_weight_semantics": (
            "mean of nonnegative projected owner supports, broadcast over four families"),
        "projected_primitive_differences": {
            name: weighted_relative(inputs_a[name], inputs_b[name], weight)
            for name in inputs_a},
        "projected_effective_stress_difference": weighted_relative(
            effective_a, effective_b, weight),
        "projected_actual_speed_difference": weighted_relative(
            projected_actual_a, projected_actual_b, weight),
        "recomputed_after_projection_speed_difference": baseline_error,
        "constitutive_projection_error": {
            "left": weighted_relative(recomputed_a, projected_actual_a, weight),
            "right": weighted_relative(recomputed_b, projected_actual_b, weight),
            "semantics": (
                "nonlinear rate of projected primitive drivers versus projection "
                "of the independently computed nonlinear rate"),
        },
        "one_input_at_a_time_substitutions": substitutions,
        "right_effective_stress_only_with_left_temperature": {
            "remaining_rate_error_to_right": weighted_relative(
                direct_effective_with_left_temperature, recomputed_b, weight),
            "rate_change_from_left": weighted_relative(
                direct_effective_with_left_temperature, recomputed_a, weight),
        },
        "all_right_inputs_reproduce_right_rate_relative_l2": weighted_relative(
            recomputed_b, recomputed_b, weight),
        "limitations": (
            "diagnostic reevaluation only; projected primitives are not an "
            "admissible production state and no counterfactual is published"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--half-width", type=int, default=15)
    parser.add_argument("--quadrature-n", type=int, default=96)
    args = parser.parse_args()
    result = run(args.left, args.right, args.output,
                 args.half_width, args.quadrature_n)
    print(json.dumps({
        "sha256": digest(args.output),
        "projected_effective_stress_difference": result[
            "projected_effective_stress_difference"],
        "projected_actual_speed_difference": result[
            "projected_actual_speed_difference"],
        "constitutive_projection_error": result["constitutive_projection_error"],
        "substitutions": result["one_input_at_a_time_substitutions"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
