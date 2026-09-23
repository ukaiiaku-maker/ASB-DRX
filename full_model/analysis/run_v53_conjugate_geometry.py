#!/usr/bin/env python3
"""Qualify full-tensor geometry mechanics and a recurrent common clock."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from full_model.analysis.run_v39_common_horizon import atomic_json
from full_model.analysis.run_v50_subcell_qualification import fixture
from full_model.production.arrhenius_kinetics import (
    exp_floor_activation_volume_m3,
)
from full_model.production.common_tensorial_wall import CommonWallDriving
from full_model.production.coupled_geometry_clock import (
    accepted_common_clock_subcell_macro,
)
from full_model.production.extensive_wall import (
    extensive_wall_energy_components_J_m3,
)
from full_model.production.nonlocal_elasticity import (
    solve_periodic_eigenstrain_3d_z_invariant,
    z_invariant_equilibrium_residual,
)
from full_model.production.state_dependent_subcell import (
    state_dependent_face_rates,
)
from full_model.production.subcell_segment_geometry import (
    propose_subcell_x_face_moves,
)
from full_model.production.v24_mechanical_wall import (
    _elastic_energy_sum_J_m3_cells, mechanical_checkpoint_arrays,
    mechanical_from_checkpoint_arrays,
)
from tests.test_v50_production_subcell_geometry import intrinsic_kinetics


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative_l2(left, right):
    left = np.asarray(left); right = np.asarray(right)
    return float(np.linalg.norm(left-right)/max(np.linalg.norm(right), 1e-300))


def full_driving(base):
    mean = np.zeros((3, 3)); mean[:2, :2] = base.mean_strain
    # Manufactured, strain-controlled out-of-plane shear verifies the missing
    # conjugate channel.  It is not a calibrated production migration load.
    mean[0, 2] = mean[2, 0] = 7.5e-4
    mean[1, 2] = mean[2, 1] = -4.0e-4
    return CommonWallDriving(
        mean_strain=base.mean_strain,
        fixed_eigenstrain=base.fixed_eigenstrain,
        full_tensor_z_invariant_enabled=True,
        mean_strain_3d=mean)


def elastic_energy(state, driving, parameters):
    return _elastic_energy_sum_J_m3_cells(
        state.common, state.common.beta_p, driving, parameters)


def state_metrics(state, data):
    geometry = state.subcell_geometry
    density = state.density
    return {
        "lower_x_m": float(geometry.lower_left_m[0]),
        "upper_x_m": float(geometry.upper_right_m[0]),
        "width_m": float(geometry.upper_right_m[0]-geometry.lower_left_m[0]),
        "accepted_geometry_event_count": int(geometry.accepted_event_count),
        "mean_temperature_K": float(np.mean(state.common.temperature_K)),
        "temperature_range_K": float(np.ptp(state.common.temperature_K)),
        "orientation_min_rad": float(np.min(state.common.orientation_rad)),
        "orientation_max_rad": float(np.max(state.common.orientation_rad)),
        "beta_p_rms": float(np.sqrt(np.mean(state.common.beta_p**2))),
        "family_nye_rms_m-1": float(np.sqrt(np.mean(
            state.common.family_nye_m1**2))),
        "signed_reservoir_integrals": {
            name: float(np.sum(getattr(density, name), dtype=np.longdouble)
                        *data[9]**2)
            for name in density.__dataclass_fields__
        },
    }


def geometry_ledger_totals(audit, cell_volume):
    events = 0.0; signed = 0.0; complete = 0.0; elastic = 0.0
    residual = 0.0; displacements = {"lower_x": 0.0, "upper_x": 0.0}
    for exposure in audit["geometry"]["substeps"]:
        ledger = exposure["ledger"]
        for row in ledger.get("substeps", []):
            events += sum(float(value) for value in row[
                "face_physical_event_counts"].values())
            signed += float(row["signed_material_exchange_count"])
            complete += float(row["joint_complete_energy_change_J_m3_cells"])
            elastic += float(row["elastic_energy_change_J_m3_cells"])
            residual += float(row["heat_plus_complete_energy_residual_J_m3_cells"])
            for face, value in row["face_displacements_m"].items():
                displacements[face] += float(value)
            # The state-dependent summary stores the probe affinity per face;
            # accepted physical exchange is recovered below from displacement.
    return {
        "face_displacements_m": displacements,
        "complete_energy_change_J": complete*cell_volume,
        "elastic_energy_change_J": elastic*cell_volume,
        "heat_plus_complete_energy_residual_J": residual*cell_volume,
        "physical_event_count": events,
        "signed_material_exchange_count": signed,
    }


def run_common_clock(n, duration):
    state, data = fixture(n)
    driving = full_driving(data[1])
    directions = {"lower_x": 1.0, "upper_x": -1.0}
    kinetics = intrinsic_kinetics()
    before = state_metrics(state, data)
    started = time.perf_counter()
    final, audit = accepted_common_clock_subcell_macro(
        state, duration, directions, data[4], data[5], driving, data[3],
        data[6], data[7], data[8], kinetics, quadrature_order=16)
    wall = time.perf_counter()-started
    if not audit["accepted"]:
        raise RuntimeError(f"common-clock candidate rejected: {audit['classification']}")
    final.validate(data[4], data[5])
    after = state_metrics(final, data)
    cell_volume = data[9]**2*float(state.subcell_geometry.section_thickness_m)
    return state, final, data, driving, audit, {
        "grid": n, "duration_s": duration, "wall_seconds": wall,
        "before": before, "after": after,
        "lower_displacement_m": after["lower_x_m"]-before["lower_x_m"],
        "upper_displacement_m": after["upper_x_m"]-before["upper_x_m"],
        "lower_displacement_over_400nm": (
            after["lower_x_m"]-before["lower_x_m"])/4e-7,
        "upper_displacement_over_400nm": (
            after["upper_x_m"]-before["upper_x_m"])/4e-7,
        "operator_times_s": {
            "physical_elapsed": audit["physical_elapsed_time_s"],
            "bulk_exposure": audit["bulk_operator_exposure_s"],
            "geometry_exposure": audit["geometry_operator_exposure_s"],
        },
        "geometry_ledger": geometry_ledger_totals(audit, cell_volume),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=2e-9)
    args = parser.parse_args()
    state, data = fixture(32)
    driving = full_driving(data[1]); old_driving = data[1]
    kinetics = intrinsic_kinetics()
    probe = 2e-10
    proposals = {
        label: propose_subcell_x_face_moves(
            state.subcell_geometry, state.density, state.reservoir_alignment,
            state.common, data[4], lower_displacement_m=0.0,
            upper_displacement_m=value)[3]
        for label, value in (("plus", probe), ("minus", -probe))}
    old_delta = elastic_energy(proposals["plus"], old_driving, data[6])-elastic_energy(
        state, old_driving, data[6])
    full_delta = elastic_energy(proposals["plus"], driving, data[6])-elastic_energy(
        state, driving, data[6])
    finite_derivative = (elastic_energy(proposals["plus"], driving, data[6])
                         -elastic_energy(proposals["minus"], driving, data[6]))/(2*probe)
    beta = state.common.beta_p
    eigen = .5*(beta+np.swapaxes(beta, -1, -2))
    stress, strain = solve_periodic_eigenstrain_3d_z_invariant(
        eigen, driving.mean_strain_3d, data[6].spacing_m,
        data[6].c11_Pa, data[6].c12_Pa, data[6].c44_Pa)
    dbeta = (proposals["plus"].beta_p-proposals["minus"].beta_p)/(2*probe)
    virtual_work = -float(np.sum(stress*.5*(dbeta+np.swapaxes(dbeta, -1, -2))))
    equilibrium = z_invariant_equilibrium_residual(stress, data[6].spacing_m)
    equilibrium_relative = float(
        np.linalg.norm(equilibrium)/(np.linalg.norm(stress)/data[6].spacing_m))
    rates = state_dependent_face_rates(
        state, {"lower_x": 1.0, "upper_x": -1.0}, data[4], data[5],
        driving, data[6], data[7], kinetics, quadrature_order=64)
    activation = {}
    for face, row in rates.items():
        stress_input = .5*(row["stress_minimum_Pa"]+row["stress_maximum_Pa"])
        activation[face] = {
            "signed_complete_affinity_available_J_per_event": row[
                "available_energy_per_event_J"],
            "geometric_event_measure": "one exchanged species per climb event per face",
            "stress_like_barrier_input_Pa": stress_input,
            "stress_like_input_scope": "absolute effective glide RSS used as an uncalibrated non-glide comparator",
            "activation_volume_minus_dG_dstress_m3": (
                exp_floor_activation_volume_m3(
                    stress_input, kinetics.enthalpy_J,
                    kinetics.critical_stress_Pa, kinetics.exp_a,
                    kinetics.exp_n, kinetics.exp_floor)),
            "generalized_rate_s-1": row["generalized_rate_s"],
        }

    initial, whole, whole_data, whole_drive, whole_audit, whole_row = (
        run_common_clock(32, args.duration_s))
    split = initial; split_audits = []
    for _ in range(2):
        split, audit = accepted_common_clock_subcell_macro(
            split, .5*args.duration_s,
            {"lower_x": 1.0, "upper_x": -1.0}, whole_data[4], whole_data[5],
            whole_drive, whole_data[3], whole_data[6], whole_data[7],
            whole_data[8], kinetics, quadrature_order=16)
        if not audit["accepted"]:
            raise RuntimeError("subdivided common-clock path rejected")
        split_audits.append(audit)
    whole_split = {
        "lower_face_displacement_relative_difference": abs(
            (split.subcell_geometry.lower_left_m[0]-initial.subcell_geometry.lower_left_m[0])
            -(whole.subcell_geometry.lower_left_m[0]-initial.subcell_geometry.lower_left_m[0]))/max(
                abs(whole.subcell_geometry.lower_left_m[0]-initial.subcell_geometry.lower_left_m[0]), 1e-300),
        "upper_face_displacement_relative_difference": abs(
            (split.subcell_geometry.upper_right_m[0]-initial.subcell_geometry.upper_right_m[0])
            -(whole.subcell_geometry.upper_right_m[0]-initial.subcell_geometry.upper_right_m[0]))/max(
                abs(whole.subcell_geometry.upper_right_m[0]-initial.subcell_geometry.upper_right_m[0]), 1e-300),
        "beta_p_relative_l2": relative_l2(
            whole.common.beta_p, split.common.beta_p),
        "temperature_relative_l2": relative_l2(
            whole.common.temperature_K, split.common.temperature_K),
    }

    first, first_audit = accepted_common_clock_subcell_macro(
        initial, .5*args.duration_s,
        {"lower_x": 1.0, "upper_x": -1.0}, whole_data[4], whole_data[5],
        whole_drive, whole_data[3], whole_data[6], whole_data[7],
        whole_data[8], kinetics, quadrature_order=16)
    arrays = mechanical_checkpoint_arrays(first)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.checkpoint, **arrays)
    with np.load(args.checkpoint, allow_pickle=False) as archive:
        restored = mechanical_from_checkpoint_arrays(
            {name: archive[name] for name in archive.files},
            whole_data[4], whole_data[5])
    continuous, _ = accepted_common_clock_subcell_macro(
        first, .5*args.duration_s,
        {"lower_x": 1.0, "upper_x": -1.0}, whole_data[4], whole_data[5],
        whole_drive, whole_data[3], whole_data[6], whole_data[7],
        whole_data[8], kinetics, quadrature_order=16)
    restarted, _ = accepted_common_clock_subcell_macro(
        restored, .5*args.duration_s,
        {"lower_x": 1.0, "upper_x": -1.0}, whole_data[4], whole_data[5],
        whole_drive, whole_data[3], whole_data[6], whole_data[7],
        whole_data[8], kinetics, quadrature_order=16)
    restart_exact = all(np.array_equal(
        value, mechanical_checkpoint_arrays(restarted)[name])
        for name, value in mechanical_checkpoint_arrays(continuous).items())

    _, _, _, _, _, grid64 = run_common_clock(64, args.duration_s)
    initial_old = elastic_energy(state, old_driving, data[6])
    initial_full = elastic_energy(state, driving, data[6])
    zero_target = np.zeros(state.common.orientation_rad.shape+(3, 3))
    wall_components = extensive_wall_energy_components_J_m3(
        state.density, data[4], data[5], state.common.orientation_rad,
        zero_target, data[7])
    payload = {
        "schema": "asb-drx/v53/conjugate-geometry-common-clock/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "mechanical_model": {
            "spatial_grid": "two-dimensional periodic xy",
            "displacement_components": 3,
            "constraint": "z-invariant fluctuations with prescribed mean E33=0 plane strain",
            "energy": "0.5*(Ebar+sym(grad_xy u)-sym(beta_p)-e_fixed):C:(same)",
            "default_enabled": False,
        },
        "source_derived_old_operator_null": {
            "classification": "IN_PLANE_ELASTIC_ENERGY_HAS_A_NULL_FOR_THE_RETAINED_XY_SURFACE_GEOMETRY_INCREMENT",
            "old_in_plane_elastic_increment_J_m3_cells": old_delta,
            "full_tensor_elastic_increment_J_m3_cells": full_delta,
        },
        "mechanical_verification": {
            "finite_difference_shape_derivative_J_m3_cells_per_m": finite_derivative,
            "negative_virtual_work_J_m3_cells_per_m": virtual_work,
            "relative_conjugacy_residual": abs(finite_derivative-virtual_work)/max(
                abs(virtual_work), 1e-300),
            "equilibrium_relative_residual": equilibrium_relative,
            "full_tensor_stress_components_max_abs_Pa": {
                f"sigma_{i+1}{j+1}": float(np.max(np.abs(stress[..., i, j])))
                for i in range(3) for j in range(i, 3)},
        },
        "representation_change": {
            "old_elastic_energy_J_m3_cells": initial_old,
            "new_full_tensor_elastic_energy_J_m3_cells": initial_full,
            "difference_J_m3_cells": initial_full-initial_old,
            "deposited_as_heat": False,
            "classification": "MODEL_REPRESENTATION_REPRICING_NOT_PHYSICAL_RELEASE",
        },
        "energy_partition": {
            "full_tensor_elastic_energy_J_m3_cells": initial_full,
            "stored_line_and_ordering_components_J_m3_cells": {
                name: float(np.sum(value, dtype=np.longdouble))
                for name, value in wall_components.items()},
            "outcome_dependent_self_energy_subtracted": False,
        },
        "activation_coordinate": activation,
        "common_clock_n32": whole_row,
        "whole_vs_two_half_common_clock": whole_split,
        "checkpoint_restart": {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": digest(args.checkpoint),
            "first_half_accepted": bool(first_audit["accepted"]),
            "exact": bool(restart_exact),
        },
        "fixed_physical_scale_grid_comparison": {
            "n32": {key: whole_row[key] for key in (
                "lower_displacement_m", "upper_displacement_m")},
            "n64": {key: grid64[key] for key in (
                "lower_displacement_m", "upper_displacement_m")},
            "relative_differences": {
                face: abs(whole_row[f"{face}_displacement_m"]
                          -grid64[f"{face}_displacement_m"])/max(
                              abs(grid64[f"{face}_displacement_m"]), 1e-300)
                for face in ("lower", "upper")},
        },
        "chemical_work_J_per_defect": 0.0,
        "artificial_scalar_migration_pressure": False,
        "prepared_geometry_not_spontaneously_formed": True,
        "drx_claimed": False,
        "persistent_lagb_claimed": False,
        "strict_asb_claimed": False,
        "material_calibration_claimed": False,
    }
    atomic_json(args.output, payload)
    print(json.dumps({"output": str(args.output), "sha256": digest(args.output),
                      "old_null": old_delta, "full_increment": full_delta,
                      "restart_exact": restart_exact}, sort_keys=True))


if __name__ == "__main__":
    main()
