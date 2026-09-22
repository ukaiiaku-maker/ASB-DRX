"""State-sampled shared-clock kinetics for straight subcell rectangle faces."""

from __future__ import annotations

import numpy as np

from .arrhenius_kinetics import activated_rate_array_s, exp_floor_enthalpy_j
from .common_tensorial_wall import resolved_driving_components
from .extensive_wall import (
    KB_J_K, extensive_wall_energy_components_J_m3,
    ordered_gradient_increment_J_m3_cells,
)
from .subcell_segment_geometry import propose_subcell_x_face_moves
from .v24_mechanical_wall import (
    _elastic_energy_sum_J_m3_cells, accepted_subcell_x_faces_shared_clock,
)


def _periodic_bilinear(field, points_m, spacing_m):
    """Sample a cell-centred periodic field at physical xy points."""
    value = np.asarray(field, dtype=float)
    points = np.asarray(points_m, dtype=float)
    if value.ndim < 2 or points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("periodic sampling requires a grid field and xy points")
    nx, ny = value.shape[:2]; spacing = float(spacing_m)
    coordinate = points/spacing-.5
    i0 = np.floor(coordinate[:, 0]).astype(int)
    j0 = np.floor(coordinate[:, 1]).astype(int)
    fx = coordinate[:, 0]-i0; fy = coordinate[:, 1]-j0
    i1 = (i0+1) % nx; j1 = (j0+1) % ny
    i0 %= nx; j0 %= ny
    trailing = (1,)*(value.ndim-2)
    wx = fx.reshape((-1,)+trailing); wy = fy.reshape((-1,)+trailing)
    return ((1-wx)*(1-wy)*value[i0, j0]
            +wx*(1-wy)*value[i1, j0]
            +(1-wx)*wy*value[i0, j1]
            +wx*wy*value[i1, j1])


def sample_straight_x_face_state(state, face, drive, family, quadrature_order):
    """Gauss sample actual stress and temperature along one physical face."""
    geometry = state.subcell_geometry
    if geometry is None or face not in ("lower_x", "upper_x"):
        raise ValueError("state sampling requires a named rectangle x face")
    order = int(quadrature_order)
    if order < 2:
        raise ValueError("face quadrature requires at least two sites")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    y0, y1 = float(geometry.lower_left_m[1]), float(geometry.upper_right_m[1])
    x = (float(geometry.lower_left_m[0]) if face == "lower_x"
         else float(geometry.upper_right_m[0]))
    y = .5*(y1-y0)*nodes+.5*(y1+y0)
    points = np.stack((np.full_like(y, x), y), axis=1)
    normalized_weights = .5*weights
    stress = _periodic_bilinear(
        np.abs(np.asarray(drive["effective_stress_Pa"])[..., int(family)]),
        points, geometry.spacing_m)
    temperature = _periodic_bilinear(
        state.common.temperature_K, points, geometry.spacing_m)
    return {
        "points_m": points,
        "normalized_site_weights": normalized_weights,
        "effective_stress_Pa": stress,
        "temperature_K": temperature,
    }


def _face_complete_affinity(
        state, face, direction, probe_m, systems, topologies, driving,
        common_parameters, extensive_parameters, kinetics):
    geometry = state.subcell_geometry
    lower = float(direction*probe_m) if face == "lower_x" else 0.0
    upper = float(direction*probe_m) if face == "upper_x" else 0.0
    proposal = propose_subcell_x_face_moves(
        geometry, state.density, state.reservoir_alignment, state.common,
        systems, lower_displacement_m=lower,
        upper_displacement_m=upper)
    candidate_geometry, inventory, _, common, ledger = proposal
    zero = np.zeros(state.common.orientation_rad.shape+(3, 3))
    before = extensive_wall_energy_components_J_m3(
        state.density, systems, topologies, state.common.orientation_rad,
        zero, extensive_parameters)
    after = extensive_wall_energy_components_J_m3(
        inventory, systems, topologies, common.orientation_rad,
        zero, extensive_parameters)
    components = {
        name: float(np.sum(np.asarray(after[name])-before[name],
                           dtype=np.longdouble))
        for name in before if name != "total"
    }
    components["ordered_gradient"] = ordered_gradient_increment_J_m3_cells(
        state.density, inventory, extensive_parameters)
    wall_delta = float(sum(components.values()))
    elastic_before = _elastic_energy_sum_J_m3_cells(
        state.common, state.common.beta_p, driving, common_parameters)
    elastic_after = _elastic_energy_sum_J_m3_cells(
        common, common.beta_p, driving, common_parameters)
    elastic_delta = (0.0 if elastic_before is None
                     else float(elastic_after-elastic_before))
    height = float(geometry.upper_right_m[1]-geometry.lower_left_m[1])
    coefficient = (abs(kinetics.exchange_stoichiometry_defects_per_atom
                       *float(geometry.burgers_vector_m[2])*height)
                   /kinetics.atomic_volume_m3_per_atom)
    event_count = coefficient*probe_m
    if event_count <= 0.0:
        raise ValueError("state-dependent climb requires a species event measure")
    # The sign convention follows the existing shared-face exchange ledger.
    signed_count = ((-1.0 if face == "lower_x" else 1.0)
                    *coefficient*direction*probe_m)
    chemical_work_J = kinetics.chemical_potential_J_per_defect*signed_count
    cell_volume = (float(geometry.spacing_m)**2
                   *float(geometry.section_thickness_m))
    complete_delta = wall_delta+elastic_delta-chemical_work_J/cell_volume
    available_per_event = -complete_delta*cell_volume/event_count
    return {
        "probe_displacement_m": float(direction*probe_m),
        "physical_event_count": event_count,
        "complete_energy_change_J_m3_cells": complete_delta,
        "available_energy_per_event_J": available_per_event,
        "wall_energy_change_J_m3_cells": wall_delta,
        "elastic_energy_change_J_m3_cells": elastic_delta,
        "chemical_reservoir_work_J": chemical_work_J,
        "line_surface_event_compatibility_passed": ledger[
            "line_surface_event_compatibility_passed"],
    }


def state_dependent_face_rates(
        state, directions, systems, topologies, driving, common_parameters,
        extensive_parameters, kinetics, *, quadrature_order=16,
        probe_displacement_m=None):
    """Evaluate nonlinear site rates and one rigid velocity for each face."""
    if kinetics.atomic_volume_m3_per_atom <= 0.0:
        raise ValueError("state-dependent shared kinetics currently supports climb")
    geometry = state.subcell_geometry
    event_jump = (float(kinetics.physical_event_jump_m)
                  if kinetics.physical_event_jump_m > 0.0
                  else float(common_parameters.burgers_m))
    probe = (event_jump if probe_displacement_m is None
             else float(probe_displacement_m))
    drive = resolved_driving_components(
        state.common, driving, systems, topologies, common_parameters)
    family = int(geometry.family)
    result = {}
    for face in ("lower_x", "upper_x"):
        direction = float(np.sign(directions[face]))
        if direction == 0.0:
            raise ValueError("each state-dependent face needs a direction")
        sample = sample_straight_x_face_state(
            state, face, drive, family, quadrature_order)
        enthalpy = exp_floor_enthalpy_j(
            sample["effective_stress_Pa"], kinetics.enthalpy_J,
            kinetics.critical_stress_Pa, kinetics.exp_a, kinetics.exp_n,
            kinetics.exp_floor)
        unbiased = activated_rate_array_s(
            kinetics.process, enthalpy, sample["temperature_K"])
        affinity = _face_complete_affinity(
            state, face, direction, probe, systems, topologies, driving,
            common_parameters, extensive_parameters, kinetics)
        activity = np.maximum(np.tanh(
            affinity["available_energy_per_event_J"]
            /(2*KB_J_K*sample["temperature_K"])), 0.0)
        local_rates = unbiased*activity
        weights = sample["normalized_site_weights"]
        generalized_rate = float(np.sum(weights*local_rates))
        result[face] = {
            **affinity,
            "quadrature_order": int(quadrature_order),
            "generalized_rate_s": generalized_rate,
            "generalized_velocity_m_s": generalized_rate*event_jump,
            "site_rate_minimum_s": float(np.min(local_rates)),
            "site_rate_maximum_s": float(np.max(local_rates)),
            "site_rate_weighted_mean_s": generalized_rate,
            "rate_at_weighted_mean_inputs_s": float(activated_rate_array_s(
                kinetics.process,
                exp_floor_enthalpy_j(
                    np.sum(weights*sample["effective_stress_Pa"]),
                    kinetics.enthalpy_J, kinetics.critical_stress_Pa,
                    kinetics.exp_a, kinetics.exp_n, kinetics.exp_floor),
                np.sum(weights*sample["temperature_K"]))*max(float(np.tanh(
                    affinity["available_energy_per_event_J"]/(2*KB_J_K*np.sum(
                        weights*sample["temperature_K"])))), 0.0)),
            "stress_minimum_Pa": float(np.min(sample["effective_stress_Pa"])),
            "stress_maximum_Pa": float(np.max(sample["effective_stress_Pa"])),
            "temperature_minimum_K": float(np.min(sample["temperature_K"])),
            "temperature_maximum_K": float(np.max(sample["temperature_K"])),
        }
    return result


def accepted_state_dependent_subcell_x_faces(
        state, directions, systems, topologies, driving, common_parameters,
        extensive_parameters, kinetics, dt_s, *, quadrature_order=16):
    """Advance both faces on one event-split clock with refreshed site rates."""
    remaining = float(dt_s); current = state; elapsed = 0.0; substeps = []
    event_jump = (float(kinetics.physical_event_jump_m)
                  if kinetics.physical_event_jump_m > 0.0
                  else float(common_parameters.burgers_m))
    # A quarter-cell is a numerical rate-refresh cap, not an event length or
    # changed physical coefficient.  The underlying transaction retains its
    # own (looser) hard cap.
    maximum = .25*kinetics.maximum_extent_per_step*float(
        state.subcell_geometry.spacing_m)
    for index in range(1024):
        if remaining <= 64*np.finfo(float).eps*max(float(dt_s), 1e-300):
            return current, {
                "operator": "state_dependent_shared_x_faces",
                "accepted": True,
                "classification": "ADMISSIBLE_STATE_DEPENDENT_SHARED_CLOCK",
                "requested_duration_s": float(dt_s),
                "consumed_duration_s": elapsed,
                "substeps": substeps,
                "quadrature_order": int(quadrature_order),
                "rate_refresh_count": len(substeps),
                "clock_combination_rule": "first_numerical_cap_then_refresh",
                "rate_refresh_displacement_cap_m": maximum,
            }
        rates = state_dependent_face_rates(
            current, directions, systems, topologies, driving,
            common_parameters, extensive_parameters, kinetics,
            quadrature_order=quadrature_order)
        maximum_rate = max(row["generalized_rate_s"] for row in rates.values())
        if maximum_rate <= 0.0:
            return current, {
                "operator": "state_dependent_shared_x_faces",
                "accepted": False,
                "classification": "STATE_DEPENDENT_ALL_FACES_STALLED",
                "requested_duration_s": float(dt_s),
                "consumed_duration_s": elapsed,
                "substeps": substeps,
                "last_face_rates": rates,
            }
        if any(row["generalized_rate_s"] <= 0.0 for row in rates.values()):
            return current, {
                "operator": "state_dependent_shared_x_faces",
                "accepted": False,
                "classification": "PARTIAL_FACE_STALL_REQUIRES_ONE_FACE_OWNER",
                "requested_duration_s": float(dt_s),
                "consumed_duration_s": elapsed,
                "substeps": substeps,
                "last_face_rates": rates,
            }
        sub_dt = min(remaining, maximum/(maximum_rate*event_jump))
        events = [{
            "face": face,
            "proposed_displacement_m": float(np.sign(directions[face]))*maximum,
            "fixed_rate_s": rates[face]["generalized_rate_s"],
        } for face in ("lower_x", "upper_x")]
        candidate, ledger = accepted_subcell_x_faces_shared_clock(
            current, events, systems, topologies, driving, common_parameters,
            extensive_parameters, kinetics, sub_dt)
        if not ledger["accepted"]:
            return current, {
                "operator": "state_dependent_shared_x_faces",
                "accepted": False,
                "classification": "JOINT_STATE_DEPENDENT_SUBSTEP_REJECTED",
                "requested_duration_s": float(dt_s),
                "consumed_duration_s": elapsed,
                "substeps": substeps,
                "last_face_rates": rates,
                "rejected_joint_ledger": ledger,
            }
        substeps.append({
            "substep": index+1,
            "duration_s": ledger["consumed_duration_s"],
            "face_rates": rates,
            "joint_complete_energy_change_J_m3_cells": ledger[
                "complete_energy_change_J_m3_cells"],
            "face_displacements_m": ledger["face_displacements_m"],
            "heat_plus_complete_energy_residual_J_m3_cells": ledger[
                "heat_plus_complete_energy_residual_J_m3_cells"],
        })
        current = candidate; elapsed += sub_dt; remaining -= sub_dt
    raise RuntimeError("state-dependent shared clock exceeded substep bound")
