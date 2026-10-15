"""Atomic lattice-orientation relaxation driven by conserved ordered wall line."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np
from scipy import ndimage

from .complete_multigrain_energy import evaluate_complete_multigrain_energy
from .orientation_wall_compatibility import (
    evaluate_orientation_wall_compatibility,
)
from .structural_boundary import (
    StructuralBoundaryParameters, exp_floor_event_rate_Hz,
)


@dataclass(frozen=True)
class WallOrientationRelaxationParameters:
    activation: StructuralBoundaryParameters = StructuralBoundaryParameters(
        activation_enthalpy_eV=.70, barrier_floor_eV=.05,
        attempt_frequency_Hz=1.0e7, activation_segment_m=2.5e-9)
    maximum_fraction_per_interval: float = .05
    owner_support_threshold: float = .75
    boundary_clearance_m: float = 2.0e-7

    def __post_init__(self):
        if (not math.isfinite(self.maximum_fraction_per_interval)
                or not 0.0 < self.maximum_fraction_per_interval <= 1.0):
            raise ValueError("orientation relaxation fraction must lie in (0,1]")
        if not 0.0 < self.owner_support_threshold < 1.0:
            raise ValueError("owner support threshold must lie in (0,1)")
        if not math.isfinite(self.boundary_clearance_m) or self.boundary_clearance_m < 0.0:
            raise ValueError("boundary clearance must be finite and nonnegative")


@dataclass(frozen=True)
class WallOrientationRelaxationDecision:
    accepted: bool
    classification: str
    candidate_state: object
    committed_state: object
    owner_fraction_by_id: dict[int, float]
    owner_affinity_J_m_by_id: dict[int, float]
    owner_compatibility: dict[int, object]
    delta_helmholtz_J: float
    generated_heat_J: float
    first_law_residual_J: float
    relative_first_law_residual: float
    heat_source_J_by_cell: np.ndarray


def _owner_domain(support, spacing_m, parameters):
    domain = np.asarray(support) > parameters.owner_support_threshold
    cells = int(math.ceil(parameters.boundary_clearance_m/spacing_m))
    if cells > 0:
        domain = ndimage.binary_erosion(domain, iterations=cells)
    return domain


def evaluate_wall_orientation_relaxation(
        state, *, interval_s, spacing_m, represented_thickness_m, systems,
        wall_parameters, energy_kwargs=None,
        parameters=WallOrientationRelaxationParameters(),
        maximum_relative_first_law_residual=.05):
    """Price and commit common-axis rotation without changing line or beta/Nye."""
    dt = float(interval_s)
    spacing = float(spacing_m)
    thickness = float(represented_thickness_m)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("positive finite relaxation interval required")
    if wall_parameters.wall_nye_match_coefficient_J_m <= 0.0:
        raise ValueError("wall orientation relaxation requires the match energy")
    options = {} if energy_kwargs is None else dict(energy_kwargs)
    common = dict(spacing_m=spacing, represented_thickness_m=thickness,
                  wall_parameters=wall_parameters, **options)
    before = evaluate_complete_multigrain_energy(state, **common)
    cell_volume = spacing**2*thickness
    fractions = {}
    affinities = {}
    compatibility = {}
    targets = {}
    domains = {}
    for grain_id, support, owner in zip(
            state.grain_ids, state.supports, state.owners):
        domain = _owner_domain(support, spacing, parameters)
        domains[int(grain_id)] = domain
        if np.count_nonzero(domain) < 2:
            fractions[int(grain_id)] = 0.0
            affinities[int(grain_id)] = 0.0
            continue
        diagnostic = evaluate_orientation_wall_compatibility(
            owner, systems, domain, spacing)
        compatibility[int(grain_id)] = diagnostic
        increment = np.arctan2(
            np.sin(diagnostic.reconstructed_orientation_rad
                   -np.asarray(owner.orientation_rad)),
            np.cos(diagnostic.reconstructed_orientation_rad
                   -np.asarray(owner.orientation_rad)))
        target = np.asarray(owner.orientation_rad).copy()
        target[domain] += parameters.maximum_fraction_per_interval*increment[domain]
        targets[int(grain_id)] = target
        trial_owners = tuple(
            replace(item, orientation_rad=target) if gid == grain_id else item
            for gid, item in zip(state.grain_ids, state.owners))
        trial = replace(state, owners=trial_owners)
        trial.validate()
        trial_energy = evaluate_complete_multigrain_energy(trial, **common)
        ordered_density = (np.asarray(owner.wall_order)
                           *np.sum(owner.wall_plus_m2+owner.wall_minus_m2,
                                  axis=2))
        participating_line_m = float(np.sum(
            np.asarray(support)*domain*ordered_density,
            dtype=np.longdouble)*cell_volume)
        affinity = ((before.helmholtz_J-trial_energy.helmholtz_J)
                    /participating_line_m if participating_line_m > 0.0 else 0.0)
        temperature = float(np.mean(np.asarray(owner.temperature_K)[domain]))
        rate = float(exp_floor_event_rate_Hz(
            max(affinity, 0.0), temperature, parameters.activation))
        fraction = (min(-math.expm1(-rate*dt),
                        parameters.maximum_fraction_per_interval)
                    if participating_line_m > 0.0 and affinity >= 0.0 else 0.0)
        fractions[int(grain_id)] = fraction
        affinities[int(grain_id)] = affinity

    owners = []
    activity = np.zeros(state.supports.shape[1:])
    for grain_id, support, owner in zip(
            state.grain_ids, state.supports, state.owners):
        fraction = fractions[int(grain_id)]
        domain = domains[int(grain_id)]
        if fraction > 0.0 and int(grain_id) in compatibility:
            target = compatibility[int(grain_id)].reconstructed_orientation_rad
            increment = np.arctan2(
                np.sin(target-np.asarray(owner.orientation_rad)),
                np.cos(target-np.asarray(owner.orientation_rad)))
            orientation = np.asarray(owner.orientation_rad).copy()
            orientation[domain] += fraction*increment[domain]
            owners.append(replace(owner, orientation_rad=orientation))
            activity += (np.asarray(support)*domain*np.asarray(owner.wall_order)
                         *np.sum(owner.wall_plus_m2+owner.wall_minus_m2, axis=2))
        else:
            owners.append(owner)
    cold = replace(state, owners=tuple(owners))
    cold.validate()
    cold_energy = evaluate_complete_multigrain_energy(cold, **common)
    delta = cold_energy.helmholtz_J-before.helmholtz_J
    resolution = 4096*np.finfo(float).eps*max(
        abs(before.helmholtz_J), abs(cold_energy.helmholtz_J), 1e-300)
    active = any(value > 0.0 for value in fractions.values())
    downhill = delta <= resolution
    heat = max(-delta, 0.0) if active and downhill else 0.0
    heat_cells = np.zeros(state.supports.shape[1:])
    if heat > 0.0 and np.sum(activity) > 0.0:
        heat_cells = heat*activity/float(np.sum(activity, dtype=np.longdouble))
        rise = heat_cells/(wall_parameters.volumetric_heat_capacity_J_m3_K
                          *cell_volume)
        committed = replace(cold, owners=tuple(
            replace(owner, temperature_K=np.asarray(owner.temperature_K)+rise)
            for owner in cold.owners))
        committed.validate()
        hot = evaluate_complete_multigrain_energy(committed, **common)
        residual = hot.internal_J-before.internal_J
        scale = max(abs(delta), heat, resolution, 1e-300)
        relative = abs(residual)/scale
    else:
        committed = state
        residual = relative = 0.0
    accepted = bool(active and downhill
                    and relative <= maximum_relative_first_law_residual)
    if not active:
        classification = "INACTIVE_NO_DOWNHILL_COMPATIBLE_ROTATION"
    elif not downhill:
        classification = "REJECTED_COMPLETE_ENERGY_UPHILL"
    elif relative > maximum_relative_first_law_residual:
        classification = "REJECTED_COMPLETE_FIRST_LAW"
    else:
        classification = "ACCEPTED_ORDERED_WALL_ORIENTATION_RELAXATION"
    return WallOrientationRelaxationDecision(
        accepted, classification, cold, committed if accepted else state,
        fractions, affinities, compatibility, delta,
        heat if accepted else 0.0, residual if accepted else 0.0,
        relative if accepted else 0.0,
        heat_cells if accepted else np.zeros_like(heat_cells))
